"""LLM client wrapper for the OpenAI-compatible relay (1m1ng).

Single entry point: `match_logo_in_photo(logo_path, photo_path)` → MatchResult.
Prompt text lives in `prompts/v_<n>.md`; the active version is picked by env var
`LINEBASE_PROMPT_VERSION` (default = newest file under prompts/).

Multi-provider note: pass `model=` and/or `provider=` to override the default
provider routing — used by the benchmark script. Without overrides we fall back
to `Settings.primary` for full backward compatibility.

Per-call timeout: every call goes through `_create_completion` which passes
`timeout=` to the OpenAI SDK. Default is `LINEBASE_LLM_TIMEOUT_S` env (90 s).
A stalled provider request raises `openai.APITimeoutError` after that budget,
which the pipeline catches → row marked as needs_review / failed evidence,
and the rest of the batch continues. See spec/backend/llm-gotchas.md
"Doubao Seed 2.0 — real-world stalls" for the original bug.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from linebase.config import ProviderConfig, Settings


def _default_timeout_s() -> float:
    """Per-call OpenAI SDK timeout. Configurable via LINEBASE_LLM_TIMEOUT_S env.

    Default 90 s — covers Doubao Seed 2.0 Pro's 99th-percentile latency (~42 s
    in the 9-evidence walk diagnostic) with comfortable headroom. Set lower in
    fixture/bench contexts where you'd rather fail fast and retry.
    """
    raw = os.environ.get("LINEBASE_LLM_TIMEOUT_S", "").strip()
    if not raw:
        return 90.0
    try:
        v = float(raw)
    except ValueError:
        return 90.0
    return max(1.0, v)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"


@dataclass
class MatchResult:
    found: bool
    bbox: tuple[int, int, int, int] | None
    confidence: float
    reason: str
    raw_response: str
    prompt_version: str
    model: str
    usage: dict[str, int] | None = None
    # v2+ scalar tie-breakers; older prompts leave these at 0.0
    clarity: float = 0.0
    completeness: float = 0.0
    isolation: float = 0.0
    # Debug: how many extra LLM calls were needed before JSON parsed cleanly.
    # 0 = parsed on first try, 1 = parsed after the stricter-retry, >1 only if
    # we ever raise the retry budget.
    json_retry_count: int = 0


@dataclass
class VerifyAnswer:
    """Output of the self-verify call (see prompts/verify_v_<n>.md)."""

    contains_full_logo: bool
    fit: str  # one of {"tight", "loose", "too_tight", "wrong"}
    confidence: float
    reason: str
    suggested_bbox: tuple[int, int, int, int] | None  # in CROPPED image's coords
    raw_response: str
    prompt_version: str
    model: str
    usage: dict[str, int] | None = None


def _image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _active_prompt() -> tuple[str, str]:
    """Return (version_label, prompt_text). Falls back to v0 baseline if no file yet."""
    import os

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    requested = os.environ.get("LINEBASE_PROMPT_VERSION")
    files = sorted(PROMPTS_DIR.glob("v_*.md"))
    if requested:
        target = PROMPTS_DIR / f"v_{requested}.md"
        if not target.exists():
            raise FileNotFoundError(f"Requested prompt version not found: {target}")
        return requested, target.read_text(encoding="utf-8")
    if files:
        f = files[-1]
        return f.stem.removeprefix("v_"), f.read_text(encoding="utf-8")
    return "0-baseline-inline", _BASELINE_PROMPT


def _active_verify_prompt() -> tuple[str, str]:
    """Return (version_label, prompt_text) for the verify-loop prompt.

    Picks the latest `prompts/verify_v_*.md` unless `LINEBASE_VERIFY_PROMPT_VERSION`
    overrides it (e.g. `LINEBASE_VERIFY_PROMPT_VERSION=1` → prompts/verify_v_1.md).
    """
    import os

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    requested = os.environ.get("LINEBASE_VERIFY_PROMPT_VERSION")
    files = sorted(PROMPTS_DIR.glob("verify_v_*.md"))
    if requested:
        target = PROMPTS_DIR / f"verify_v_{requested}.md"
        if not target.exists():
            raise FileNotFoundError(f"Requested verify prompt version not found: {target}")
        return f"verify-{requested}", target.read_text(encoding="utf-8")
    if not files:
        raise FileNotFoundError(
            "No prompts/verify_v_*.md file found — required by verify_crop()"
        )
    f = files[-1]
    return f"verify-{f.stem.removeprefix('verify_v_')}", f.read_text(encoding="utf-8")


_BASELINE_PROMPT = """You are given two images.

Image 1: a line-art trademark logo (typically black on white background).
Image 2: a real-world product photograph that MAY or MAY NOT contain that logo.
The logo, if present, could be printed / embossed / embroidered / a decal — possibly small, rotated,
partially occluded, or low-contrast against the product material.

Respond with strict JSON, nothing else (no prose, no markdown fences):

{
  "found": true | false,
  "bbox": [x1, y1, x2, y2] or null,
  "confidence": 0.0 - 1.0,
  "reason": "<one short sentence>"
}

Coordinate system: pixel coordinates in Image 2, 0-indexed, with (0,0) at the TOP-LEFT corner.
x1 < x2, y1 < y2. bbox should tightly enclose the logo region, with at most ~5% padding on each side.

If the logo appears multiple times, return the most prominent / least occluded instance.
If you are unsure, return found=false and confidence near 0 — false positives are worse than misses.
"""


# Match the first balanced { ... } blob. DOTALL so newlines are matched.
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Tokens used by some models to wrap their final answer. Strip before JSON parse.
# GLM-4.5V emits <|begin_of_box|>...<|end_of_box|>; some Qwen variants emit
# <|im_start|> / <|im_end|>; some emit ```json fences. We zap them all.
_WRAPPER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|begin_of_box\|>", re.IGNORECASE),
    re.compile(r"<\|end_of_box\|>", re.IGNORECASE),
    re.compile(r"<\|im_start\|>(?:\w+)?", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"```(?:json|JSON)?", re.IGNORECASE),
    re.compile(r"```"),
)


def _strip_wrappers(raw: str) -> str:
    s = raw
    for pat in _WRAPPER_PATTERNS:
        s = pat.sub("", s)
    return s


def _iter_balanced_json(text: str):
    """Yield every balanced {...} blob found in `text`, in order of appearance.

    Walks forwards through `text`. For each `{`, tries to find a matching `}`
    while respecting double-quoted strings and escape sequences. Useful when a
    response contains multiple `{...}` snippets (thinking content with curly
    braces, followed by the actual JSON answer).
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        j = i
        while j < n:
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[i : j + 1]
                        break
            j += 1
        # advance past this `{` regardless of whether it balanced
        i += 1


def _looks_like_answer(obj) -> bool:
    """True if `obj` looks like our target schema (has at least 'found' or 'bbox')."""
    if not isinstance(obj, dict):
        return False
    return any(k in obj for k in ("found", "bbox", "confidence", "clarity"))


def _parse_json_response(raw: str) -> dict:
    """Robust JSON extraction.

    Order of operations:
      1) Strip known wrapper tokens (GLM <|begin_of_box|>, code fences, etc).
      2) Enumerate every balanced {...} blob in the cleaned text.
      3) Prefer the LAST blob that parses AND looks like our target schema
         (has 'found'/'bbox'/'confidence'). This handles models that emit
         visible thinking text containing stray `{...}` before the answer.
      4) If no schema-shaped blob parses, fall back to the last parseable blob.
      5) If nothing parses, fall back to the legacy greedy regex.
    """
    cleaned = _strip_wrappers(raw).strip()
    schema_match: dict | None = None
    last_parseable: dict | None = None
    for blob in _iter_balanced_json(cleaned):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        last_parseable = obj
        if _looks_like_answer(obj):
            schema_match = obj  # keep the LAST schema-shaped one
    if schema_match is not None:
        return schema_match
    if last_parseable is not None:
        return last_parseable
    match = _JSON_RE.search(cleaned)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")
    return json.loads(match.group(0))


_STRICT_RETRY_MSG = (
    "Your previous reply could not be parsed as JSON. "
    "Respond again with raw JSON only — no prose, no markdown code fences, "
    "no <|begin_of_box|> tokens, no commentary before or after. "
    "Output must start with { and end with }."
)


def _build_client(settings: Settings, provider: ProviderConfig | None, model: str) -> OpenAI:
    """Pick the right OpenAI-compatible client for (model, provider) override.

    Priority:
      1) explicit `provider` arg
      2) `settings.resolve_provider(model)` based on model prefix / LINEBASE_PROVIDER env
      3) fall back to settings.primary

    NB: we set `max_retries=0` because we manage retries ourselves at the
    benchmark layer (one stricter-prompt retry per call). With the SDK's default
    `max_retries=2`, a stalled HTTP call could be retried 2x×60s = 120s+ before
    surfacing, which busts our per-fixture latency budget.
    """
    if provider is not None:
        return OpenAI(api_key=provider.api_key, base_url=provider.base_url, max_retries=0)
    try:
        pc = settings.resolve_provider(model)
    except Exception:
        pc = settings.primary
    return OpenAI(api_key=pc.api_key, base_url=pc.base_url, max_retries=0)


def _create_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    timeout: float | None,
):
    # The OpenAI Python SDK accepts `timeout` per-call; pass it through when set.
    # When caller didn't supply a timeout, default to LINEBASE_LLM_TIMEOUT_S
    # (90 s) so a stalled provider can never hang the pipeline indefinitely —
    # see module docstring for the night-of-2026-05-23 Doubao stall background.
    eff_timeout = timeout if timeout is not None else _default_timeout_s()
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "timeout": eff_timeout,
    }
    return client.chat.completions.create(**kwargs)


def match_logo_in_photo(
    logo_path: Path,
    photo_path: Path,
    settings: Settings | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
    provider: ProviderConfig | None = None,
    timeout: float | None = None,
) -> MatchResult:
    """Run the matcher with optional model + provider overrides.

    - `model=None` → use `settings.model`.
    - `provider=None` → resolved from the model id via `settings.resolve_provider`.
    - `client` is only used when both `provider` and `model` are unset (legacy path).
    - `timeout` (seconds) is passed to the OpenAI SDK per-call.
    """
    settings = settings or Settings.from_env()
    use_model = model or settings.model

    # When the caller supplied a client and didn't override model/provider, keep
    # the legacy fast path (no client churn between calls).
    if client is None or model is not None or provider is not None:
        client = _build_client(settings, provider, use_model)

    version, prompt = _active_prompt()

    logo_url = _image_to_data_url(logo_path)
    photo_url = _image_to_data_url(photo_path)

    user_content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Image 1 (LOGO):"},
        {"type": "image_url", "image_url": {"url": logo_url}},
        {"type": "text", "text": "Image 2 (PHOTO):"},
        {"type": "image_url", "image_url": {"url": photo_url}},
    ]
    messages: list[dict] = [{"role": "user", "content": user_content}]

    completion = _create_completion(client, use_model, messages, timeout)
    raw = completion.choices[0].message.content or ""
    usage = _usage_dict(completion)

    retry_count = 0
    try:
        data = _parse_json_response(raw)
    except (ValueError, json.JSONDecodeError):
        # One stricter retry, with the original raw response + new instruction.
        retry_count = 1
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _STRICT_RETRY_MSG},
        ]
        retry_completion = _create_completion(client, use_model, retry_messages, timeout)
        retry_raw = retry_completion.choices[0].message.content or ""
        # Accumulate token usage from the retry too — caller's cost accounting
        # must see all calls.
        retry_usage = _usage_dict(retry_completion)
        usage = _sum_usage(usage, retry_usage)
        # If even the retry blows up, let the exception escape.
        data = _parse_json_response(retry_raw)
        raw = raw + "\n---retry---\n" + retry_raw

    bbox_raw = data.get("bbox")
    bbox: tuple[int, int, int, int] | None = None
    if bbox_raw and isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        try:
            bbox = tuple(int(v) for v in bbox_raw)  # type: ignore[assignment]
        except (TypeError, ValueError):
            bbox = None

    def _scalar(key: str) -> float:
        v = data.get(key)
        if v is None:
            return 0.0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))

    return MatchResult(
        found=bool(data.get("found", False)),
        bbox=bbox,
        confidence=float(data.get("confidence", 0.0)),
        reason=str(data.get("reason", "")),
        raw_response=raw,
        prompt_version=version,
        model=use_model,
        usage=usage,
        clarity=_scalar("clarity"),
        completeness=_scalar("completeness"),
        isolation=_scalar("isolation"),
        json_retry_count=retry_count,
    )


def _usage_dict(completion) -> dict[str, int] | None:
    u = getattr(completion, "usage", None)
    if not u:
        return None
    # Reasoning-token field varies across providers; pull from details if present.
    reasoning_tokens = 0
    details = getattr(u, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    return {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
        "reasoning_tokens": reasoning_tokens,
    }


def _sum_usage(a: dict[str, int] | None, b: dict[str, int] | None) -> dict[str, int] | None:
    if not a:
        return b
    if not b:
        return a
    keys = set(a) | set(b)
    return {k: int(a.get(k, 0) or 0) + int(b.get(k, 0) or 0) for k in keys}


_VALID_FITS = {"tight", "loose", "too_tight", "wrong"}


def verify_crop(
    logo_path: Path,
    cropped_path: Path,
    settings: Settings | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
    provider: ProviderConfig | None = None,
    timeout: float | None = None,
) -> VerifyAnswer:
    """Self-verify call: send (logo, candidate cropped region) → VerifyAnswer.

    The cropped region is expected to be a ~20%-padded crop of an earlier bbox
    candidate. `suggested_bbox` in the response (if any) is in the cropped image's
    own pixel coordinates — the caller must translate back to the original photo.

    Per-call provider routing (same rules as `match_logo_in_photo`): the
    effective model id is `model` if supplied, otherwise `settings.review_model`.
    The client is rebuilt from the correct provider when no client is given OR
    when the caller overrides model/provider; this prevents legacy clients
    pinned to the OpenAI block from being reused against SiliconFlow models.
    """
    settings = settings or Settings.from_env()
    use_model = model or settings.review_model

    if client is None or model is not None or provider is not None:
        client = _build_client(settings, provider, use_model)

    version, prompt = _active_verify_prompt()

    logo_url = _image_to_data_url(logo_path)
    crop_url = _image_to_data_url(cropped_path)

    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Image 1 (LOGO):"},
                {"type": "image_url", "image_url": {"url": logo_url}},
                {"type": "text", "text": "Image 2 (CANDIDATE CROP):"},
                {"type": "image_url", "image_url": {"url": crop_url}},
            ],
        }
    ]

    completion = _create_completion(client, use_model, messages, timeout)

    raw = completion.choices[0].message.content or ""
    try:
        data = _parse_json_response(raw)
    except (ValueError, json.JSONDecodeError):
        # One stricter retry — same robustness contract as `match_logo_in_photo`.
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _STRICT_RETRY_MSG},
        ]
        retry_completion = _create_completion(client, use_model, retry_messages, timeout)
        retry_raw = retry_completion.choices[0].message.content or ""
        data = _parse_json_response(retry_raw)
        raw = raw + "\n---retry---\n" + retry_raw

    fit_raw = str(data.get("fit", "wrong")).strip().lower()
    fit = fit_raw if fit_raw in _VALID_FITS else "wrong"

    sugg_raw = data.get("suggested_bbox")
    suggested: tuple[int, int, int, int] | None = None
    if sugg_raw and isinstance(sugg_raw, (list, tuple)) and len(sugg_raw) == 4:
        try:
            suggested = (int(sugg_raw[0]), int(sugg_raw[1]), int(sugg_raw[2]), int(sugg_raw[3]))
            # Enforce x1<x2, y1<y2 — otherwise drop the suggestion.
            if suggested[0] >= suggested[2] or suggested[1] >= suggested[3]:
                suggested = None
        except (TypeError, ValueError):
            suggested = None

    usage = _usage_dict(completion)

    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    contains = bool(data.get("contains_full_logo", False))
    # Enforce contract: fit=="wrong" => contains_full_logo=False.
    if fit == "wrong":
        contains = False

    return VerifyAnswer(
        contains_full_logo=contains,
        fit=fit,
        confidence=conf,
        reason=str(data.get("reason", "")),
        suggested_bbox=suggested,
        raw_response=raw,
        prompt_version=version,
        model=use_model,
        usage=usage,
    )
