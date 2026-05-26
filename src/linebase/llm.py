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
import contextlib
import json
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from PIL import Image

from linebase.config import ProviderConfig, Settings

# Longest-side budget (pixels) for what we ship to a vision model. Above this
# we downscale before encoding to base64, then scale the returned bbox back
# into original-image coords. This was added in iter 6.1 after both Qwen3-VL
# and GLM-4.5V were observed returning bboxes in their *post-resize* internal
# coordinate frame (~1280 px), which mapped to a wildly wrong region when the
# crop step used the original 1836x2376 image. 1280 is conservative — well
# under every provider's documented vision-input cap and well above the
# resolution needed to read product photos.
_MAX_VLM_PIXELS = 1280


def _resize_for_vlm(path: Path) -> tuple[Path, float]:
    """Return (path_to_send, scale_factor).

    If the image's longest side <= `_MAX_VLM_PIXELS`, return the original
    path unchanged with `scale_factor=1.0`. Otherwise write a downscaled
    JPEG/PNG copy to a tempfile and return `(tempfile_path, scale)` where
    `scale = new_longest / old_longest`. Callers MUST divide any bbox
    returned by the model by `scale` to recover original-image coords, and
    are responsible for unlinking the tempfile when they're done with it
    (the helper does not own its lifetime).
    """
    with Image.open(path) as img:
        w, h = img.size
        longest = max(w, h)
        if longest <= _MAX_VLM_PIXELS:
            return path, 1.0
        scale = _MAX_VLM_PIXELS / longest
        new_size = (round(w * scale), round(h * scale))
        resized = img.convert("RGB").resize(new_size, Image.LANCZOS)
    suffix = path.suffix or ".png"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    tmp = Path(tmp_name)
    fmt = "JPEG" if suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    if fmt == "JPEG":
        resized.save(tmp, format=fmt, quality=92)
    else:
        resized.save(tmp, format=fmt)
    return tmp, scale


_QWEN_COORD_MODE_PREFIXES: tuple[str, ...] = ("Qwen/", "Pro/Qwen/")
_NORMALIZED_BBOX_COORD_MAX = 1000.0


def _is_qwen_model(model: str) -> bool:
    return model.startswith(_QWEN_COORD_MODE_PREFIXES)


def _coords_look_normalized_1000(coords: list[float], send_w: int, send_h: int) -> bool:
    """Heuristic for Qwen's visual-grounding 0-1000 coordinate frame.

    Qwen-family models often emit bbox coordinates in a 0-1000 frame even when
    prompted for pixels. Treat them as normalized only when the transmitted
    image is larger than that frame on at least one axis; for smaller images a
    value like x=900 can be a genuine pixel coordinate.
    """
    if max(send_w, send_h) <= _NORMALIZED_BBOX_COORD_MAX:
        return False
    if not all(0.0 <= c <= _NORMALIZED_BBOX_COORD_MAX for c in coords):
        return False
    x1, y1, x2, y2 = coords
    return not (x2 <= x1 or y2 <= y1)


def _map_bbox_coords_to_source(
    coords: list[float],
    *,
    model: str,
    source_w: int,
    source_h: int,
    sent_w: int,
    sent_h: int,
    sent_scale: float,
) -> tuple[list[float], str]:
    """Map provider-returned bbox coords into source-image pixels.

    Most OpenAI-compatible providers follow our prompt and return pixel coords
    in the transmitted image frame; those need only the existing resize scale
    reversal. Qwen is the exception we have observed in reports: on large
    images it may still answer in the 0-1000 grounding frame, which must be
    expanded directly to the original source dimensions.
    """
    if _is_qwen_model(model) and _coords_look_normalized_1000(coords, sent_w, sent_h):
        return (
            [
                coords[0] * source_w / _NORMALIZED_BBOX_COORD_MAX,
                coords[1] * source_h / _NORMALIZED_BBOX_COORD_MAX,
                coords[2] * source_w / _NORMALIZED_BBOX_COORD_MAX,
                coords[3] * source_h / _NORMALIZED_BBOX_COORD_MAX,
            ],
            "qwen_normalized_1000",
        )
    if sent_scale != 1.0:
        return [c / sent_scale for c in coords], "sent_pixels_scaled"
    return coords, "source_pixels"


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
    # Coordinate diagnostics. `raw_bbox` is exactly what the provider returned
    # before any provider-specific frame conversion; `bbox_coord_mode` records
    # how we interpreted it. Kept optional so older tests/fixtures can construct
    # MatchResult without caring about bbox plumbing details.
    raw_bbox: tuple[float, float, float, float] | None = None
    bbox_coord_mode: str | None = None
    source_size: tuple[int, int] | None = None
    sent_size: tuple[int, int] | None = None


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
x1 < x2, y1 < y2. bbox should tightly enclose the logo region, with at most ~5% padding
on each side.

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
    version, prompt = _active_prompt()
    return _run_match_call(
        logo_path=logo_path,
        photo_path=photo_path,
        prompt_text=prompt,
        prompt_version=version,
        settings=settings,
        client=client,
        model=model,
        provider=provider,
        timeout=timeout,
    )


def match_logo_in_zoomed(
    logo_path: Path,
    photo_path: Path,
    *,
    region_bbox: tuple[int, int, int, int],
    pad_ratio: float = 0.30,
    settings: Settings | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
    provider: ProviderConfig | None = None,
    timeout: float | None = None,
) -> tuple[MatchResult, tuple[int, int]]:
    """Refine pass — crop the photo to a zoomed region around `region_bbox`
    (with `pad_ratio` padding per side), optionally upscale 2x for tiny crops,
    then call the VLM with `prompts/v_4_refine.md`.

    Returns `(MatchResult, (zoom_origin_x, zoom_origin_y))` where `zoom_origin`
    is the top-left of the zoom crop in the ORIGINAL photo's pixel coords. The
    returned `MatchResult.bbox` is in ZOOM-CROP coords — the caller must
    translate by `(zoom_origin_x, zoom_origin_y)` to recover original-image
    coords.

    Notes:
      - When the zoom crop's longest side is < 400 px we upscale 2x with LANCZOS
        before sending; VLMs are much more reliable on bigger crops.
      - We always read the v_4_refine.md prompt directly (not via _active_prompt)
        so this stays decoupled from `LINEBASE_PROMPT_VERSION` overrides — the
        refine prompt is the only valid one for this call shape.
    """
    target = PROMPTS_DIR / "v_4_refine.md"
    if not target.exists():
        raise FileNotFoundError(f"Required refine prompt missing: {target}")
    prompt_text = target.read_text(encoding="utf-8")
    if prompt_text.startswith("---"):  # strip YAML front-matter
        end = prompt_text.find("\n---", 3)
        if end != -1:
            prompt_text = prompt_text[end + 4 :].lstrip("\n")

    with Image.open(photo_path) as img:
        photo_w, photo_h = img.size
    rx1, ry1, rx2, ry2 = region_bbox
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)
    pad_x = int(rw * pad_ratio)
    pad_y = int(rh * pad_ratio)
    zx1 = max(0, rx1 - pad_x)
    zy1 = max(0, ry1 - pad_y)
    zx2 = min(photo_w, rx2 + pad_x)
    zy2 = min(photo_h, ry2 + pad_y)
    if zx2 <= zx1:
        zx2 = min(photo_w, zx1 + 1)
    if zy2 <= zy1:
        zy2 = min(photo_h, zy1 + 1)

    # Crop + optionally upscale 2x when the longest side is < 400 px.
    fd, tmp_name = tempfile.mkstemp(suffix=photo_path.suffix or ".png")
    os.close(fd)
    zoom_path = Path(tmp_name)
    zoom_upscale = 1
    try:
        with Image.open(photo_path) as src:
            crop_img = src.convert("RGB").crop((zx1, zy1, zx2, zy2))
        cw, ch = crop_img.size
        if max(cw, ch) < 400:
            zoom_upscale = 2
            crop_img = crop_img.resize((cw * 2, ch * 2), Image.LANCZOS)
        save_fmt = "JPEG" if zoom_path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
        if save_fmt == "JPEG":
            crop_img.save(zoom_path, format=save_fmt, quality=92)
        else:
            crop_img.save(zoom_path, format=save_fmt)

        result = _run_match_call(
            logo_path=logo_path,
            photo_path=zoom_path,
            prompt_text=prompt_text,
            prompt_version="4_refine",
            settings=settings,
            client=client,
            model=model,
            provider=provider,
            timeout=timeout,
        )
    finally:
        with contextlib.suppress(OSError):
            zoom_path.unlink(missing_ok=True)

    # `_run_match_call` reads the photo dimensions of the zoom-crop (which may
    # have been upscaled 2x) and scales bbox into THOSE coords. If we upscaled,
    # the returned bbox is in upscaled-crop coords — we need to bring it back
    # to the un-upscaled crop coords (= zoom_origin frame) so the caller's
    # translate to global coords lines up.
    if result.bbox is not None:
        # Use the explicit factor; inferring from bbox size misses marks in
        # the upper-left half of an upscaled crop.
        bx1, by1, bx2, by2 = result.bbox
        crop_w = zx2 - zx1
        crop_h = zy2 - zy1
        if zoom_upscale > 1:
            bx1 //= zoom_upscale
            by1 //= zoom_upscale
            bx2 = max(bx1 + 1, bx2 // zoom_upscale)
            by2 = max(by1 + 1, by2 // zoom_upscale)
        bx1 = max(0, min(bx1, crop_w - 1))
        by1 = max(0, min(by1, crop_h - 1))
        bx2 = max(bx1 + 1, min(bx2, crop_w))
        by2 = max(by1 + 1, min(by2, crop_h))
        result.bbox = (bx1, by1, bx2, by2)

    return result, (zx1, zy1)


def match_logo_with_feedback(
    logo_path: Path,
    photo_path: Path,
    *,
    prior_bbox: tuple[int, int, int, int],
    prior_reason: str,
    verify_reason: str,
    settings: Settings | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
    provider: ProviderConfig | None = None,
    timeout: float | None = None,
) -> MatchResult:
    """Pass-3 retry: re-ask the matcher after a blank-crop verify rejection.

    Renders prompts/v_4_retry.md with the prior bbox + verifier reason baked
    in, sends (logo, photo) again, and parses the response with the same
    JSON contract as `match_logo_in_photo`. `prior_reason` is currently kept
    for caller debuggability (the prompt itself ignores it — the model is
    instructed to start fresh, not be primed by old reasoning).
    """
    del prior_reason  # noqa: B018 — accepted for symmetry but intentionally unused
    target = PROMPTS_DIR / "v_4_retry.md"
    if not target.exists():
        raise FileNotFoundError(f"Required retry prompt missing: {target}")
    text = target.read_text(encoding="utf-8")
    if text.startswith("---"):  # strip YAML front-matter
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")
    px1, py1, px2, py2 = prior_bbox
    prompt = (
        text.replace("%PRIOR_X1", str(int(px1)))
        .replace("%PRIOR_Y1", str(int(py1)))
        .replace("%PRIOR_X2", str(int(px2)))
        .replace("%PRIOR_Y2", str(int(py2)))
        .replace("%VERIFY_REASON", verify_reason.replace('"', "'"))
    )
    return _run_match_call(
        logo_path=logo_path,
        photo_path=photo_path,
        prompt_text=prompt,
        prompt_version="4_retry",
        settings=settings,
        client=client,
        model=model,
        provider=provider,
        timeout=timeout,
    )


def _run_match_call(
    *,
    logo_path: Path,
    photo_path: Path,
    prompt_text: str,
    prompt_version: str,
    settings: Settings | None,
    client: OpenAI | None,
    model: str | None,
    provider: ProviderConfig | None,
    timeout: float | None,
) -> MatchResult:
    """Shared body for `match_logo_in_photo` and `match_logo_with_feedback`.

    Sends (prompt + LOGO + PHOTO), parses the JSON response with the same
    stricter-retry-on-parse-error contract, and packs the result into a
    MatchResult. Extracted as a helper so the retry variant doesn't have to
    duplicate ~70 lines of message-building + JSON parsing.
    """
    settings = settings or Settings.from_env()
    use_model = model or settings.model
    if client is None or model is not None or provider is not None:
        client = _build_client(settings, provider, use_model)

    # Iter 6.1 — pre-resize both images to a known longest-side budget so the
    # model's internal coordinate frame is predictable, then scale the bbox
    # it returns back into the ORIGINAL photo's pixel space.
    with Image.open(photo_path) as _img:
        orig_w, orig_h = _img.size

    send_logo_path, _logo_scale = _resize_for_vlm(logo_path)
    send_photo_path, photo_scale = _resize_for_vlm(photo_path)
    with Image.open(send_logo_path) as _li:
        logo_send_w, logo_send_h = _li.size
    with Image.open(send_photo_path) as _pi:
        photo_send_w, photo_send_h = _pi.size

    try:
        user_content = [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": f"Image 1 (LOGO): {logo_send_w}x{logo_send_h} pixels"},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(send_logo_path)}},
            {
                "type": "text",
                "text": (
                    f"Image 2 (PHOTO): {photo_send_w}x{photo_send_h} pixels. "
                    "Return bbox coords as integers within "
                    f"[0,{photo_send_w - 1}] x [0,{photo_send_h - 1}]; "
                    "do NOT normalize."
                ),
            },
            {"type": "image_url", "image_url": {"url": _image_to_data_url(send_photo_path)}},
        ]
        messages: list[dict] = [{"role": "user", "content": user_content}]
        completion = _create_completion(client, use_model, messages, timeout)
        raw = completion.choices[0].message.content or ""
        usage = _usage_dict(completion)

        retry_count = 0
        try:
            data = _parse_json_response(raw)
        except (ValueError, json.JSONDecodeError):
            retry_count = 1
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _STRICT_RETRY_MSG},
            ]
            retry_completion = _create_completion(client, use_model, retry_messages, timeout)
            retry_raw = retry_completion.choices[0].message.content or ""
            usage = _sum_usage(usage, _usage_dict(retry_completion))
            data = _parse_json_response(retry_raw)
            raw = raw + "\n---retry---\n" + retry_raw
    finally:
        # Clean up the resized tempfiles, if any. Never crash on cleanup —
        # OS-level transient unlink failures must not mask the API result.
        for sent_path, src_path in (
            (send_logo_path, logo_path),
            (send_photo_path, photo_path),
        ):
            if sent_path != src_path:
                with contextlib.suppress(OSError):
                    sent_path.unlink(missing_ok=True)

    bbox_raw = data.get("bbox")
    bbox: tuple[int, int, int, int] | None = None
    if bbox_raw and isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        try:
            coords = [float(v) for v in bbox_raw]
        except (TypeError, ValueError):
            coords = None
        if coords is not None:
            raw_coords = tuple(coords)
            coords, bbox_coord_mode = _map_bbox_coords_to_source(
                coords,
                model=use_model,
                source_w=orig_w,
                source_h=orig_h,
                sent_w=photo_send_w,
                sent_h=photo_send_h,
                sent_scale=photo_scale,
            )
            x1 = max(0, min(int(round(coords[0])), orig_w - 1))
            y1 = max(0, min(int(round(coords[1])), orig_h - 1))
            x2 = max(x1 + 1, min(int(round(coords[2])), orig_w))
            y2 = max(y1 + 1, min(int(round(coords[3])), orig_h))
            bbox = (x1, y1, x2, y2)
        else:
            raw_coords = None
            bbox_coord_mode = None
    else:
        raw_coords = None
        bbox_coord_mode = None

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
        prompt_version=prompt_version,
        model=use_model,
        usage=usage,
        clarity=_scalar("clarity"),
        completeness=_scalar("completeness"),
        isolation=_scalar("isolation"),
        json_retry_count=retry_count,
        raw_bbox=raw_coords,
        bbox_coord_mode=bbox_coord_mode,
        source_size=(orig_w, orig_h),
        sent_size=(photo_send_w, photo_send_h),
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

    # Iter 6.1 — apply the same pre-resize to the candidate crop so the
    # model's `suggested_bbox` (in CROP coords) is interpreted in a known
    # frame. We scale `suggested_bbox` back to the original crop's pixels
    # before returning so downstream callers can translate to photo coords
    # without knowing anything about the resize.
    with Image.open(cropped_path) as _orig_crop:
        crop_orig_w, crop_orig_h = _orig_crop.size
    send_logo_path, _logo_scale = _resize_for_vlm(logo_path)
    send_crop_path, crop_scale = _resize_for_vlm(cropped_path)
    with Image.open(send_logo_path) as _li:
        logo_send_w, logo_send_h = _li.size
    with Image.open(send_crop_path) as _ci:
        crop_send_w, crop_send_h = _ci.size

    try:
        logo_url = _image_to_data_url(send_logo_path)
        crop_url = _image_to_data_url(send_crop_path)

        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": f"Image 1 (LOGO): {logo_send_w}x{logo_send_h} pixels"},
                    {"type": "image_url", "image_url": {"url": logo_url}},
                    {
                        "type": "text",
                        "text": (
                            f"Image 2 (CANDIDATE CROP): {crop_send_w}x{crop_send_h} "
                            "pixels. suggested_bbox coords MUST be integers within "
                            f"[0,{crop_send_w - 1}] x [0,{crop_send_h - 1}]; "
                            "do NOT normalize."
                        ),
                    },
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
    finally:
        for sent_path, src_path in (
            (send_logo_path, logo_path),
            (send_crop_path, cropped_path),
        ):
            if sent_path != src_path:
                with contextlib.suppress(OSError):
                    sent_path.unlink(missing_ok=True)

    fit_raw = str(data.get("fit", "wrong")).strip().lower()
    fit = fit_raw if fit_raw in _VALID_FITS else "wrong"

    sugg_raw = data.get("suggested_bbox")
    suggested: tuple[int, int, int, int] | None = None
    if sugg_raw and isinstance(sugg_raw, (list, tuple)) and len(sugg_raw) == 4:
        try:
            sf = [float(v) for v in sugg_raw]
            sf, _sugg_mode = _map_bbox_coords_to_source(
                sf,
                model=use_model,
                source_w=crop_orig_w,
                source_h=crop_orig_h,
                sent_w=crop_send_w,
                sent_h=crop_send_h,
                sent_scale=crop_scale,
            )
            suggested = (
                int(round(sf[0])),
                int(round(sf[1])),
                int(round(sf[2])),
                int(round(sf[3])),
            )
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
