"""URL → local cache. Content-addressed by sha256 to make re-runs free."""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "images"

# USPTO TSDR returns 403 for non-browser User-Agents. Pretend to be Chrome on
# Windows. Adding Accept lets us fall back gracefully for endpoints that vary
# response by content-type negotiation.
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    ),
}


def _extension_from_response(resp: httpx.Response, url: str) -> str:
    ct = resp.headers.get("content-type", "").split(";")[0].strip()
    if ct:
        ext = mimetypes.guess_extension(ct) or ""
        if ext:
            return ext
    # fallback: url suffix
    suffix = Path(url.split("?")[0]).suffix
    if suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        return suffix.lower()
    return ".bin"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _download(url: str, client: httpx.Client) -> tuple[bytes, str]:
    resp = client.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    return resp.content, _extension_from_response(resp, url)


def fetch(url: str, cache_dir: Path | None = None, client: httpx.Client | None = None) -> Path:
    """Download `url` (with retries) and return the local cache path. Re-uses cache if present."""
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    # url hash for stable filename
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    existing = list(cache_dir.glob(f"{url_hash}.*"))
    if existing:
        return existing[0]

    owns_client = client is None
    if client is None:
        client = httpx.Client(headers=_DEFAULT_HEADERS)
    try:
        data, ext = _download(url, client)
    finally:
        if owns_client:
            client.close()
    target = cache_dir / f"{url_hash}{ext}"
    target.write_bytes(data)
    return target


def fetch_many(urls: list[str], cache_dir: Path | None = None) -> list[Path]:
    """Sequential fetch; sufficient for the small per-row volume in this project."""
    with httpx.Client(headers=_DEFAULT_HEADERS) as client:
        return [fetch(u, cache_dir=cache_dir, client=client) for u in urls]
