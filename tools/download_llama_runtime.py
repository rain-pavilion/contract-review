"""Fetch a pinned official llama.cpp CPU runtime and matching converter source.

This script only downloads, verifies, and extracts public official artifacts.
It never loads a model, runs downloaded binaries, installs dependencies, or
modifies pretrained weights. Network access requires the sandbox's approval.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com/repos/ggml-org/llama.cpp"
HEADERS = {"User-Agent": "ZLQL-official-llama-runtime-preparation", "Accept": "application/vnd.github+json"}


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
        return json.load(response)


def release_from_html(tag: str, directory: Path) -> tuple[dict, str]:
    """Use the public release page's own asset digests without the API quota."""
    page_url = "https://github.com/ggml-org/llama.cpp/releases/" + ("latest" if tag == "latest" else "tag/" + tag)
    with urllib.request.urlopen(urllib.request.Request(page_url, headers={"User-Agent": HEADERS["User-Agent"]}), timeout=30) as response:
        final_url = response.geturl()
        release_html = response.read().decode("utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "release.html").write_text(release_html, encoding="utf-8")
    (directory / "resolved_release_url.txt").write_text(final_url, encoding="utf-8")
    matched_tag = re.search(r"/releases/tag/(b\d+)(?:$|[?#])", final_url)
    if not matched_tag and tag == "latest":
        # /latest may intentionally point to a meta release. Select the first
        # actual b-number build listed on the official releases page instead.
        listing_url = "https://github.com/ggml-org/llama.cpp/releases"
        with urllib.request.urlopen(urllib.request.Request(listing_url, headers={"User-Agent": HEADERS["User-Agent"]}), timeout=30) as response:
            listing_html = response.read().decode("utf-8")
        (directory / "release_listing.html").write_text(listing_html, encoding="utf-8")
        latest_build = re.search(r"/ggml-org/llama\.cpp/releases/tag/(b\d+)", listing_html)
        if latest_build:
            final_url = "https://github.com/ggml-org/llama.cpp/releases/tag/" + latest_build.group(1)
            with urllib.request.urlopen(urllib.request.Request(final_url, headers={"User-Agent": HEADERS["User-Agent"]}), timeout=30) as response:
                release_html = response.read().decode("utf-8")
            (directory / "release.html").write_text(release_html, encoding="utf-8")
            (directory / "resolved_release_url.txt").write_text(final_url, encoding="utf-8")
            matched_tag = re.search(r"/releases/tag/(b\d+)(?:$|[?#])", final_url)
    if not matched_tag:
        raise RuntimeError("Official release page did not expose a b-number build: " + final_url)
    resolved = matched_tag.group(1)
    if tag != "latest" and tag != resolved:
        raise RuntimeError("Official release page tag mismatch")
    assets_url = "https://github.com/ggml-org/llama.cpp/releases/expanded_assets/" + resolved
    with urllib.request.urlopen(urllib.request.Request(assets_url, headers={"User-Agent": HEADERS["User-Agent"]}), timeout=30) as response:
        asset_html = response.read().decode("utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "release.html").write_text(release_html, encoding="utf-8")
    (directory / "expanded_assets.html").write_text(asset_html, encoding="utf-8")
    links = list(re.finditer(r'href="(/ggml-org/llama\.cpp/releases/download/' + re.escape(resolved) + r'/([^"/?]+))"', asset_html))
    assets = []
    for index, match in enumerate(links):
        fragment = asset_html[match.end():links[index + 1].start() if index + 1 < len(links) else len(asset_html)]
        plain = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
        digest = re.search(r"sha256:\s*([a-fA-F0-9]{64})", plain)
        assets.append({"name": html.unescape(match.group(2)), "browser_download_url": "https://github.com" + html.unescape(match.group(1)),
                       "size": 0, "digest": "sha256:" + digest.group(1).lower() if digest else ""})
    if not assets:
        raise RuntimeError("Official release asset links not found")
    commits = re.findall(r"/ggml-org/llama\.cpp/commit/([a-f0-9]{40})", release_html)
    return {"tag_name": resolved, "html_url": final_url, "assets": assets,
            "metadata_source": "official_release_html", "commit_from_html": commits[0] if commits else ""}, resolved


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url: str, target: Path, *, size: int = 0, expected_sha: str = "") -> dict:
    if target.exists() and (not size or target.stat().st_size == size):
        actual = sha256(target)
        if expected_sha and actual == expected_sha:
            print(f"Verified existing {target.name}", flush=True)
            return {"url": url, "path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
                    "sha256": actual, "expected_sha256": expected_sha, "digest_verified": True}
    print(f"Downloading {url}", flush=True)
    partial = target.with_suffix(target.suffix + ".part")
    started = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
    with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            output.write(chunk)
    if size and partial.stat().st_size != size:
        raise RuntimeError(f"Size mismatch: {target.name}")
    actual = sha256(partial)
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f"SHA256 mismatch: {target.name}")
    partial.replace(target)
    print(f"Downloaded {target.name}: {target.stat().st_size} bytes in {time.perf_counter()-started:.2f}s", flush=True)
    return {"url": url, "path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
            "sha256": actual, "expected_sha256": expected_sha, "digest_verified": bool(expected_sha),
            "digest_status": "matched_official_digest" if expected_sha else "expected_digest_unavailable_recorded_local_sha256"}


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            path = (destination / entry.filename).resolve()
            if not path.is_relative_to(base):
                raise RuntimeError("Unsafe archive member path")
        package.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="b12246")
    parser.add_argument("--official-html", action="store_true", help="Read digests from official release HTML instead of the rate-limited GitHub API")
    parser.add_argument("--out", type=Path, default=ROOT / ".model-runtime/llama.cpp")
    args = parser.parse_args()
    if args.tag != "latest" and (not args.tag.startswith("b") or not args.tag[1:].isdigit()):
        parser.error("An explicit official b-number release tag is required")
    directory = (args.out / args.tag).resolve()
    if not directory.is_relative_to(ROOT):
        parser.error("Runtime output must stay within this workspace")
    directory.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "tools/acceptance_results/llama_runtime_preparation.json"
    result = {"status": "running", "repository": "ggml-org/llama.cpp", "tag": args.tag,
              "models_loaded": False, "model_weights_modified": False}
    try:
        if args.official_html:
            release, resolved_tag = release_from_html(args.tag, directory)
            args.tag = resolved_tag
            commit = release.get("commit_from_html", "")
        else:
            release = get_json(API + "/releases/" + ("latest" if args.tag == "latest" else "tags/" + args.tag))
            args.tag = release["tag_name"]
            reference = get_json(API + "/git/ref/tags/" + args.tag)["object"]
            if reference["type"] == "tag":
                reference = get_json(reference["url"])["object"]
            if reference["type"] != "commit":
                raise RuntimeError("Release tag does not resolve to a commit")
            commit = reference["sha"]
        result.update(tag=args.tag, commit=commit, published_at=release.get("published_at"),
                      official_release_url=release.get("html_url"), metadata_source=release.get("metadata_source", "official_github_api"))
        (directory / "release.json").write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")
        asset_name = f"llama-{args.tag}-bin-win-cpu-x64.zip"
        asset = next((item for item in release.get("assets", []) if item["name"] == asset_name), None)
        if not asset:
            asset_name = f"llama-{args.tag}-bin-win-avx2-x64.zip"
            asset = next((item for item in release.get("assets", []) if item["name"] == asset_name), None)
        if not asset:
            raise RuntimeError("Official Windows x64 CPU release asset not found")
        digest = str(asset.get("digest") or "")
        expected_sha = digest[7:] if digest.startswith("sha256:") else ""
        cpu_zip = directory / asset_name
        result["cpu_archive"] = download(asset["browser_download_url"], cpu_zip,
                                         size=asset["size"], expected_sha=expected_sha)
        result["cuda_assets_not_downloaded"] = [
            {"name": item["name"], "bytes": item["size"], "url": item["browser_download_url"], "digest": item.get("digest")}
            for item in release.get("assets", []) if "win-cuda" in item["name"] or "cudart-llama" in item["name"]]
        source_zip = directory / f"llama-{args.tag}-source.zip"
        source_ref = commit or "refs/tags/" + args.tag
        result["source_archive"] = download("https://codeload.github.com/ggml-org/llama.cpp/zip/" + source_ref, source_zip)
        extract(cpu_zip, directory / "cpu")
        extract(source_zip, directory / "source")
        server = list((directory / "cpu").rglob("llama-server.exe"))
        quantizer = list((directory / "cpu").rglob("llama-quantize.exe"))
        converter = list((directory / "source").rglob("convert_hf_to_gguf.py"))
        if len(server) != 1 or len(quantizer) != 1 or len(converter) != 1:
            raise RuntimeError("Expected runtime or converter files are absent/ambiguous")
        result.update(status="prepared_not_executed", server=str(server[0].relative_to(ROOT)),
                      quantizer=str(quantizer[0].relative_to(ROOT)), converter=str(converter[0].relative_to(ROOT)),
                      cpu_libraries=[str(p.relative_to(ROOT)) for p in (directory / "cpu").rglob("ggml-cpu*.dll")])
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "prepared_not_executed" else 1


if __name__ == "__main__":
    sys.exit(main())
