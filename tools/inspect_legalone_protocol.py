"""Fetch pinned primary-source LegalKit protocol files; never execute them."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7db1cfd97ba47011fa169c55a2437cfde548dffa"
FILES = {
    "README.md": "41fa8e1c3218462acd5c3f3f75c64127b3d15a5e",
    "legalkit/main.py": "e6fefed50a05591681e51950213bd921aa83f2e8",
    "legalkit/models/local.py": "17ca5bd2086c703e64b1542e7d58b3fcd1215f13",
    "legalkit/models/api.py": "0fb6a675466f1e20fc666c4f99dc346986592283",
    "legalkit/models/huggingface.py": "5eb8756dd1e4b5c36a349e1fe85fc7103347cd5a",
    "legalkit/datasets/LawBench/generator.py": "152d9e5cd1ad8a2a5389338524deaed16c257a93",
    "legalkit/datasets/LawBench/evaluator.py": "22757f985c58b8103288343220be75b397ba313e",
    "legalkit/datasets/LawBench/utils.py": "108b742d337acb221349a30080042e94737a5d55",
}


def main():
    directory = ROOT / "tools/acceptance_vendor/LegalKit_protocol"
    manifest = {"repository": "https://github.com/DavidMiao1127/LegalKit", "revision": REVISION,
                "purpose": "inspect primary-source output processing only; downloaded code is not executed", "files": []}
    for name, expected_blob in FILES.items():
        url = f"https://raw.githubusercontent.com/DavidMiao1127/LegalKit/{REVISION}/{name}"
        target = directory / name
        if target.exists():
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ZLQL-protocol-inspection"}), timeout=60) as response:
                data = response.read()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != expected_blob:
            raise ValueError(f"official git blob mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest["files"].append({"path": name, "url": url, "git_blob_sha": blob,
                                  "sha256": hashlib.sha256(data).hexdigest()})
    (directory / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "verified_not_executed", "files": len(FILES), "revision": REVISION}))


if __name__ == "__main__":
    main()
