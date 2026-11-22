"""对政府采购合同页面图跑 OCR，把结果写回 text/ 层（供端到端测试用）。

只处理指定数量的扫描件，全页跑；每页约 7.5 s（CPU rapidocr），所以放在后台执行。

用法：
    python tools/_ocr_gov_contracts.py --limit 5
"""

from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import ocr  # noqa: E402


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    limit = 5
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])

    root = ROOT / "data" / "gov_contracts"
    engine = ocr.get_engine()
    done = 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if done >= limit:
            break
        text_dir = folder / "text"
        pages_dir = folder / "pages"
        if not pages_dir.exists():
            continue
        existing = sum(
            len((text_dir / p.name.replace(".jpeg", ".txt").replace(".png", ".txt")).read_text(encoding="utf-8", errors="ignore"))
            for p in sorted(pages_dir.glob("page_*.jpeg"))[:1]
            if (text_dir / p.name.replace(".jpeg", ".txt")).exists()
        ) if text_dir.exists() else 0
        if existing > 100:
            continue  # 已有文字层，不必 OCR
        started = time.time()
        chars = 0
        pages = 0
        for image in sorted(pages_dir.glob("page_*.jpeg")) + sorted(pages_dir.glob("page_*.png")):
            try:
                result = engine.recognize_image(image)
            except Exception as exc:  # noqa: BLE001
                print("  失败", image.name, exc)
                continue
            text = getattr(result, "text", "") or ""
            target = text_dir / (image.stem + ".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            chars += len(text)
            pages += 1
        elapsed = time.time() - started
        done += 1
        print(f"{folder.name} OCR {pages} 页 / {chars} 字 / {elapsed:.1f}s ({elapsed / max(pages, 1):.2f}s 每页)")
    print("完成", done, "份")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
