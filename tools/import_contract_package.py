"""Import a preserved contract package without live network or model calls."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zhilv_qiling_v2.contract_packages import import_contract_package, prepare_package_document

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root")
    parser.add_argument("--store-root")
    parser.add_argument("--package-id")
    parser.add_argument("--check-parse", action="store_true")
    args = parser.parse_args()
    result = import_contract_package(args.source_root, args.store_root, args.package_id)
    if args.check_parse:
        document = prepare_package_document(result["package_id"], store_root=args.store_root)
        result["parse_check"] = {"pages": document.stats["pages"], "blocks": len(document.blocks),
                                 "clauses": len(document.clauses), "characters": len(document.full_text),
                                 "source_ids": sorted({b.metadata["source_id"] for b in document.blocks})}
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
