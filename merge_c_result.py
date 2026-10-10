"""Merge a single-query worker result into the panel JSON.

Usage:
    python merge_c_result.py TMP_FILE OUT_FILE
"""
import json
import sys
from pathlib import Path


def main():
    tmp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    rec = json.loads(tmp.read_text(encoding="utf-8"))
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
    else:
        data = {"records": [], "model": "gpt-5", "cap": 4096}
    # Drop any prior record for this query_id
    data["records"] = [r for r in data["records"] if r.get("query_id") != rec.get("query_id")]
    data["records"].append(rec)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {len(data['records'])} records to {out.name}")


if __name__ == "__main__":
    main()
