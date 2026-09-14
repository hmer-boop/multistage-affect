# 1_extract_min_fields.py
# 从 dataset/metadata/items.jsonl 中提取 item_id 和 path_raw
# 输出到 dataset/metadata/items_min.jsonl

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
META_DIR = BASE_DIR / "dataset" / "metadata"

SRC_JSONL = META_DIR / "items.jsonl"
OUT_JSONL = META_DIR / "items_min.jsonl"

def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    rows = load_jsonl(SRC_JSONL)
    if not rows:
        print(f"未找到或为空：{SRC_JSONL.resolve()}")
        return

    # 如果已存在，先清空
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()

    seen = set()
    kept = 0
    for r in rows:
        item_id = r.get("item_id")
        path_raw = r.get("path_raw")
        if not item_id or not path_raw:
            continue
        if item_id in seen:
            continue
        seen.add(item_id)
        append_jsonl(OUT_JSONL, {"item_id": item_id, "path_raw": path_raw})
        kept += 1

    print(f"OK：写出 {kept} 行 -> {OUT_JSONL.resolve()}")

if __name__ == "__main__":
    main()
