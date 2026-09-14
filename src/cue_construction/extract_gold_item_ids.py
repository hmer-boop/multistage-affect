import json
from pathlib import Path

# ===== 路径配置（相对路径，适配新电脑）=====
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = PROJECT_DIR / "results" / "gold_label.json"
INPUT_1PERSON_PATH = PROJECT_DIR / "results" / "gold_label_1person.json"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "method_data"
OUTPUT_PATH = OUTPUT_DIR / "gold_item_ids.json"


def extract_ids(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item.get("item_id") for item in data if item.get("item_id") is not None]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ids_base = extract_ids(INPUT_PATH)
    ids_1person = extract_ids(INPUT_1PERSON_PATH)

    # 合并去重，并保持原有先后顺序
    merged_ids = list(dict.fromkeys(ids_base + ids_1person))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_ids, f, ensure_ascii=False, indent=2)

    print(f"已输出合并版 item_id 文件：{OUTPUT_PATH}")
    print(f"5人金标数量：{len(ids_base)}")
    print(f"1人金标数量：{len(ids_1person)}")
    print(f"合并去重后数量：{len(merged_ids)}")


if __name__ == "__main__":
    main()