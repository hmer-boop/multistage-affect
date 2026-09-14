# extract_stage_inputs_filtered.py
import json
from pathlib import Path

# ===== 路径配置 =====
BASE_DIR = Path(__file__).resolve().parent  # 脚本所在目录
META_DIR = BASE_DIR / "dataset" / "metadata"  # 相对路径 dataset/metadata

INPUT_FILE = META_DIR / "items_merged.jsonl"  # 输入文件

# 原 gold_label 对应输出
OUTPUT_STAGE1_2 = META_DIR / "items_stage1_2.json"
OUTPUT_STAGE3 = META_DIR / "items_stage3.json"

# 新增 round2 对应输出
OUTPUT_STAGE1_2_round2 = META_DIR / "items_stage1_2_round2.json"
OUTPUT_STAGE3_round2 = META_DIR / "items_stage3_round2.json"

# 金标文件路径
GOLD_PATH = BASE_DIR / "results" / "gold_label.json"
GOLD_round2_PATH = BASE_DIR / "results" / "gold_label_round2.json"


def load_gold_ids(gold_path: Path):
    """读取金标文件，返回其中全部 item_id 集合"""
    with open(gold_path, "r", encoding="utf-8") as f_gold:
        gold_data = json.load(f_gold)

    gold_ids = {str(x.get("item_id")) for x in gold_data if x.get("item_id")}
    return gold_ids


def extract_records_by_ids(input_file: Path, target_ids: set):
    """从 items_merged.jsonl 中筛选 target_ids，对应输出阶段1/2与阶段3数据"""
    out12_list = []
    out3_list = []

    with open(input_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            item_id = str(item.get("item_id"))

            if item_id not in target_ids:
                continue

            # 阶段1/2字段
            rec12 = {
                "item_id": item_id,
                "path_raw": item.get("path_raw")
            }
            out12_list.append(rec12)

            # 阶段3字段
            rec3 = {
                "item_id": item_id,
                "path_raw": item.get("path_raw"),
                "title_en": item.get("title_en"),
                "caption_en": item.get("caption_en")
            }
            out3_list.append(rec3)

    return out12_list, out3_list


def save_json(data, output_path: Path):
    """保存 JSON"""
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=2)


def main():
    # ===== 第一套：gold_label.json =====
    gold_ids = load_gold_ids(GOLD_PATH)
    out12_list, out3_list = extract_records_by_ids(INPUT_FILE, gold_ids)

    save_json(out12_list, OUTPUT_STAGE1_2)
    save_json(out3_list, OUTPUT_STAGE3)

    print(f"完成！gold_label.json -> 输出 {len(out12_list)} 条到 {OUTPUT_STAGE1_2}")
    print(f"完成！gold_label.json -> 输出 {len(out3_list)} 条到 {OUTPUT_STAGE3}")

    # ===== 第二套：gold_label_round2.json =====
    gold_round2_ids = load_gold_ids(GOLD_round2_PATH)
    out12_list_round2, out3_list_round2 = extract_records_by_ids(INPUT_FILE, gold_round2_ids)

    save_json(out12_list_round2, OUTPUT_STAGE1_2_round2)
    save_json(out3_list_round2, OUTPUT_STAGE3_round2)

    print(f"完成！gold_label_round2.json -> 输出 {len(out12_list_round2)} 条到 {OUTPUT_STAGE1_2_round2}")
    print(f"完成！gold_label_round2.json -> 输出 {len(out3_list_round2)} 条到 {OUTPUT_STAGE3_round2}")


if __name__ == "__main__":
    main()
