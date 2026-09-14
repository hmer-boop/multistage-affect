# run_pipeline.py
# -*- coding: utf-8 -*-
"""
一键跑全流程：
1) 运行 clean_data_caption.py -> 生成 items.jsonl、双语标题与清晰图 dataset/images_raw
2) 运行 extract_min_fields.py -> 生成 items_min.jsonl
3) 运行 generate_image_descriptions.py -> 生成 items_with_captions.jsonl
4) 合并 items_with_captions.jsonl 到 items.jsonl -> 输出 items_merged.jsonl（或 --overwrite 直接覆盖 items.jsonl）

用法示例：
python run_pipeline.py --max -1 --overwrite
python run_pipeline.py --max 5  # 先少量跑5条看效果
"""

import argparse
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

# ---------------------------
# 基本路径
# ---------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = next(p for p in SCRIPT_DIR.parents if (p / "requirements.txt").exists() and (p / "src").exists())
DATASET_DIR = BASE_DIR / "dataset"
META_DIR = DATASET_DIR / "metadata"
IMAGES_RAW = DATASET_DIR / "images_raw"

ITEMS_JSONL = META_DIR / "items.jsonl"
ITEMS_MIN = META_DIR / "items_min.jsonl"
ITEMS_WITH_CAP = META_DIR / "items_with_captions.jsonl"
ITEMS_MERGED = META_DIR / "items_merged.jsonl"
ITEMS_JSONL_BAK = META_DIR / "items.jsonl.bak"

# ---------------------------
# 工具函数
# ---------------------------
def run_step(pyfile: Path, extra_args=None, py_exe=None):
    if extra_args is None:
        extra_args = []
    if py_exe is None:
        py_exe = sys.executable
    cmd = [py_exe, str(pyfile)] + list(map(str, extra_args))
    print(f"\n[RUN] {cmd}")
    subprocess.run(cmd, check=True)
    print(f"[OK ] {pyfile.name} 完成")

def ensure_dirs():
    META_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_RAW.mkdir(parents=True, exist_ok=True)

def detect_id_key(rec: Dict[str, Any]) -> str:
    """尝试识别主键字段名：优先 'item_id'，其次 'id'。"""
    if "item_id" in rec:
        return "item_id"
    if "id" in rec:
        return "id"
    # 兜底：如果都没有，抛错
    raise KeyError("记录中找不到 'item_id' 或 'id' 字段，请检查 JSONL 结构。")

def load_jsonl_to_dict(path: Path) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """加载 JSONL -> dict[id] = record，并返回使用的主键字段名。"""
    db: Dict[str, Dict[str, Any]] = {}
    id_key_used = None
    if not path.exists():
        raise FileNotFoundError(f"未找到文件：{path}")
    cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if id_key_used is None:
                id_key_used = detect_id_key(rec)
            rid = rec[id_key_used]
            db[str(rid)] = rec
            cnt += 1
    print(f"[LOAD] {path.name}: {cnt} 行（键：{id_key_used}）")
    return db, id_key_used or "item_id"

def write_jsonl_from_dict(path: Path, db: Dict[str, Dict[str, Any]]):
    cnt = 0
    with path.open("w", encoding="utf-8") as f:
        for _, rec in db.items():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt += 1
    print(f"[SAVE] {path.name}: {cnt} 行")

def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def merge_captions_into_items(items_path: Path, with_caps_path: Path, out_path: Path, overwrite: bool):
    print("\n[STEP5] 开始合并 captions -> items ...")
    items_db, id_key_items = load_jsonl_to_dict(items_path)
    caps_db, id_key_caps = load_jsonl_to_dict(with_caps_path)

    # 提示：如果两个文件的主键字段名不同，尝试同步为 items 的键名
    if id_key_caps != id_key_items:
        # 重映射 captions 数据的 id 键
        new_caps_db: Dict[str, Dict[str, Any]] = {}
        for _, rec in caps_db.items():
            rid = str(rec[id_key_caps])
            rec[id_key_items] = rid
            new_caps_db[rid] = rec
        caps_db = new_caps_db

    updated = 0
    added = 0
    for rid, crec in caps_db.items():
        if rid in items_db:
            # 仅更新 caption 字段（若存在）
            cap_zh = crec.get("caption_zh")
            cap_en = crec.get("caption_en")
            # 也兼容有的脚本可能用其他键名
            cap_zh = cap_zh if cap_zh is not None else crec.get("caption_cn")
            cap_en = cap_en if cap_en is not None else crec.get("caption_en_us")

            changed = False
            if cap_zh is not None:
                items_db[rid]["caption_zh"] = cap_zh
                changed = True
            if cap_en is not None:
                items_db[rid]["caption_en"] = cap_en
                changed = True
            if changed:
                updated += 1
        else:
            # 如果原 items.jsonl 中没有该条，选择“追加合并”
            items_db[rid] = crec
            added += 1

    print(f"[MERGE] 更新 {updated} 条；追加 {added} 条。")

    # 输出
    if overwrite:
        # 先备份
        if items_path.exists():
            shutil.copyfile(items_path, ITEMS_JSONL_BAK)
            print(f"[BAK ] 备份原始 items.jsonl -> {ITEMS_JSONL_BAK.name}")
        write_jsonl_from_dict(items_path, items_db)
        print("[DONE] 已覆盖写回 items.jsonl")
    else:
        write_jsonl_from_dict(out_path, items_db)
        print(f"[DONE] 已写出 {out_path.name}")

# ---------------------------
# 主流程
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="一键合并运行 1~4 步并合并 captions 回 items.jsonl")
    parser.add_argument("--python", type=str, default=sys.executable, help="解释器路径（默认当前环境）")
    parser.add_argument("--max", type=int, default=-1,
                        help="Step4 本轮最多处理的图片数；-1 表示全量（默认：5）")
    parser.add_argument("--overwrite", action="store_true",
                        help="将合并后的结果直接覆盖写回 items.jsonl（谨慎）")
    parser.add_argument("--skip1", action="store_true", help="跳过 Step1 clean_data_caption.py")
    parser.add_argument("--skip3", action="store_true", help="跳过 Step2 extract_min_fields.py")
    parser.add_argument("--skip4", action="store_true", help="跳过 Step3 generate_image_descriptions.py")
    args = parser.parse_args()

    ensure_dirs()

    py_exe = args.python

    step1 = SCRIPT_DIR / "clean_data_caption.py"
    step3 = SCRIPT_DIR / "extract_min_fields.py"
    step4 = SCRIPT_DIR / "generate_image_descriptions.py"

    # ---- Step 1
    if not args.skip1:
        if not step1.exists():
            raise FileNotFoundError(f"未找到脚本：{step1}")
        run_step(step1, [], py_exe)
    else:
        print("[SKIP] 跳过 Step1")

    # 简单存在性检查
    if not ITEMS_JSONL.exists():
        raise FileNotFoundError(f"期望生成的 {ITEMS_JSONL} 不存在，请检查 Step1。")

    # ---- Step 2
    if not args.skip3:
        if not step3.exists():
            raise FileNotFoundError(f"未找到脚本：{step3}")
        run_step(step3, [], py_exe)
    else:
        print("[SKIP] 跳过 Step2")

    # ---- Step 3（带 --max，-1=全量）
    if not args.skip4:
        if not step4.exists():
            raise FileNotFoundError(f"未找到脚本：{step4}")

        # 统计 Step3 运行前已有多少条
        before_cnt = count_jsonl_lines(ITEMS_WITH_CAP)

        # 兼容 generate_image_descriptions.py 的 argparse 设计
        extra = ["--max", str(args.max)]
        run_step(step4, extra, py_exe)

        # 统计 Step4 运行后有多少条
        after_cnt = count_jsonl_lines(ITEMS_WITH_CAP)

        print(
            f"[STAT] {ITEMS_WITH_CAP.name} 本轮新增 {after_cnt - before_cnt} 条 "
            f"(运行前 {before_cnt} → 运行后 {after_cnt})"
        )
    else:
        print("[SKIP] 跳过 Step3")

    # ---- Step 4 合并
    if not ITEMS_WITH_CAP.exists():
        raise FileNotFoundError(f"期望生成的 {ITEMS_WITH_CAP} 不存在，请检查 Step3。")

    # ---- Step 4 合并（加：本轮新增统计）
    before_items = count_jsonl_lines(ITEMS_JSONL)
    before_merged = count_jsonl_lines(ITEMS_MERGED)

    merge_captions_into_items(
        items_path=ITEMS_JSONL,
        with_caps_path=ITEMS_WITH_CAP,
        out_path=ITEMS_MERGED,
        overwrite=args.overwrite
    )

    # overwrite=True 时会覆盖写回 items.jsonl；overwrite=False 时会写出 items_merged.jsonl
    after_items = count_jsonl_lines(ITEMS_JSONL)
    after_merged = count_jsonl_lines(ITEMS_MERGED)

    if args.overwrite:
        print(
            f"[STAT] {ITEMS_JSONL.name} 本轮新增 {after_items - before_items} 条 "
            f"(运行前 {before_items} → 运行后 {after_items})"
        )
    else:
        print(
            f"[STAT] {ITEMS_MERGED.name} 本轮新增 {after_merged - before_merged} 条 "
            f"(运行前 {before_merged} → 运行后 {after_merged})"
        )

    print("\n🎯 全流程完成！")
    print(f" - 清晰图目录: {IMAGES_RAW}")
    print(f" - items.jsonl: {ITEMS_JSONL}")
    print(f" - items_min.jsonl: {ITEMS_MIN}（由 Step2 生成）")
    print(f" - items_with_captions.jsonl: {ITEMS_WITH_CAP}")
    if args.overwrite:
        print(f" - 已覆盖写回 items.jsonl；原始备份: {ITEMS_JSONL_BAK.name}")
    else:
        print(f" - 合并输出: {ITEMS_MERGED}")

if __name__ == "__main__":
    main()
