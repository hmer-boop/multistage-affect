# -*- coding: utf-8 -*-
"""
4.1.3_stage3_all_information.py

用途：
把 4.2 融合标签所需的信息整理为一个完整 JSON：
1) stage3_text_only_labels.json           -> 文本标签信息
2) summary.json + summary_1person.json    -> path_raw（原图路径）
3) stage2_gpt_labels4.json                -> stage2_label

输出：
- 4.1.3_stage3_all_information.json

特点：
- 断点安全（先写 tmp 再覆盖）
- 防重复（按 id 合并）
- 进度显示
- 定期落盘
- 打印缺失项统计
"""

import os
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Iterable

from tqdm import tqdm


# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "01improvement method" / "method data")
RESULTS_DIR = str(PROJECT_ROOT / "results")

TEXT_ONLY_PATH = os.path.join(DATA_DIR, "stage3_text_only_labels.json")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")
SUMMARY_1PERSON_PATH = os.path.join(RESULTS_DIR, "summary_1person.json")
STAGE2_PATH = os.path.join(DATA_DIR, "stage2_gpt_labels4.json")

OUT_PATH = os.path.join(DATA_DIR, "stage3_all_information.json")
OUT_TMP_PATH = OUT_PATH + ".tmp"

SAVE_EVERY = 200


# =========================
# 基础工具
# =========================
def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"文件为空：{path}")
    return json.loads(content)


def atomic_save(obj: Any, tmp_path: str, final_path: str):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def normalize_id(x: Any) -> str:
    return str(x).strip()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def iter_records(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    兼容：
    - list[dict]
    - {"items": [...]}
    - {"id1": {...}, "id2": {...}}
    """
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
        return

    if isinstance(obj, dict):
        if "items" in obj and isinstance(obj["items"], list):
            for x in obj["items"]:
                if isinstance(x, dict):
                    yield x
            return

        for k, v in obj.items():
            if isinstance(v, dict):
                if "id" not in v:
                    v = dict(v)
                    v["id"] = k
                yield v
        return

    raise ValueError("JSON 顶层结构不支持，必须是 list / {'items': [...]} / {'id': {...}}")


def sort_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(r: Dict[str, Any]):
        sid = safe_str(r.get("id", ""))
        try:
            return (0, int(sid))
        except Exception:
            return (1, sid)
    return sorted(records, key=sort_key)


# =========================
# 各来源建表
# =========================
def build_text_only_map(path: str) -> Dict[str, Dict[str, Any]]:
    """
    以 stage3_text_only_labels.json 为主表
    """
    obj = load_json(path)
    mp: Dict[str, Dict[str, Any]] = {}

    for rec in iter_records(obj):
        _id = normalize_id(rec.get("id", ""))
        if not _id:
            continue

        mp[_id] = {
            "id": _id,
            "title_zh": safe_str(rec.get("title_zh")),
            "title_en": safe_str(rec.get("title_en")),
            "text_label": safe_str(rec.get("text_label")),
            "text_trigger": safe_str(rec.get("text_trigger")),
            "text_confidence": rec.get("text_confidence", 0.0) if rec.get("text_confidence", None) is not None else 0.0,
            "text_error": safe_str(rec.get("text_error")),
            "model": safe_str(rec.get("model")),
            "ts": safe_str(rec.get("ts")),
            # 待补
            "path_raw": "",
            "stage2_label": "",
        }

    return mp


def build_summary_path_map(*paths: str) -> Dict[str, str]:
    """
    从 summary.json / summary_1person.json 中读取 path_raw
    """
    path_map: Dict[str, str] = {}

    for path in paths:
        obj = load_json(path)
        for rec in iter_records(obj):
            _id = normalize_id(
                rec.get("id")
                or rec.get("item_id")
                or rec.get("wulunnage_id")
                or ""
            )
            if not _id:
                continue

            path_raw = safe_str(
                rec.get("path_raw")
                or rec.get("raw_path")
                or rec.get("image_path")
                or rec.get("path")
            )

            if path_raw and _id not in path_map:
                path_map[_id] = path_raw
            elif path_raw and not path_map.get(_id):
                path_map[_id] = path_raw

    return path_map


def build_stage2_map(path: str) -> Dict[str, str]:
    """
    从 stage2_gpt_labels4.json 读取 stage2_label
    """
    obj = load_json(path)
    mp: Dict[str, str] = {}

    for rec in iter_records(obj):
        _id = normalize_id(rec.get("id", ""))
        if not _id:
            continue

        stage2_label = safe_str(rec.get("stage2_label"))
        if stage2_label:
            mp[_id] = stage2_label

    return mp


# =========================
# 主逻辑
# =========================
def validate_record(rec: Dict[str, Any]) -> List[str]:
    required_fields = [
        "id",
        "title_zh",
        "title_en",
        "text_label",
        "text_trigger",
        "text_confidence",
        "text_error",
        "model",
        "ts",
        "path_raw",
        "stage2_label",
    ]
    missing = []

    for k in required_fields:
        if k not in rec:
            missing.append(k)
            continue
        if k == "text_confidence":
            if rec[k] is None:
                missing.append(k)
        else:
            if rec[k] is None:
                missing.append(k)

    return missing


def main():
    t0 = time.time()

    print("[LOAD] 读取 stage3_text_only_labels.json ...")
    main_map = build_text_only_map(TEXT_ONLY_PATH)
    print(f"[INFO] text_only 条目数: {len(main_map)}")

    print("[LOAD] 读取 summary.json / summary_1person.json ...")
    summary_path_map = build_summary_path_map(SUMMARY_PATH, SUMMARY_1PERSON_PATH)
    print(f"[INFO] 可补 path_raw 的条目数: {len(summary_path_map)}")

    print("[LOAD] 读取 stage2_gpt_labels4.json ...")
    stage2_map = build_stage2_map(STAGE2_PATH)
    print(f"[INFO] 可补 stage2_label 的条目数: {len(stage2_map)}")

    ids = list(main_map.keys())
    total = len(ids)

    miss_counter = Counter()
    miss_examples = []

    pbar = tqdm(total=total, desc="Merging all information", unit="item")

    out_list: List[Dict[str, Any]] = []

    for idx, _id in enumerate(ids, start=1):
        rec = dict(main_map[_id])

        # 补 path_raw
        rec["path_raw"] = summary_path_map.get(_id, "")

        # 补 stage2_label
        rec["stage2_label"] = stage2_map.get(_id, "")

        # 字段兜底，确保每条都存在这些键
        rec = {
            "id": safe_str(rec.get("id")),
            "title_zh": safe_str(rec.get("title_zh")),
            "title_en": safe_str(rec.get("title_en")),
            "text_label": safe_str(rec.get("text_label")),
            "text_trigger": safe_str(rec.get("text_trigger")),
            "text_confidence": rec.get("text_confidence", 0.0) if rec.get("text_confidence", None) is not None else 0.0,
            "text_error": safe_str(rec.get("text_error")),
            "model": safe_str(rec.get("model")),
            "ts": safe_str(rec.get("ts")),
            "path_raw": safe_str(rec.get("path_raw")),
            "stage2_label": safe_str(rec.get("stage2_label")),
        }

        # 缺失统计
        missing_keys = validate_record(rec)
        for k in missing_keys:
            miss_counter[f"{k} 字段不存在或为 None"] += 1

        if not rec["path_raw"]:
            miss_counter["path_raw 为空"] += 1
            if len(miss_examples) < 20:
                miss_examples.append((_id, "path_raw", "未在 summary.json / summary_1person.json 中找到"))

        if not rec["stage2_label"]:
            miss_counter["stage2_label 为空"] += 1
            if len(miss_examples) < 20:
                miss_examples.append((_id, "stage2_label", "未在 stage2_gpt_labels4.json 中找到"))

        if not rec["text_label"]:
            reason = rec["text_error"] or "stage3_text_only_labels.json 中 text_label 为空"
            miss_counter[f"text_label 为空，原因是：{reason}"] += 1
            if len(miss_examples) < 20:
                miss_examples.append((_id, "text_label", reason))

        out_list.append(rec)

        if idx % SAVE_EVERY == 0:
            atomic_save(sort_records(out_list), OUT_TMP_PATH, OUT_PATH)

        pbar.update(1)

    pbar.close()

    out_list = sort_records(out_list)
    atomic_save(out_list, OUT_TMP_PATH, OUT_PATH)

    print(f"[DONE] 输出文件：{OUT_PATH}")
    print(f"[DONE] 总条目数：{len(out_list)}")

    if not miss_counter:
        print("[CHECK] 所有条目关键字段均已齐全。")
    else:
        total_missing = sum(miss_counter.values())
        print(f"[CHECK] 发现 {total_missing} 处缺失/异常：")
        for k, v in miss_counter.items():
            print(f"  - {v} 条：{k}")

        if miss_examples:
            print("[CHECK] 缺失样例（最多20条）：")
            for _id, field_name, reason in miss_examples:
                print(f"  - id={_id} 缺少 {field_name}，原因是：{reason}")

    dt = time.time() - t0
    print(f"[TIME] 耗时：{dt:.2f}s")


if __name__ == "__main__":
    main()
