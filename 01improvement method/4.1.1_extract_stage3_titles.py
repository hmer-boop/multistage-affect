# -*- coding: utf-8 -*-
"""
从 summary.json + summary_1person.json 中抽取 gold_item_ids.json 覆盖到的金标ID对应字段：
- id
- item_id
- title_zh
- title_en

输出：stage3_titles_only.json
支持：断点续跑 / 防重复 / 定期落盘 / 进度显示

注意：
1) 本脚本只保留标题相关信息，不输出 path_raw / image_path / 描述等任何其他字段
2) 这样可确保下一步模型输入仅来自标题文本
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "01improvement method" / "method data"
RESULTS_DIR = PROJECT_ROOT / "results"

GOLD_IDS_PATH = str(DATA_DIR / "gold_item_ids.json")
SUMMARY_PATH = str(RESULTS_DIR / "summary.json")
SUMMARY_1PERSON_PATH = str(RESULTS_DIR / "summary_1person.json")

OUT_DIR = str(DATA_DIR)
OUT_PATH = os.path.join(OUT_DIR, "stage3_1titles_only.json")
OUT_TMP_PATH = OUT_PATH + ".tmp"

SAVE_EVERY = 200


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _iter_summary_items(summary_obj: Any):
    if isinstance(summary_obj, list):
        for it in summary_obj:
            if isinstance(it, dict):
                yield it
    elif isinstance(summary_obj, dict):
        if "items" in summary_obj and isinstance(summary_obj["items"], list):
            for it in summary_obj["items"]:
                if isinstance(it, dict):
                    yield it
        else:
            for k, v in summary_obj.items():
                if isinstance(v, dict):
                    if "id" not in v and "item_id" not in v:
                        v = dict(v)
                        v["id"] = k
                    yield v
    else:
        raise ValueError("summary.json / summary_1person.json 的结构不支持，既不是 list 也不是 dict")


def _load_existing_out(out_path: str) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    if not os.path.exists(out_path):
        return {}, set()

    try:
        obj = _load_json(out_path)
        if isinstance(obj, list):
            mp = {}
            for r in obj:
                if isinstance(r, dict) and "id" in r:
                    mp[str(r["id"]).strip()] = r
            return mp, set(mp.keys())
        elif isinstance(obj, dict):
            mp = {str(k).strip(): v for k, v in obj.items() if isinstance(v, dict)}
            return mp, set(mp.keys())
        else:
            return {}, set()
    except Exception:
        raise RuntimeError(f"已有输出文件读取失败：{out_path}，请检查是否为合法JSON")


def _build_summary_map(*summary_objs: Any) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for summary_obj in summary_objs:
        for it in _iter_summary_items(summary_obj):
            sid = _safe_get(it, ["id", "item_id", "wulunnage_id"], default=None)
            if sid is None:
                continue
            sid = str(sid).strip()
            if not sid:
                continue

            if sid not in merged:
                merged[sid] = dict(it)
            else:
                old = merged[sid]
                for k, v in it.items():
                    if (k not in old or old[k] in (None, "", [])) and v not in (None, "", []):
                        old[k] = v

    return merged


def _atomic_save_list(id2rec: Dict[str, Dict[str, Any]], tmp_path: str, final_path: str):
    def sort_key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    out_list = [id2rec[k] for k in sorted(id2rec.keys(), key=sort_key)]

    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def main():
    t0 = time.time()

    gold_ids = {str(x).strip() for x in _load_json(GOLD_IDS_PATH) if str(x).strip()}

    summary = _load_json(SUMMARY_PATH)
    summary_1person = _load_json(SUMMARY_1PERSON_PATH)
    summary_map = _build_summary_map(summary, summary_1person)

    existing_map, done_ids = _load_existing_out(OUT_PATH)

    target_ids = gold_ids
    remaining_ids = target_ids - done_ids

    print(f"[INFO] gold ids 总数: {len(gold_ids)}")
    print(f"[INFO] 已在输出中存在(断点续跑命中): {len(done_ids)}")
    print(f"[INFO] 待抽取: {len(remaining_ids)}")

    found = 0
    scanned = 0
    updated_map = dict(existing_map)

    pbar = tqdm(total=len(remaining_ids), desc="Extracting titles only", unit="id")

    for sid in list(remaining_ids):
        scanned += 1

        it = summary_map.get(sid)
        if not it:
            continue

        # 只保留标题相关字段，彻底不输出任何图片/描述路径信息
        rec = {
            "id": sid,
            "item_id": _safe_get(it, ["item_id", "id"], default=sid),
            "title_zh": _safe_get(it, ["title_zh", "title", "title_cn", "name_zh"], default=""),
            "title_en": _safe_get(it, ["title_en", "title_en_us", "name_en"], default=""),
        }

        updated_map[sid] = rec
        found += 1
        remaining_ids.remove(sid)
        pbar.update(1)

        if found % SAVE_EVERY == 0:
            _atomic_save_list(updated_map, OUT_TMP_PATH, OUT_PATH)

        if not remaining_ids:
            break

    pbar.close()

    _atomic_save_list(updated_map, OUT_TMP_PATH, OUT_PATH)

    dt = time.time() - t0
    print(f"[DONE] 扫描目标 id 数: {scanned}")
    print(f"[DONE] 新增抽取到的金标ID数: {found}")
    print(f"[DONE] 输出文件: {OUT_PATH}")
    print(f"[DONE] 未在两个 summary 找到的 gold id 数: {len(gold_ids - set(updated_map.keys()))}")
    print(f"[DONE] 耗时: {dt:.2f}s")


if __name__ == "__main__":
    main()
