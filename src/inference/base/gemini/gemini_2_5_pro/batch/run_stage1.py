# -*- coding: utf-8 -*-
"""
8.1_Gemini25pro_run_label1_batch.py

Gemini 2.5 Pro Stage1 baseline 批处理版（RPD 优化）。

设计目标
1) 保留原 stage1 baseline 的任务定义：只看图像，输出 8 类单标签。
2) 改为 batch 模式：每次请求处理多张图（默认 10 张），以减少每日请求次数。
3) 优先复用 Gemini Files API：同一张 512 图像上传一次，后续批次复用。
4) 自动生成 master / batch manifests，并支持断点续跑。
5) 自动读取已有单图版 stage1 结果目录，已完成样本默认跳过不重跑。
6) 已有结果会直接合并复制到本次 batch 输出目录的 merged 结果文件中，便于后续只对这一套文件与金标对照。
7) 最终输出仍保持与原始指标口径兼容的 merged 结果文件：
   - results_stage1.txt / json / missing_report_stage1.json
   - results_stage1_round2.txt / json / missing_report_stage1_round2.json

建议放置目录：
src/inference/base/gemini/gemini_2_5_pro/batch/
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_BATCH_SIZE = 10
DEFAULT_IMAGE_DIR = Path("dataset_hf_upload/images_cloud_512")
FILE_TTL_HOURS = 47  # 48h 略留缓冲


# =========================================================
# 路径
# =========================================================
def find_project_root(start_dir: Path) -> Path:
    candidates = [start_dir, *start_dir.parents]
    for p in candidates:
        if (p / "dataset" / "metadata").exists():
            return p
    raise FileNotFoundError("未找到项目根目录：请确认脚本位于仓库内部。")


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(BASE_DIR)
META_DIR = PROJECT_ROOT / "dataset" / "metadata"

INPUT_PATH = META_DIR / "items_stage1_2.json"
INPUT_PATH_ROUND2 = META_DIR / "items_stage1_2_round2.json"

SHARED_DIR = BASE_DIR.parent / "shared"
MANIFEST_MASTER_DIR = BASE_DIR / "manifests" / "master"
MANIFEST_BATCH_DIR = BASE_DIR / "manifests" / "batches"
OUT_DIR = BASE_DIR / "result Gemini_stage1"

REGISTRY_PATH = SHARED_DIR / "gemini_files_registry_512.json"

TXT_OUT = OUT_DIR / "results_stage1.txt"
JSON_OUT = OUT_DIR / "results_stage1.json"
REPORT_PATH = OUT_DIR / "missing_report_stage1.json"

TXT_OUT_ROUND2 = OUT_DIR / "results_stage1_round2.txt"
JSON_OUT_ROUND2 = OUT_DIR / "results_stage1_round2.json"
REPORT_PATH_ROUND2 = OUT_DIR / "missing_report_stage1_round2.json"

LEGACY_RESULT_DIR_DEFAULT = BASE_DIR.parent.parent / "1_baseline" / "result Gemini_stage1"
LEGACY_TXT_OUT = LEGACY_RESULT_DIR_DEFAULT / "results_stage1.txt"
LEGACY_TXT_OUT_ROUND2 = LEGACY_RESULT_DIR_DEFAULT / "results_stage1_round2.txt"

BATCH_OUTPUT_BASE_DIR = OUT_DIR / "base" / "batch_outputs"
BATCH_OUTPUT_ROUND2_DIR = OUT_DIR / "round2" / "batch_outputs"

MASTER_BASE_PATH = MANIFEST_MASTER_DIR / "stage1_baseline_base_master.json"
MASTER_ROUND2_PATH = MANIFEST_MASTER_DIR / "stage1_baseline_round2_master.json"
BATCH_BASE_DIR = MANIFEST_BATCH_DIR / "stage1_baseline_base"
BATCH_ROUND2_DIR = MANIFEST_BATCH_DIR / "stage1_baseline_round2"

for p in [SHARED_DIR, MANIFEST_MASTER_DIR, MANIFEST_BATCH_DIR, OUT_DIR, BATCH_OUTPUT_BASE_DIR, BATCH_OUTPUT_ROUND2_DIR, BATCH_BASE_DIR, BATCH_ROUND2_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# 基础 I/O
# =========================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_auto(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt[0] in "[{":
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                if isinstance(obj.get("items"), list):
                    return obj["items"]
                return [obj]
        except Exception:
            pass
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_json_obj(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def atomic_write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_txt_index(txt_path: Path) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    if not txt_path.exists():
        return idx
    with txt_path.open("r", encoding="utf-8") as f:
        header_seen = False
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            col1, col2 = parts[0].strip(), parts[1].strip()
            if not header_seen:
                header_seen = True
                if col1 == "item_id" and col2 == "stage1":
                    continue
            if col1:
                idx[col1] = col2 or ""
    return idx


def write_txt_from_map(txt_path: Path, data_map: Dict[str, str]):
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("item_id\tstage1\n")
        for iid in sorted(data_map.keys()):
            f.write(f"{iid}\t{data_map[iid]}\n")


def write_json_results(json_path: Path, data_map: Dict[str, str]):
    rows = [{"item_id": iid, "stage1": (lab or None)} for iid, lab in sorted(data_map.items())]
    atomic_write_json(json_path, rows)


def merge_existing_results(primary_map: Dict[str, str], extra_map: Dict[str, str]) -> Tuple[Dict[str, str], int]:
    merged = dict(primary_map)
    added = 0
    for iid, lab in extra_map.items():
        lab = (lab or "").strip()
        if not iid or not lab:
            continue
        if merged.get(iid, "").strip():
            continue
        merged[iid] = lab
        added += 1
    return merged, added


def normalize_label(text: str) -> str:
    if not text:
        return "未知"
    t = text.strip().replace("：", ":").replace(" ", "")
    aliases = {
        "开心": "快乐", "喜悦": "快乐", "高兴": "快乐",
        "满足感": "宁静", "安宁": "宁静", "平静": "宁静",
        "敬佩": "敬畏", "崇敬": "敬畏", "震撼": "敬畏", "惊叹": "敬畏",
        "害怕": "恐惧", "恐怖": "恐惧",
        "愤慨": "愤怒", "生气": "愤怒",
        "厌烦": "厌恶", "恶心": "厌恶",
        "悲痛": "悲伤", "忧伤": "悲伤",
        "惊讶": "惊奇", "诧异": "惊奇", "意外": "惊奇",
    }
    for k, v in aliases.items():
        if k in t:
            return v
    for lab in LABELS:
        if lab in t:
            return lab
    return "未知"


# =========================================================
# 图片定位 / Files API registry
# =========================================================
def guess_local_512_image(image_dir: Path, item_id: str) -> Optional[Path]:
    exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    for ext in exts:
        p = image_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    candidates = list(image_dir.glob(f"{item_id}.*"))
    if not candidates:
        return None
    for ext in exts:
        for c in candidates:
            if c.suffix.lower() == ext:
                return c
    return candidates[0]


def get_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def load_registry(path: Path) -> Dict[str, Any]:
    data = read_json_obj(path, default={})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", 1)
    data.setdefault("files", {})
    return data


def save_registry(path: Path, registry: Dict[str, Any]):
    atomic_write_json(path, registry)


def registry_entry_valid(entry: Dict[str, Any], local_path: Path) -> bool:
    try:
        st = local_path.stat()
    except Exception:
        return False
    if not entry:
        return False
    if entry.get("local_size") != st.st_size:
        return False
    if entry.get("local_mtime_ns") != st.st_mtime_ns:
        return False
    expires_at = entry.get("expires_at")
    if not expires_at:
        return False
    try:
        expires_dt = datetime.fromisoformat(expires_at)
    except Exception:
        return False
    if datetime.now(timezone.utc) >= expires_dt:
        return False
    if not entry.get("name") or not entry.get("uri") or not entry.get("mime_type"):
        return False
    return True


def ensure_file_registered(client, local_path: Path, registry: Dict[str, Any], display_name: str) -> Dict[str, Any]:
    local_key = str(local_path.resolve())
    entry = registry.get("files", {}).get(local_key, {})
    if registry_entry_valid(entry, local_path):
        return entry

    uploaded = client.files.upload(file=str(local_path))
    st = local_path.stat()
    uploaded_at = datetime.now(timezone.utc)
    new_entry = {
        "local_path": local_key,
        "display_name": display_name,
        "local_size": st.st_size,
        "local_mtime_ns": st.st_mtime_ns,
        "uploaded_at": uploaded_at.isoformat(),
        "expires_at": (uploaded_at + timedelta(hours=FILE_TTL_HOURS)).isoformat(),
        "name": getattr(uploaded, "name", None),
        "uri": getattr(uploaded, "uri", None),
        "mime_type": getattr(uploaded, "mime_type", None) or get_mime_type(local_path),
        "state": str(getattr(uploaded, "state", "")),
    }
    registry.setdefault("files", {})[local_key] = new_entry
    return new_entry


# =========================================================
# manifest 构建
# =========================================================
def build_master_manifest(items: List[Dict[str, Any]], dataset_name: str, image_dir: Path) -> Dict[str, Any]:
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("item_id", "")).strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        local_512 = guess_local_512_image(image_dir, item_id)
        rows.append(
            {
                "item_id": item_id,
                "dataset_name": dataset_name,
                "original_path_raw": item.get("path_raw"),
                "image_path_512": str(local_512) if local_512 else None,
                "image_exists_512": bool(local_512 and local_512.exists()),
            }
        )
    return {
        "task": "stage1_baseline",
        "dataset_name": dataset_name,
        "image_dir_512": str(image_dir),
        "generated_at": utc_now_iso(),
        "total_items": len(rows),
        "items": rows,
    }


def write_batch_manifests(master: Dict[str, Any], batch_dir: Path, batch_size: int) -> List[Path]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    for old in batch_dir.glob("*.json"):
        old.unlink()
    items = master.get("items", [])
    total_batches = math.ceil(len(items) / batch_size) if items else 0
    paths: List[Path] = []
    task = master.get("task")
    dataset_name = master.get("dataset_name")
    for i in range(total_batches):
        chunk = items[i * batch_size:(i + 1) * batch_size]
        batch_obj = {
            "task": task,
            "dataset_name": dataset_name,
            "batch_index": i + 1,
            "batch_size": len(chunk),
            "generated_at": utc_now_iso(),
            "items": chunk,
        }
        p = batch_dir / f"{task}_{dataset_name}_batch_{i + 1:04d}.json"
        atomic_write_json(p, batch_obj)
        paths.append(p)
    return paths


# =========================================================
# Gemini 批请求
# =========================================================
STAGE1_BATCH_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "label": {"type": "string", "enum": LABELS},
                },
                "required": ["item_id", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_stage1_batch_intro(batch_items: List[Dict[str, Any]]) -> str:
    ids = [str(x["item_id"]) for x in batch_items]
    ids_text = "\n".join([f"- {iid}" for iid in ids])
    return (
        "你将执行 stage1 baseline 情感标注。\n"
        "你会收到一批图片。每个图片前面都有一个 item_id 文本标记，紧接着就是该图片。\n"
        "你的任务是：仅根据图像内容，模拟人在快速浏览这幅艺术作品时的即时直觉反应。\n"
        f"必须从以下 8 个标签中为每个 item_id 选择 1 个：{LABELS}\n"
        "不要解释，不要补充说明。\n"
        "请严格返回 JSON，对每个 item_id 恰好输出一条结果。\n"
        "本批 item_id 如下：\n"
        f"{ids_text}"
    )


def parse_batch_response(response) -> Tuple[Optional[Dict[str, Any]], str]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed, getattr(response, "text", "") or ""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        try:
            return json.loads(text), text
        except Exception:
            return None, text
    # fallback
    try:
        candidates = getattr(response, "candidates", None) or []
        parts = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                txt = getattr(part, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
        raw = "\n".join(parts).strip()
        if raw:
            return json.loads(raw), raw
        return None, raw
    except Exception:
        return None, ""


def validate_batch_results(batch_items: List[Dict[str, Any]], parsed_obj: Dict[str, Any]) -> Dict[str, str]:
    expected_ids = [str(x["item_id"]) for x in batch_items]
    results = parsed_obj.get("results") if isinstance(parsed_obj, dict) else None
    if not isinstance(results, list):
        raise ValueError("模型返回中缺少 results 数组")

    out: Dict[str, str] = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError(f"results 中存在非对象项: {row!r}")
        item_id = str(row.get("item_id", "")).strip()
        label = normalize_label(str(row.get("label", "")))
        if not item_id:
            raise ValueError(f"存在缺少 item_id 的结果项: {row!r}")
        if label not in LABELS:
            raise ValueError(f"item_id={item_id} 返回非法标签: {label!r}")
        if item_id in out:
            raise ValueError(f"item_id={item_id} 重复返回")
        out[item_id] = label

    missing = [iid for iid in expected_ids if iid not in out]
    extra = [iid for iid in out.keys() if iid not in expected_ids]
    if missing or extra:
        raise ValueError(f"返回 item_id 与请求不一致 | missing={missing} | extra={extra}")
    return out


def run_one_batch(client, types_module, registry: Dict[str, Any], batch_items: List[Dict[str, Any]], model: str, retry_sleep: float, max_retries: int) -> Dict[str, Any]:
    intro = build_stage1_batch_intro(batch_items)
    contents: List[Any] = [types_module.Part.from_text(text=intro)]

    for item in batch_items:
        item_id = str(item["item_id"])
        local_path = Path(item["image_path_512"])
        reg = ensure_file_registered(client, local_path, registry, display_name=item_id)
        contents.append(types_module.Part.from_text(text=f"item_id: {item_id}"))
        contents.append(types_module.Part.from_uri(file_uri=reg["uri"], mime_type=reg["mime_type"]))

    last_error: Optional[str] = None
    last_raw_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": STAGE1_BATCH_RESPONSE_JSON_SCHEMA,
                    "temperature": 0,
                },
            )
            parsed_obj, raw_text = parse_batch_response(response)
            last_raw_text = raw_text
            if parsed_obj is None:
                raise ValueError("模型未返回可解析 JSON")
            labels_map = validate_batch_results(batch_items, parsed_obj)
            usage = getattr(response, "usage_metadata", None)
            usage_dict = None
            if usage is not None:
                try:
                    usage_dict = json.loads(json.dumps(usage, default=lambda o: getattr(o, "__dict__", str(o))))
                except Exception:
                    usage_dict = str(usage)
            return {
                "status": "ok",
                "labels_map": labels_map,
                "parsed": parsed_obj,
                "raw_text": raw_text,
                "usage_metadata": usage_dict,
            }
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
            else:
                break

    return {
        "status": "error",
        "error": last_error,
        "raw_text": last_raw_text,
    }


# =========================================================
# 主流程：单数据集
# =========================================================
def process_dataset(
    *,
    client,
    types_module,
    registry: Dict[str, Any],
    dataset_name: str,
    input_path: Path,
    master_path: Path,
    batch_manifest_dir: Path,
    batch_output_dir: Path,
    merged_txt_path: Path,
    merged_json_path: Path,
    report_path: Path,
    image_dir: Path,
    batch_size: int,
    model: str,
    max_retries: int,
    retry_sleep: float,
    max_batches: int,
    regen_manifests: bool,
    legacy_txt_path: Optional[Path] = None,
):
    print(f"\n===== stage1 baseline | {dataset_name} =====")
    items = read_json_auto(input_path)
    print(f"[load:{dataset_name}] metadata loaded = {len(items)} | {input_path}")

    if regen_manifests or not master_path.exists():
        master = build_master_manifest(items, dataset_name=dataset_name, image_dir=image_dir)
        atomic_write_json(master_path, master)
        print(f"[manifest:{dataset_name}] master generated -> {master_path}")
    else:
        master = read_json_obj(master_path, default={})
        print(f"[manifest:{dataset_name}] master reused -> {master_path}")

    batch_paths = write_batch_manifests(master, batch_manifest_dir, batch_size=batch_size)
    print(f"[manifest:{dataset_name}] batches generated = {len(batch_paths)} | batch_size = {batch_size}")

    existing = read_txt_index(merged_txt_path)
    current_done_count = len([iid for iid, lab in existing.items() if lab.strip()])

    legacy_added = 0
    if legacy_txt_path and legacy_txt_path.exists():
        legacy_existing = read_txt_index(legacy_txt_path)
        legacy_nonempty = {iid: lab for iid, lab in legacy_existing.items() if (lab or "").strip()}
        existing, legacy_added = merge_existing_results(existing, legacy_nonempty)
        # 无论是否新增，都把当前已知结果写入本次 batch 输出目录，确保后续只看这一套 merged 文件即可。
        write_txt_from_map(merged_txt_path, existing)
        write_json_results(merged_json_path, existing)
        print(f"[resume:{dataset_name}] legacy txt loaded = {len(legacy_nonempty)} | merged_add = {legacy_added} | synced_to_current_outputs = True | {legacy_txt_path}")
    else:
        if legacy_txt_path:
            print(f"[resume:{dataset_name}] legacy txt not found -> {legacy_txt_path}")

    done_ids = {iid for iid, lab in existing.items() if lab.strip()}
    all_ids = [str(x.get("item_id")) for x in master.get("items", []) if x.get("item_id")]
    print(f"[resume:{dataset_name}] done_ids in batch merged txt = {current_done_count} | total done after legacy merge = {len(done_ids)}")

    processed_batches = 0
    for batch_path in batch_paths:
        batch = read_json_obj(batch_path, default={})
        batch_items_all = batch.get("items", []) if isinstance(batch, dict) else []
        batch_items = [x for x in batch_items_all if str(x.get("item_id")) not in done_ids]
        batch_output_path = batch_output_dir / batch_path.name

        if batch_output_path.exists() and batch_items:
            prev = read_json_obj(batch_output_path, default={})
            if isinstance(prev, dict) and prev.get("status") == "ok":
                prev_results = prev.get("results", [])
                prev_map = {str(r.get("item_id")): str(r.get("label")) for r in prev_results if isinstance(r, dict)}
                needed_ids = {str(x.get("item_id")) for x in batch_items}
                if needed_ids and needed_ids.issubset(prev_map.keys()):
                    for iid in needed_ids:
                        existing[iid] = normalize_label(prev_map[iid])
                    done_ids.update(needed_ids)
                    print(f"[skip-batch:{dataset_name}] reuse existing batch output -> {batch_output_path.name}")
                    continue

        if not batch_items:
            print(f"[skip-batch:{dataset_name}] {batch_path.name} already completed")
            continue

        image_missing = [x for x in batch_items if not x.get("image_exists_512") or not x.get("image_path_512")]
        if image_missing:
            miss_ids = [str(x.get("item_id")) for x in image_missing]
            atomic_write_json(
                batch_output_path,
                {
                    "status": "error",
                    "dataset_name": dataset_name,
                    "batch_manifest": str(batch_path),
                    "error": f"512 图像缺失: {miss_ids}",
                    "results": [],
                },
            )
            print(f"[error-batch:{dataset_name}] {batch_path.name} 缺少 512 图像: {miss_ids}")
            continue

        print(f"[run-batch:{dataset_name}] {batch_path.name} | size={len(batch_items)}")
        result = run_one_batch(
            client=client,
            types_module=types_module,
            registry=registry,
            batch_items=batch_items,
            model=model,
            retry_sleep=retry_sleep,
            max_retries=max_retries,
        )
        save_registry(REGISTRY_PATH, registry)

        if result["status"] == "ok":
            labels_map: Dict[str, str] = result["labels_map"]
            for iid, lab in labels_map.items():
                existing[iid] = lab
                done_ids.add(iid)
            atomic_write_json(
                batch_output_path,
                {
                    "status": "ok",
                    "dataset_name": dataset_name,
                    "batch_manifest": str(batch_path),
                    "generated_at": utc_now_iso(),
                    "results": [{"item_id": iid, "label": labels_map[iid]} for iid in sorted(labels_map.keys())],
                    "parsed": result["parsed"],
                    "raw_text": result.get("raw_text", ""),
                    "usage_metadata": result.get("usage_metadata"),
                },
            )
            print(f"[ok-batch:{dataset_name}] {batch_path.name} -> {len(labels_map)} items")
        else:
            atomic_write_json(
                batch_output_path,
                {
                    "status": "error",
                    "dataset_name": dataset_name,
                    "batch_manifest": str(batch_path),
                    "generated_at": utc_now_iso(),
                    "error": result.get("error", "unknown error"),
                    "raw_text": result.get("raw_text", ""),
                    "results": [],
                },
            )
            print(f"[error-batch:{dataset_name}] {batch_path.name} -> {result.get('error')}")

        write_txt_from_map(merged_txt_path, existing)
        write_json_results(merged_json_path, existing)
        processed_batches += 1
        if max_batches > 0 and processed_batches >= max_batches:
            print(f"[stop:{dataset_name}] reached max_batches={max_batches}")
            break

    # final report
    write_txt_from_map(merged_txt_path, existing)
    write_json_results(merged_json_path, existing)

    missing = sorted([iid for iid in all_ids if not existing.get(iid, "").strip()])
    report = {
        "summary": {
            "input_count": len(master.get("items", [])),
            "input_unique_ids": len(set(all_ids)),
            "done_count": len([iid for iid in set(all_ids) if existing.get(iid, "").strip()]),
            "missing_count": len(missing),
        },
        "missing_ids": missing,
    }
    atomic_write_json(report_path, report)
    print(f"[done:{dataset_name}] merged txt -> {merged_txt_path}")
    print(f"[done:{dataset_name}] merged json -> {merged_json_path}")
    print(f"[done:{dataset_name}] report -> {report_path}")


# =========================================================
# CLI / main
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Gemini 2.5 Pro Stage1 baseline batch mode")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="默认 gemini-2.5-pro")
    ap.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR), help="本地 512 图像目录")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="每批图片数，默认 10")
    ap.add_argument("--max_retries", type=int, default=2, help="单个 batch 最大重试次数")
    ap.add_argument("--retry_sleep", type=float, default=2.0, help="batch 失败后的重试起始等待秒数")
    ap.add_argument("--max_batches", type=int, default=0, help="仅调试：每个 dataset 最多处理多少个 batch；0 表示不限")
    ap.add_argument("--prepare_only", action="store_true", help="只生成 master / batch manifests，不调用模型")
    ap.add_argument("--regen_manifests", action="store_true", help="强制重建 master / batch manifests")
    ap.add_argument("--skip_base", action="store_true", help="跳过 base 数据集")
    ap.add_argument("--skip_round2", action="store_true", help="跳过 round2 数据集")
    ap.add_argument("--legacy_result_dir", default=str(LEGACY_RESULT_DIR_DEFAULT), help="已有 stage1 结果目录，用于跳过已完成样本")
    ap.add_argument("--ignore_legacy_results", action="store_true", help="不读取已有结果目录")
    return ap.parse_args()


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).expanduser()
    if not image_dir.is_absolute():
        image_dir = PROJECT_ROOT / image_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 不存在: {image_dir}")

    print("[debug] PROJECT_ROOT =", PROJECT_ROOT)
    print("[debug] INPUT_PATH =", INPUT_PATH)
    print("[debug] INPUT_PATH_ROUND2 =", INPUT_PATH_ROUND2)
    print("[debug] image_dir_512 =", image_dir)
    print("[debug] batch_size =", args.batch_size)
    print("[debug] model =", args.model)
    print("[debug] legacy_result_dir =", args.legacy_result_dir)

    legacy_result_dir = Path(args.legacy_result_dir).expanduser()
    legacy_txt_base = None if args.ignore_legacy_results else (legacy_result_dir / "results_stage1.txt")
    legacy_txt_round2 = None if args.ignore_legacy_results else (legacy_result_dir / "results_stage1_round2.txt")

    # manifests 预生成
    if args.regen_manifests or not MASTER_BASE_PATH.exists():
        atomic_write_json(MASTER_BASE_PATH, build_master_manifest(read_json_auto(INPUT_PATH), "base", image_dir))
    if args.regen_manifests or not MASTER_ROUND2_PATH.exists():
        atomic_write_json(MASTER_ROUND2_PATH, build_master_manifest(read_json_auto(INPUT_PATH_ROUND2), "round2", image_dir))
    write_batch_manifests(read_json_obj(MASTER_BASE_PATH, default={}), BATCH_BASE_DIR, args.batch_size)
    write_batch_manifests(read_json_obj(MASTER_ROUND2_PATH, default={}), BATCH_ROUND2_DIR, args.batch_size)

    if args.prepare_only:
        print("[done] manifests prepared only.")
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 GEMINI_API_KEY 或 GOOGLE_API_KEY")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    registry = load_registry(REGISTRY_PATH)

    try:
        if not args.skip_base:
            process_dataset(
                client=client,
                types_module=types,
                registry=registry,
                dataset_name="base",
                input_path=INPUT_PATH,
                master_path=MASTER_BASE_PATH,
                batch_manifest_dir=BATCH_BASE_DIR,
                batch_output_dir=BATCH_OUTPUT_BASE_DIR,
                merged_txt_path=TXT_OUT,
                merged_json_path=JSON_OUT,
                report_path=REPORT_PATH,
                image_dir=image_dir,
                batch_size=args.batch_size,
                model=args.model,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                max_batches=args.max_batches,
                regen_manifests=args.regen_manifests,
                legacy_txt_path=legacy_txt_base,
            )

        if not args.skip_round2:
            process_dataset(
                client=client,
                types_module=types,
                registry=registry,
                dataset_name="round2",
                input_path=INPUT_PATH_ROUND2,
                master_path=MASTER_ROUND2_PATH,
                batch_manifest_dir=BATCH_ROUND2_DIR,
                batch_output_dir=BATCH_OUTPUT_ROUND2_DIR,
                merged_txt_path=TXT_OUT_ROUND2,
                merged_json_path=JSON_OUT_ROUND2,
                report_path=REPORT_PATH_ROUND2,
                image_dir=image_dir,
                batch_size=args.batch_size,
                model=args.model,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                max_batches=args.max_batches,
                regen_manifests=args.regen_manifests,
                legacy_txt_path=legacy_txt_round2,
            )
    finally:
        save_registry(REGISTRY_PATH, registry)
        try:
            client.close()
        except Exception:
            pass

    print("[all done] stage1 baseline batch mode finished.")


if __name__ == "__main__":
    main()
