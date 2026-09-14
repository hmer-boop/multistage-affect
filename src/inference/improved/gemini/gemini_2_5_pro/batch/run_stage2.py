# -*- coding: utf-8 -*-
"""
8.2_Gemini25pro_run_label2_improve_batch.py

Gemini 2.5 Pro Stage2 improve 批处理版（RPD 优化）。

特点：
1) 直接复用已有 stage2_cues.json，不重新跑证据抽取
2) 改为 batch 模式：每次请求处理 10 张图（默认 10）
3) 使用本地 512 图像目录作为模型输入图片来源
4) 优先复用 Gemini Files API：同一张 512 图像上传一次，后续批次复用
5) 支持断点续跑
6) 兼容已有单图版 stage2 improve 输出，并同步合并到当前 batch 输出中
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
LABEL_SET = set(LABELS)
DEFAULT_MODEL = "gemini-2.5-pro"
PROMPT_VERSION = "stage2_scene_assisted_v1_batch"
DEFAULT_BATCH_SIZE = 10
DEFAULT_IMAGE_DIR = Path("dataset_hf_upload/images_cloud_512")
FILE_TTL_HOURS = 47


def find_project_root(start_dir: Path) -> Path:
    candidates = [start_dir, *start_dir.parents]
    for p in candidates:
        if (p / "requirements.txt").exists() and (p / "src").exists():
            return p
    raise FileNotFoundError("未找到项目根目录：请确认脚本位于仓库内部。")


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(BASE_DIR)
METHOD_DIR = PROJECT_ROOT / "outputs" / "method_data"

GOLD_IDS_PATH = METHOD_DIR / "gold_item_ids.json"
CUES_PATH = METHOD_DIR / "stage2_cues.json"

SHARED_DIR = BASE_DIR.parent / "shared"
MANIFEST_MASTER_DIR = BASE_DIR / "manifests" / "master"
MANIFEST_BATCH_DIR = BASE_DIR / "manifests" / "batches"
OUT_DIR = BASE_DIR / "result Gemini_stage2"

REGISTRY_PATH = SHARED_DIR / "gemini_files_registry_512.json"
MERGED_JSON_PATH = OUT_DIR / "stage2_gemini_labels4.json"
REPORT_PATH = OUT_DIR / "stage2_gemini_labels4_missing_report.json"
BATCH_OUTPUT_DIR = OUT_DIR / "batch_outputs"
MASTER_PATH = MANIFEST_MASTER_DIR / "stage2_improve_master.json"
BATCH_DIR = MANIFEST_BATCH_DIR / "stage2_improve"

LEGACY_RESULT_DIR_DEFAULT = BASE_DIR.parent.parent / "2_improve" / "result Gemini_stage2"
LEGACY_RESULT_JSON_DEFAULT = LEGACY_RESULT_DIR_DEFAULT / "stage2_gemini_labels4.json"

for p in [SHARED_DIR, MANIFEST_MASTER_DIR, MANIFEST_BATCH_DIR, OUT_DIR, BATCH_OUTPUT_DIR, BATCH_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_obj(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_json_required(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def dedup_keep_order(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    data = read_json_obj(path, default=[])
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict) and "id" in x]
    return []


def normalize_label(text: str) -> str:
    t = (text or "").strip().replace("：", ":").replace(" ", "")
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


def sort_records_by_input_order(records: Dict[str, Dict[str, Any]], ids: List[str]) -> List[Dict[str, Any]]:
    ordered = [records[iid] for iid in ids if iid in records]
    extras = [records[iid] for iid in records.keys() if iid not in set(ids)]
    return ordered + extras


def merge_existing_records(primary_map: Dict[str, Dict[str, Any]], extra_records: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    merged = dict(primary_map)
    added = 0
    for rec in extra_records:
        iid = str(rec.get("id", "")).strip()
        if not iid:
            continue
        existing = merged.get(iid)
        existing_label = normalize_label(str((existing or {}).get("stage2_label", ""))) if existing else "未知"
        new_label = normalize_label(str(rec.get("stage2_label", "")))
        if existing is None:
            merged[iid] = rec
            added += 1
        elif existing_label not in LABEL_SET and new_label in LABEL_SET:
            merged[iid] = rec
            added += 1
    return merged, added


def build_done_id_set(records: Dict[str, Dict[str, Any]]) -> set[str]:
    done = set()
    for iid, r in records.items():
        label = normalize_label(str(r.get("stage2_label", "")))
        status = str(r.get("status", "")).strip().lower()
        if label in LABEL_SET and (status in {"", "ok"} or not status):
            done.add(iid)
    return done


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


def build_stage2_cues_map(cues_data: Any) -> Dict[str, Dict[str, Any]]:
    cues_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(cues_data, dict):
        for k, v in cues_data.items():
            if isinstance(v, dict):
                cues_map[str(k)] = v
            else:
                cues_map[str(k)] = {"scene_cues": v}
        return cues_map
    if isinstance(cues_data, list):
        for obj in cues_data:
            if not isinstance(obj, dict) or "id" not in obj:
                continue
            iid = str(obj["id"])
            if "scene_cues" in obj and isinstance(obj["scene_cues"], dict):
                cues_map[iid] = obj["scene_cues"]
            else:
                cues_map[iid] = {k: v for k, v in obj.items() if k != "id"}
        return cues_map
    raise ValueError("stage2_cues.json 格式不符合预期：应为 dict 或 list[dict]")


def pick_stage2_cues(cues: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cues or not isinstance(cues, dict):
        return {}
    out = {}
    for k in ("entities", "relations", "scene_or_event"):
        if k in cues and cues[k] is not None:
            out[k] = cues[k]
    return out


def build_master_manifest(ids: List[str], cues_map: Dict[str, Dict[str, Any]], image_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for item_id in ids:
        local_512 = guess_local_512_image(image_dir, item_id)
        cues3 = pick_stage2_cues(cues_map.get(item_id))
        rows.append(
            {
                "item_id": item_id,
                "image_path_512": str(local_512) if local_512 else None,
                "image_exists_512": bool(local_512 and local_512.exists()),
                "has_stage2_cues": bool(cues3),
                "stage2_cues": cues3,
            }
        )
    return {
        "task": "stage2_improve",
        "generated_at": utc_now_iso(),
        "image_dir_512": str(image_dir),
        "gold_ids_path": str(GOLD_IDS_PATH),
        "cues_path": str(CUES_PATH),
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
    for i in range(total_batches):
        chunk = items[i * batch_size:(i + 1) * batch_size]
        batch_obj = {
            "task": task,
            "batch_index": i + 1,
            "batch_size": len(chunk),
            "generated_at": utc_now_iso(),
            "items": chunk,
        }
        p = batch_dir / f"{task}_batch_{i + 1:04d}.json"
        atomic_write_json(p, batch_obj)
        paths.append(p)
    return paths


STAGE2_IMPROVE_BATCH_RESPONSE_JSON_SCHEMA = {
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


def build_stage2_improve_batch_intro(batch_items: List[Dict[str, Any]]) -> str:
    ids_text = "\n".join([f"- {x['item_id']}" for x in batch_items])
    return (
        "你将给一批图片打“情绪单标签”。\n"
        "【顺序要求】先看图片本身，形成主要判断；再阅读该图片对应的 Stage2 证据做辅助校正。\n"
        "注意：证据只是辅助，不应覆盖你对图片的直接判断。\n\n"
        "【约束】\n"
        "- 不输出解释、不输出原因、不输出多标签\n"
        "- 不做文化象征、叙事推理、艺术风格评论\n"
        "- 若证据与图片直觉冲突，以图片为准\n\n"
        "【第一轮判定（正常判定）】\n"
        "请你基于图片本身（Stage2 证据仅作辅助），先做一次直觉性的情绪判断，\n"
        "从以下 8 个标签中选出一个最符合图片整体情绪的候选标签：\n"
        "悲伤、恐惧、厌恶、愤怒、宁静、快乐、惊奇、敬畏\n\n"
        "【第二轮判定（仅当候选为“宁静”或“敬畏”时才执行）】\n"
        "如果你在第一轮中选择了“宁静”或“敬畏”，请进行一次二次确认：\n\n"
        "—— 宁静 二次确认条件（至少满足1条强证据） ——\n"
        "1) 画面整体为低激活状态（无明显强动作、强对抗或强刺激）\n"
        "2) 画面中不存在威胁、紧迫、怪异、压迫或冲突事件\n"
        "3) 画面更像稳定呈现的状态或景观，而非正在发生的事件\n"
        "若缺少明确强证据，请重新选择更贴近的标签（悲伤/恐惧/厌恶/愤怒/快乐/惊奇）。\n\n"
        "—— 敬畏 二次确认条件（至少满足一条强证据） ——\n"
        "1) 明确的宏大尺度与渺小对比（人物或物体显著渺小）\n"
        "2) 宗教、祭祀或庄严肃穆的仪式性场景\n"
        "3) 自然伟力或极端险峻的壮观场景（如高山、风暴、火山、宇宙等）\n"
        "若缺少明确强证据，请重新选择更贴近的标签（悲伤/恐惧/厌恶/愤怒/快乐/惊奇）。\n\n"
        "【最终任务】\n"
        "在完成上述判定后，请输出你最终确认的那个情绪标签。\n"
        "只输出 JSON，对每个 item_id 恰好给出一个标签。\n"
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
        if label not in LABEL_SET:
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
    intro = build_stage2_improve_batch_intro(batch_items)
    contents: List[Any] = [types_module.Part.from_text(text=intro)]

    for item in batch_items:
        item_id = str(item["item_id"])
        local_path = Path(item["image_path_512"])
        reg = ensure_file_registered(client, local_path, registry, display_name=item_id)
        contents.append(types_module.Part.from_text(text=f"item_id: {item_id}\n请先看紧随其后的图片，形成主要判断。"))
        contents.append(types_module.Part.from_uri(file_uri=reg["uri"], mime_type=reg["mime_type"]))
        cues_text = json.dumps(item.get("stage2_cues") or {}, ensure_ascii=False) if item.get("stage2_cues") else "(无)"
        contents.append(types_module.Part.from_text(text=(
            f"item_id: {item_id}\n"
            "下面是该图片对应的 Stage2 证据（辅助，三要素）：\n"
            f"{cues_text}\n"
            "这些证据只做辅助校正；若与图片直觉冲突，以图片为准。"
        )))

    last_error: Optional[str] = None
    last_raw_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": STAGE2_IMPROVE_BATCH_RESPONSE_JSON_SCHEMA,
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


def process_all(
    *,
    client,
    types_module,
    registry: Dict[str, Any],
    ids: List[str],
    cues_map: Dict[str, Dict[str, Any]],
    image_dir: Path,
    batch_size: int,
    model: str,
    max_retries: int,
    retry_sleep: float,
    max_batches: int,
    regen_manifests: bool,
    legacy_result_json: Optional[Path],
):
    print("\n===== stage2 improve | batch =====")

    if regen_manifests or not MASTER_PATH.exists():
        master = build_master_manifest(ids, cues_map=cues_map, image_dir=image_dir)
        atomic_write_json(MASTER_PATH, master)
        print(f"[manifest] master generated -> {MASTER_PATH}")
    else:
        master = read_json_obj(MASTER_PATH, default={})
        print(f"[manifest] master reused -> {MASTER_PATH}")

    batch_paths = write_batch_manifests(master, BATCH_DIR, batch_size=batch_size)
    print(f"[manifest] batches generated = {len(batch_paths)} | batch_size = {batch_size}")

    existing_list = load_existing_results(MERGED_JSON_PATH)
    result_map: Dict[str, Dict[str, Any]] = {str(x["id"]): x for x in existing_list if "id" in x}
    current_done_count = len(build_done_id_set(result_map))

    if legacy_result_json and legacy_result_json.exists():
        legacy_list = load_existing_results(legacy_result_json)
        result_map, legacy_added = merge_existing_records(result_map, legacy_list)
        atomic_write_json(MERGED_JSON_PATH, sort_records_by_input_order(result_map, ids))
        print(f"[resume] legacy json loaded = {len(legacy_list)} | merged_add = {legacy_added} | synced_to_current_outputs = True | {legacy_result_json}")
    else:
        if legacy_result_json:
            print(f"[resume] legacy json not found -> {legacy_result_json}")

    done_ids = build_done_id_set(result_map)
    print(f"[resume] done_ids in batch merged json = {current_done_count} | total done after legacy merge = {len(done_ids)}")

    processed_batches = 0
    for batch_path in batch_paths:
        batch = read_json_obj(batch_path, default={})
        batch_items_all = batch.get("items", []) if isinstance(batch, dict) else []
        batch_items = [x for x in batch_items_all if str(x.get("item_id")) not in done_ids]
        batch_output_path = BATCH_OUTPUT_DIR / batch_path.name

        if batch_output_path.exists() and batch_items:
            prev = read_json_obj(batch_output_path, default={})
            if isinstance(prev, dict) and prev.get("status") == "ok":
                prev_results = prev.get("results", [])
                prev_map = {str(r.get("item_id")): str(r.get("label")) for r in prev_results if isinstance(r, dict)}
                needed_ids = {str(x.get("item_id")) for x in batch_items}
                if needed_ids and needed_ids.issubset(prev_map.keys()):
                    for item in batch_items:
                        iid = str(item.get("item_id"))
                        label = normalize_label(prev_map.get(iid, ""))
                        if label in LABEL_SET:
                            result_map[iid] = {
                                "id": iid,
                                "stage2_label": label,
                                "status": "ok",
                                "image_path": str(item.get("image_path_512") or ""),
                                "has_stage2_cues": bool(item.get("has_stage2_cues")),
                                "prompt_version": PROMPT_VERSION,
                                "meta": {
                                    "model": model,
                                    "prompt_version": PROMPT_VERSION,
                                    "raw_output": prev.get("raw_text", ""),
                                    "reused_from_batch_output": True,
                                },
                            }
                            done_ids.add(iid)
                    print(f"[skip-batch] reuse existing batch output -> {batch_output_path.name}")
                    continue

        if not batch_items:
            print(f"[skip-batch] {batch_path.name} already completed")
            continue

        image_missing = [x for x in batch_items if not x.get("image_exists_512") or not x.get("image_path_512")]
        if image_missing:
            miss_ids = [str(x.get("item_id")) for x in image_missing]
            atomic_write_json(
                batch_output_path,
                {
                    "status": "error",
                    "batch_manifest": str(batch_path),
                    "error": f"512 图像缺失: {miss_ids}",
                    "results": [],
                },
            )
            print(f"[error-batch] {batch_path.name} 缺少 512 图像: {miss_ids}")
            continue

        print(f"[run-batch] {batch_path.name} | size={len(batch_items)}")
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
            for item in batch_items:
                iid = str(item["item_id"])
                label = labels_map.get(iid)
                if label not in LABEL_SET:
                    continue
                result_map[iid] = {
                    "id": iid,
                    "stage2_label": label,
                    "status": "ok",
                    "image_path": str(item.get("image_path_512") or ""),
                    "has_stage2_cues": bool(item.get("has_stage2_cues")),
                    "prompt_version": PROMPT_VERSION,
                    "meta": {
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                        "raw_output": result.get("raw_text", ""),
                        "stage2_cues": item.get("stage2_cues") or {},
                    },
                }
                done_ids.add(iid)
            atomic_write_json(
                batch_output_path,
                {
                    "status": "ok",
                    "batch_manifest": str(batch_path),
                    "generated_at": utc_now_iso(),
                    "results": [{"item_id": iid, "label": labels_map[iid]} for iid in sorted(labels_map.keys())],
                    "parsed": result["parsed"],
                    "raw_text": result.get("raw_text", ""),
                    "usage_metadata": result.get("usage_metadata"),
                },
            )
            print(f"[ok-batch] {batch_path.name} -> {len(labels_map)} items")
        else:
            atomic_write_json(
                batch_output_path,
                {
                    "status": "error",
                    "batch_manifest": str(batch_path),
                    "generated_at": utc_now_iso(),
                    "error": result.get("error", "unknown error"),
                    "raw_text": result.get("raw_text", ""),
                    "results": [],
                },
            )
            print(f"[error-batch] {batch_path.name} -> {result.get('error')}")

        atomic_write_json(MERGED_JSON_PATH, sort_records_by_input_order(result_map, ids))
        processed_batches += 1
        if max_batches > 0 and processed_batches >= max_batches:
            print(f"[stop] reached max_batches={max_batches}")
            break

    atomic_write_json(MERGED_JSON_PATH, sort_records_by_input_order(result_map, ids))

    done_ids_final = build_done_id_set(result_map)
    missing_ids = [iid for iid in ids if iid not in done_ids_final]
    report = {
        "summary": {
            "input_count": len(ids),
            "done_count": len(done_ids_final),
            "missing_count": len(missing_ids),
        },
        "missing_ids": missing_ids,
    }
    atomic_write_json(REPORT_PATH, report)
    print(f"[done] merged json -> {MERGED_JSON_PATH}")
    print(f"[done] report -> {REPORT_PATH}")


def parse_args():
    ap = argparse.ArgumentParser(description="Gemini 2.5 Pro Stage2 improve batch mode")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR), help="本地 512 图像目录")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="每批图片数，默认 10")
    ap.add_argument("--limit", type=int, default=-1, help="仅调试：只取前多少个 gold ids；-1 表示全量")
    ap.add_argument("--max_retries", type=int, default=1, help="单个 batch 最大重试次数")
    ap.add_argument("--retry_sleep", type=float, default=1.5, help="batch 失败后的重试起始等待秒数")
    ap.add_argument("--max_batches", type=int, default=0, help="仅调试：最多处理多少个 batch；0 表示不限")
    ap.add_argument("--prepare_only", action="store_true", help="只生成 master / batch manifests，不调用模型")
    ap.add_argument("--regen_manifests", action="store_true", help="强制重建 master / batch manifests")
    ap.add_argument("--legacy_result_json", default=str(LEGACY_RESULT_JSON_DEFAULT), help="已有 stage2 improved 结果 json，用于跳过已完成样本")
    ap.add_argument("--ignore_legacy_results", action="store_true", help="不读取已有结果 json")
    return ap.parse_args()


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).expanduser()
    if not image_dir.is_absolute():
        image_dir = PROJECT_ROOT / image_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 不存在: {image_dir}")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 GEMINI_API_KEY 或 GOOGLE_API_KEY")

    gold_ids_data = load_json_required(GOLD_IDS_PATH)
    if isinstance(gold_ids_data, dict) and "ids" in gold_ids_data:
        ids = [str(x) for x in gold_ids_data["ids"]]
    elif isinstance(gold_ids_data, list):
        ids = [str(x) for x in gold_ids_data]
    else:
        raise ValueError("gold_item_ids.json 格式不符合预期：应为 [id,...] 或 {ids:[...]}")

    ids = dedup_keep_order(ids)
    if args.limit and args.limit > 0:
        ids = ids[:args.limit]

    cues_data = load_json_required(CUES_PATH)
    cues_map = build_stage2_cues_map(cues_data)

    print("[debug] PROJECT_ROOT =", PROJECT_ROOT)
    print("[debug] GOLD_IDS_PATH =", GOLD_IDS_PATH)
    print("[debug] CUES_PATH =", CUES_PATH)
    print("[debug] image_dir_512 =", image_dir)
    print("[debug] batch_size =", args.batch_size)
    print("[debug] model =", args.model)
    print("[debug] legacy_result_json =", args.legacy_result_json)
    print("[debug] input gold ids =", len(ids))

    if args.regen_manifests or not MASTER_PATH.exists():
        atomic_write_json(MASTER_PATH, build_master_manifest(ids, cues_map=cues_map, image_dir=image_dir))
    write_batch_manifests(read_json_obj(MASTER_PATH, default={}), BATCH_DIR, args.batch_size)

    if args.prepare_only:
        print("[done] manifests prepared only.")
        return

    legacy_result_json = None if args.ignore_legacy_results else Path(args.legacy_result_json).expanduser()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    registry = load_registry(REGISTRY_PATH)

    try:
        process_all(
            client=client,
            types_module=types,
            registry=registry,
            ids=ids,
            cues_map=cues_map,
            image_dir=image_dir,
            batch_size=args.batch_size,
            model=args.model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            max_batches=args.max_batches,
            regen_manifests=args.regen_manifests,
            legacy_result_json=legacy_result_json,
        )
    finally:
        save_registry(REGISTRY_PATH, registry)
        try:
            client.close()
        except Exception:
            pass

    print("[all done] stage2 improve batch mode finished.")


if __name__ == "__main__":
    main()
