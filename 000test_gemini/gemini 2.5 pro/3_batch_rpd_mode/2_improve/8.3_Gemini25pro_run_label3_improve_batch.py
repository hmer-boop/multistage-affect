# -*- coding: utf-8 -*-
"""
8.3_Gemini25pro_run_label3_improve_batch.py

Gemini 2.5 Pro Stage3 improve 批处理版（RPD 优化）。

特点：
1) 直接复用 stage3_all_information.json
2) 改为 batch 模式：每次请求处理 10 张图（默认 10）
3) 使用本地 512 图像目录作为模型输入图片来源
4) 优先复用 Gemini Files API：同一张 512 图像上传一次，后续批次复用
5) 支持断点续跑
6) 兼容旧单图版 stage3 improve 输出，并同步合并到当前 batch 输出中
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
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_BATCH_SIZE = 10
DEFAULT_IMAGE_DIR = Path("dataset_hf_upload/images_cloud_512")
FILE_TTL_HOURS = 47

LABELS_8 = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]
LABEL_SET_8 = set(LABELS_8)
TEXT_TRIGGER_SET = {"none", "event", "appraisal", "explicit_emotion"}


def find_project_root(start_dir: Path) -> Path:
    candidates = [start_dir, *start_dir.parents]
    for p in candidates:
        if (p / "dataset").exists() and (p / "01improvement method").exists():
            return p
    raise FileNotFoundError("未找到项目根目录：请确认脚本位于 PythonProject 内部。")


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(BASE_DIR)
METHOD_DIR = PROJECT_ROOT / "01improvement method" / "method data"

IN_PATH = METHOD_DIR / "stage3_all_information.json"

SHARED_DIR = BASE_DIR.parent / "shared"
MANIFEST_MASTER_DIR = BASE_DIR / "manifests" / "master"
MANIFEST_BATCH_DIR = BASE_DIR / "manifests" / "batches"
OUT_DIR = BASE_DIR / "result Gemini_stage3"

REGISTRY_PATH = SHARED_DIR / "gemini_files_registry_512.json"
MERGED_JSON_PATH = OUT_DIR / "stage3_gemini_labels3.json"
REPORT_PATH = OUT_DIR / "stage3_gemini_labels3_missing_report.json"
BATCH_OUTPUT_DIR = OUT_DIR / "batch_outputs"
MASTER_PATH = MANIFEST_MASTER_DIR / "stage3_improve_master.json"
BATCH_DIR = MANIFEST_BATCH_DIR / "stage3_improve"

LEGACY_RESULT_DIR_DEFAULT = BASE_DIR.parent.parent / "2_improve" / "result Gemini_stage3"
LEGACY_RESULT_JSON_DEFAULT = LEGACY_RESULT_DIR_DEFAULT / "stage3_gemini_labels3.json"

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
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        s = f.read().strip()
    if not s:
        raise ValueError(f"文件为空：{path}")
    return json.loads(s)


def atomic_write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def normalize_id(x: Any) -> str:
    return str(x).strip()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_items(path: Path) -> List[Dict[str, Any]]:
    obj = load_json_required(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
        return [x for x in obj["items"] if isinstance(x, dict)]
    raise ValueError(f"输入结构不支持：{path}")


def load_existing_out(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    obj = read_json_obj(path, default=None)
    if obj is None:
        return {}
    mp: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict) and "id" in it:
                mp[normalize_id(it["id"])] = it
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                vv = dict(v)
                if "id" not in vv:
                    vv["id"] = normalize_id(k)
                mp[normalize_id(vv["id"])] = vv
    return mp


def sort_out_list(id2rec: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(s: str):
        try:
            return (0, int(s))
        except Exception:
            return (1, s)
    return [id2rec[k] for k in sorted(id2rec.keys(), key=sort_key)]


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
    for lab in LABELS_8:
        if lab in t:
            return lab
    return "未知"


def merge_existing_records(primary_map: Dict[str, Dict[str, Any]], extra_records: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    merged = dict(primary_map)
    added = 0
    for rec in extra_records:
        iid = normalize_id(rec.get("id", ""))
        if not iid:
            continue
        existing = merged.get(iid)
        existing_label = normalize_label(safe_str((existing or {}).get("stage3_label", ""))) if existing else "未知"
        new_label = normalize_label(safe_str(rec.get("stage3_label", rec.get("final_label", ""))))
        if existing is None:
            merged[iid] = rec
            added += 1
        elif existing_label not in LABEL_SET_8 and new_label in LABEL_SET_8:
            merged[iid] = rec
            added += 1
    return merged, added


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


def resolve_title(rec: Dict[str, Any]) -> str:
    title_zh = safe_str(rec.get("title_zh"))
    title_en = safe_str(rec.get("title_en"))
    title = safe_str(rec.get("title"))
    if title:
        return title
    if title_zh and title_en:
        return f"{title_zh} / {title_en}"
    return title_zh or title_en


def build_output_record(rec: Dict[str, Any], model: str, image_path_512: str = "") -> Dict[str, Any]:
    _id = normalize_id(rec.get("id", ""))
    title_zh = safe_str(rec.get("title_zh"))
    title_en = safe_str(rec.get("title_en"))
    title = safe_str(rec.get("title")) or title_zh or title_en
    path_raw = safe_str(rec.get("path_raw"))
    stage2_label = safe_str(rec.get("stage2_label"))
    ts_old = safe_str(rec.get("ts"))

    return {
        "id": _id,
        "title_zh": title_zh,
        "title_en": title_en,
        "title": title,
        "path_raw": path_raw,
        "image_path": image_path_512 or safe_str(rec.get("image_path")),
        "stage2_label": stage2_label,
        "ts": ts_old,
        "text_label": safe_str(rec.get("text_label")),
        "text_trigger": safe_str(rec.get("text_trigger")),
        "text_confidence": float(rec.get("text_confidence", 0.0) or 0.0),
        "text_error": safe_str(rec.get("text_error")),
        "model": model,
        "stage3_label": safe_str(rec.get("stage3_label") or rec.get("final_label")),
        "stage3_confidence": float(rec.get("stage3_confidence", rec.get("final_confidence", 0.0)) or 0.0),
        "error": safe_str(rec.get("error")),
    }


def is_complete_success_record(rec: Dict[str, Any]) -> bool:
    if safe_str(rec.get("stage3_label")) not in LABEL_SET_8:
        return False
    if safe_str(rec.get("error")):
        return False
    if not safe_str(rec.get("id")):
        return False
    if not safe_str(rec.get("title_zh")) and not safe_str(rec.get("title_en")) and not safe_str(rec.get("title")):
        return False
    if safe_str(rec.get("stage2_label")) not in LABEL_SET_8:
        return False
    if safe_str(rec.get("text_label")) not in LABEL_SET_8:
        return False
    if safe_str(rec.get("text_trigger")) not in TEXT_TRIGGER_SET:
        return False
    if not safe_str(rec.get("image_path")):
        return False
    return True


def build_master_manifest(items: List[Dict[str, Any]], image_dir: Path, model: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for rec in items:
        iid = normalize_id(rec.get("id", ""))
        if not iid:
            continue
        local_512 = guess_local_512_image(image_dir, iid)
        rows.append({
            "id": iid,
            "title": resolve_title(rec),
            "title_zh": safe_str(rec.get("title_zh")),
            "title_en": safe_str(rec.get("title_en")),
            "path_raw": safe_str(rec.get("path_raw")),
            "image_path_512": str(local_512) if local_512 else "",
            "image_exists_512": bool(local_512 and local_512.exists()),
            "stage2_label": safe_str(rec.get("stage2_label")),
            "ts": safe_str(rec.get("ts")),
            "text_label": safe_str(rec.get("text_label")),
            "text_trigger": safe_str(rec.get("text_trigger")),
            "text_confidence": float(rec.get("text_confidence", 0.0) or 0.0),
            "text_error": safe_str(rec.get("text_error")),
            "model": model,
        })
    return {
        "task": "stage3_improve",
        "generated_at": utc_now_iso(),
        "image_dir_512": str(image_dir),
        "input_path": str(IN_PATH),
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


def validate_prompt_b_inputs(image_path: str, stage2_label: str, text_label: str, text_trigger: str, title: str) -> None:
    missing = []
    if not image_path:
        missing.append(f"image_path无效: {image_path}")
    if not title:
        missing.append("title缺失/为空")
    if stage2_label not in LABEL_SET_8:
        missing.append(f"stage2_label非法/缺失: {stage2_label}")
    if text_label not in LABEL_SET_8:
        missing.append(f"text_label非法/缺失: {text_label}")
    if text_trigger not in TEXT_TRIGGER_SET:
        missing.append(f"text_trigger非法/缺失: {text_trigger}")
    if missing:
        raise ValueError("promptB_input_incomplete: " + "; ".join(missing))


def build_prompt_fusion(title: str, stage2_label: str, text_label: str, text_trigger: str, text_confidence: float) -> str:
    labels_str = "、".join(LABELS_8)
    return f"""你将为该作品输出最终“情绪单标签”（8选1）。

你有四个信息源：
1) stage2_label：仅看图的直觉标签（可能错）
2) text_label：仅看标题的文本建议（可能误导）
3) text_trigger：文本强度类型
4) text_confidence：文本建议置信度（0-1）
你将看到清晰图像用于最终裁决。

【已知锚点（仅看图）】
stage2_label = {stage2_label}

【文本建议（仅看标题）】
text_label = {text_label}
text_trigger = {text_trigger}
text_confidence = {text_confidence}

【标题】
{title}

【融合策略（权重）】
- stage2_label 仅为先验（prior）
- 当 text_trigger 较强且 text_confidence 较高时，允许文本主导

【规则（冲突如何选择）】
- text_trigger=none：强制 stay（final_label=stage2_label）
- text_trigger=event：
  - 若画面支持该事件/行为语义，优先 text_label（尽量不跨极性）
- text_trigger=appraisal：
  - 若 text_confidence>=0.75，默认 final_label=text_label；只有“强反证”才推翻
- text_trigger=explicit_emotion：
  - 默认 final_label=text_label；只有“强反证”才推翻

【强反证（才允许推翻文本）】
- 文本指向 NEG，但画面呈现明显欢乐/庆祝/温馨互动且无危险压迫线索
- 文本指向 POS，但画面呈现明确暴力/死亡/恐惧追逐/哭泣崩溃/强压迫威胁线索

【最终校正规则（仅用于易混标签复核）】
输出 final_label 前，请再次检查以下规则：
- “宁静”仅作为兜底标签，不可仅因画面安静、人物少、色调克制、构图平稳而直接判为“宁静”。
- 若画面整体虽安静，但具有宏大、庄严、神圣、深邃、崇高，或令人产生渺小感、肃穆感、被震住的感受，应优先判为“敬畏”。
- 若画面虽不喧闹，但整体明朗、温暖、轻快、舒展、富有生机、愉悦或积极活力，应优先判为“快乐”。
- “恐惧”侧重危险、威胁、压迫、不安和受害风险；“厌恶”侧重脏污、腐败、黏腻、病态、反胃、嫌恶和排斥。
- 仅在其他更具体情绪证据不足时，才最终选择“宁静”。

【输出格式（严格）】
只输出一个JSON（不要Markdown，不要多余文字）：
{{"results":[{{"item_id":"<id>","label":"<{labels_str}之一>","confidence":0-1}}]}}"""


STAGE3_IMPROVE_BATCH_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "label": {"type": "string", "enum": LABELS_8},
                    "confidence": {"type": "number"},
                },
                "required": ["item_id", "label", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_stage3_improve_batch_intro(batch_items: List[Dict[str, Any]]) -> str:
    ids_text = "\n".join([f"- {x['id']}" for x in batch_items])
    return (
        "你将为一批作品输出最终情绪单标签（8选1）。\n"
        "每个 item 的顺序都是：item_id 文本 -> 图片 -> 该 item 的文字锚点与融合规则信息。\n"
        "你必须为每个 item_id 返回一条 JSON 结果。\n"
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


def validate_batch_results(batch_items: List[Dict[str, Any]], parsed_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    expected_ids = [str(x["id"]) for x in batch_items]
    results = parsed_obj.get("results") if isinstance(parsed_obj, dict) else None
    if not isinstance(results, list):
        raise ValueError("模型返回中缺少 results 数组")

    out: Dict[str, Dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError(f"results 中存在非对象项: {row!r}")
        item_id = str(row.get("item_id", "")).strip()
        label = normalize_label(str(row.get("label", "")))
        conf = row.get("confidence", None)
        if not item_id:
            raise ValueError(f"存在缺少 item_id 的结果项: {row!r}")
        if label not in LABEL_SET_8:
            raise ValueError(f"item_id={item_id} 返回非法标签: {label!r}")
        if not isinstance(conf, (int, float)) or not (0 <= float(conf) <= 1):
            raise ValueError(f"item_id={item_id} confidence非法: {conf!r}")
        if item_id in out:
            raise ValueError(f"item_id={item_id} 重复返回")
        out[item_id] = {"label": label, "confidence": float(conf)}

    missing = [iid for iid in expected_ids if iid not in out]
    extra = [iid for iid in out.keys() if iid not in expected_ids]
    if missing or extra:
        raise ValueError(f"返回 item_id 与请求不一致 | missing={missing} | extra={extra}")
    return out


def run_one_batch(client, types_module, registry: Dict[str, Any], batch_items: List[Dict[str, Any]], model: str, retry_sleep: float, max_retries: int) -> Dict[str, Any]:
    intro = build_stage3_improve_batch_intro(batch_items)
    contents: List[Any] = [types_module.Part.from_text(text=intro)]

    for item in batch_items:
        iid = str(item["id"])
        local_path = Path(item["image_path_512"])
        reg = ensure_file_registered(client, local_path, registry, display_name=iid)

        validate_prompt_b_inputs(
            image_path=str(local_path),
            stage2_label=safe_str(item.get("stage2_label")),
            text_label=safe_str(item.get("text_label")),
            text_trigger=safe_str(item.get("text_trigger")),
            title=safe_str(item.get("title")),
        )
        fusion_prompt = build_prompt_fusion(
            title=safe_str(item.get("title")),
            stage2_label=safe_str(item.get("stage2_label")),
            text_label=safe_str(item.get("text_label")),
            text_trigger=safe_str(item.get("text_trigger")),
            text_confidence=float(item.get("text_confidence", 0.0) or 0.0),
        )

        contents.append(types_module.Part.from_text(text=f"item_id: {iid}\n请先看紧随其后的图片。"))
        contents.append(types_module.Part.from_uri(file_uri=reg["uri"], mime_type=reg["mime_type"]))
        contents.append(types_module.Part.from_text(text=f"item_id: {iid}\n{fusion_prompt}"))

    last_error: Optional[str] = None
    last_raw_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": STAGE3_IMPROVE_BATCH_RESPONSE_JSON_SCHEMA,
                    "temperature": 0,
                },
            )
            parsed_obj, raw_text = parse_batch_response(response)
            last_raw_text = raw_text
            if parsed_obj is None:
                raise ValueError("模型未返回可解析 JSON")
            results_map = validate_batch_results(batch_items, parsed_obj)
            usage = getattr(response, "usage_metadata", None)
            usage_dict = None
            if usage is not None:
                try:
                    usage_dict = json.loads(json.dumps(usage, default=lambda o: getattr(o, "__dict__", str(o))))
                except Exception:
                    usage_dict = str(usage)
            return {
                "status": "ok",
                "results_map": results_map,
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
    items: List[Dict[str, Any]],
    image_dir: Path,
    batch_size: int,
    model: str,
    max_retries: int,
    retry_sleep: float,
    max_batches: int,
    regen_manifests: bool,
    legacy_result_json: Optional[Path],
):
    print("\n===== stage3 improve | batch =====")

    if regen_manifests or not MASTER_PATH.exists():
        master = build_master_manifest(items, image_dir=image_dir, model=model)
        atomic_write_json(MASTER_PATH, master)
        print(f"[manifest] master generated -> {MASTER_PATH}")
    else:
        master = read_json_obj(MASTER_PATH, default={})
        print(f"[manifest] master reused -> {MASTER_PATH}")

    batch_paths = write_batch_manifests(master, BATCH_DIR, batch_size=batch_size)
    print(f"[manifest] batches generated = {len(batch_paths)} | batch_size = {batch_size}")

    existing_map = load_existing_out(MERGED_JSON_PATH)
    out_map: Dict[str, Dict[str, Any]] = dict(existing_map)
    current_done_count = len({_id for _id, rec in out_map.items() if is_complete_success_record(rec)})

    if legacy_result_json and legacy_result_json.exists():
        legacy_map = load_existing_out(legacy_result_json)
        legacy_list = list(legacy_map.values())
        out_map, legacy_added = merge_existing_records(out_map, legacy_list)
        atomic_write_json(MERGED_JSON_PATH, sort_out_list(out_map))
        print(f"[resume] legacy json loaded = {len(legacy_list)} | merged_add = {legacy_added} | synced_to_current_outputs = True | {legacy_result_json}")
    else:
        if legacy_result_json:
            print(f"[resume] legacy json not found -> {legacy_result_json}")

    done_ids: Set[str] = {_id for _id, rec in out_map.items() if is_complete_success_record(rec)}
    input_ids = [normalize_id(x.get("id", "")) for x in master.get("items", []) if normalize_id(x.get("id", ""))]
    print(f"[resume] done_ids in batch merged json = {current_done_count} | total done after legacy merge = {len(done_ids)}")

    processed_batches = 0
    for batch_path in batch_paths:
        batch = read_json_obj(batch_path, default={})
        batch_items_all = batch.get("items", []) if isinstance(batch, dict) else []
        batch_items = [x for x in batch_items_all if str(x.get("id")) not in done_ids]
        batch_output_path = BATCH_OUTPUT_DIR / batch_path.name

        if batch_output_path.exists() and batch_items:
            prev = read_json_obj(batch_output_path, default={})
            if isinstance(prev, dict) and prev.get("status") == "ok":
                prev_results = prev.get("results", [])
                prev_map = {str(r.get("item_id")): r for r in prev_results if isinstance(r, dict)}
                needed_ids = {str(x.get("id")) for x in batch_items}
                if needed_ids and needed_ids.issubset(prev_map.keys()):
                    for item in batch_items:
                        iid = str(item.get("id"))
                        row = prev_map.get(iid, {})
                        label = normalize_label(str(row.get("label", "")))
                        conf = float(row.get("confidence", 0.0) or 0.0)
                        if label in LABEL_SET_8:
                            rec = build_output_record(item, model=model, image_path_512=str(item.get("image_path_512") or ""))
                            rec["stage3_label"] = label
                            rec["stage3_confidence"] = conf
                            rec["error"] = ""
                            rec["meta"] = {"raw_output": prev.get("raw_text", ""), "reused_from_batch_output": True}
                            out_map[iid] = rec
                            done_ids.add(iid)
                    print(f"[skip-batch] reuse existing batch output -> {batch_output_path.name}")
                    continue

        if not batch_items:
            print(f"[skip-batch] {batch_path.name} already completed")
            continue

        image_missing = [x for x in batch_items if not x.get("image_exists_512") or not x.get("image_path_512")]
        if image_missing:
            miss_ids = [str(x.get("id")) for x in image_missing]
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
            results_map: Dict[str, Dict[str, Any]] = result["results_map"]
            for item in batch_items:
                iid = str(item["id"])
                row = results_map.get(iid, {})
                label = normalize_label(str(row.get("label", "")))
                conf = float(row.get("confidence", 0.0) or 0.0)
                if label not in LABEL_SET_8:
                    continue
                rec = build_output_record(item, model=model, image_path_512=str(item.get("image_path_512") or ""))
                rec["stage3_label"] = label
                rec["stage3_confidence"] = conf
                rec["error"] = ""
                rec["meta"] = {"raw_output": result.get("raw_text", "")}
                out_map[iid] = rec
                done_ids.add(iid)

            atomic_write_json(
                batch_output_path,
                {
                    "status": "ok",
                    "batch_manifest": str(batch_path),
                    "generated_at": utc_now_iso(),
                    "results": [
                        {"item_id": iid, "label": results_map[iid]["label"], "confidence": results_map[iid]["confidence"]}
                        for iid in sorted(results_map.keys())
                    ],
                    "parsed": result["parsed"],
                    "raw_text": result.get("raw_text", ""),
                    "usage_metadata": result.get("usage_metadata"),
                },
            )
            print(f"[ok-batch] {batch_path.name} -> {len(results_map)} items")
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

        atomic_write_json(MERGED_JSON_PATH, sort_out_list(out_map))
        processed_batches += 1
        if max_batches > 0 and processed_batches >= max_batches:
            print(f"[stop] reached max_batches={max_batches}")
            break

    atomic_write_json(MERGED_JSON_PATH, sort_out_list(out_map))

    done_ids_final = {_id for _id, rec in out_map.items() if is_complete_success_record(rec)}
    missing_ids = [iid for iid in input_ids if iid not in done_ids_final]
    report = {
        "summary": {
            "input_count": len(input_ids),
            "done_count": len(done_ids_final),
            "missing_count": len(missing_ids),
        },
        "missing_ids": missing_ids,
    }
    atomic_write_json(REPORT_PATH, report)
    print(f"[done] merged json -> {MERGED_JSON_PATH}")
    print(f"[done] report -> {REPORT_PATH}")


def parse_args():
    ap = argparse.ArgumentParser(description="Gemini 2.5 Pro Stage3 improve batch mode")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR), help="本地 512 图像目录")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="每批图片数，默认 10")
    ap.add_argument("--limit", type=int, default=0, help="0 表示全量")
    ap.add_argument("--max_retries", type=int, default=1, help="单个 batch 最大重试次数")
    ap.add_argument("--retry_sleep", type=float, default=1.5, help="batch 失败后的重试起始等待秒数")
    ap.add_argument("--max_batches", type=int, default=0, help="仅调试：最多处理多少个 batch；0 表示不限")
    ap.add_argument("--prepare_only", action="store_true", help="只生成 master / batch manifests，不调用模型")
    ap.add_argument("--regen_manifests", action="store_true", help="强制重建 master / batch manifests")
    ap.add_argument("--legacy_result_json", default=str(LEGACY_RESULT_JSON_DEFAULT), help="旧单图 stage3 improve 结果 json，用于跳过已完成样本")
    ap.add_argument("--ignore_legacy_results", action="store_true", help="不读取旧单图结果 json")
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

    items = load_items(IN_PATH)
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    print("[debug] PROJECT_ROOT =", PROJECT_ROOT)
    print("[debug] IN_PATH =", IN_PATH)
    print("[debug] image_dir_512 =", image_dir)
    print("[debug] batch_size =", args.batch_size)
    print("[debug] model =", args.model)
    print("[debug] legacy_result_json =", args.legacy_result_json)
    print("[debug] input items =", len(items))

    if args.regen_manifests or not MASTER_PATH.exists():
        atomic_write_json(MASTER_PATH, build_master_manifest(items, image_dir=image_dir, model=args.model))
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
            items=items,
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

    print("[all done] stage3 improve batch mode finished.")


if __name__ == "__main__":
    main()
