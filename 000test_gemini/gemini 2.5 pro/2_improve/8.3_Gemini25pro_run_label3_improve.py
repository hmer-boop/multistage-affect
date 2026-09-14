# -*- coding: utf-8 -*-
"""
8.3_Gemini25pro_run_label3_improve.py

Stage3 improve：复用 stage3_all_information.json，直接做图文融合最终标签（Gemini 2.5 Pro）

特点：
1) 不重新跑 text-only 或其他证据抽取，直接读取 stage3_all_information.json
2) 输出 JSON 到当前脚本目录下独立文件夹：result Gemini_stage3/
3) 支持断点续跑、并发、定期落盘、进度显示
4) 默认参数与前面 Gemini baseline 加速版保持一致：
   - model = gemini-2.5-pro
   - max_side = 512
   - quality = 80
   - max_workers = 4
   - max_retries = 1
   - flush_every = 50
"""

from __future__ import annotations

import os
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

DEFAULT_MODEL = "gemini-2.5-pro"
LABELS_8 = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]
LABEL_SET_8 = set(LABELS_8)
TEXT_TRIGGER_SET = {"none", "event", "appraisal", "explicit_emotion"}
RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


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

OUT_DIR = BASE_DIR / "result Gemini_stage3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "stage3_gemini_labels3.json"
OUT_TMP = OUT_JSON.with_suffix(".json.tmp")


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        s = f.read().strip()
    if not s:
        return None
    return json.loads(s)


def atomic_save(obj: Any, tmp_path: Path, final_path: Path):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def normalize_id(x: Any) -> str:
    return str(x).strip()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_items(path: Path) -> List[Dict[str, Any]]:
    obj = load_json(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
        return [x for x in obj["items"] if isinstance(x, dict)]
    raise ValueError(f"输入结构不支持：{path}")


def load_existing_out(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    obj = load_json(path)
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


def resolve_image_path(rec: Dict[str, Any]) -> str:
    image_path = safe_str(rec.get("image_path"))
    if image_path and os.path.exists(image_path):
        return image_path

    path_raw = safe_str(rec.get("path_raw"))
    if not path_raw:
        return ""

    if os.path.exists(path_raw):
        return path_raw

    candidate = os.path.normpath(os.path.join(str(PROJECT_ROOT), path_raw))
    if os.path.exists(candidate):
        return candidate

    return path_raw


def resolve_title(rec: Dict[str, Any]) -> str:
    title_zh = safe_str(rec.get("title_zh"))
    title_en = safe_str(rec.get("title_en"))
    title = safe_str(rec.get("title"))
    if title:
        return title
    if title_zh and title_en:
        return f"{title_zh} / {title_en}"
    return title_zh or title_en


def parse_model_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def extract_response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    try:
        candidates = getattr(response, "candidates", None) or []
        parts = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(str(part_text))
        return "\n".join(parts).strip()
    except Exception:
        return ""


def image_to_bytes(image_path: str, max_side: int = 512, quality: int = 80) -> Tuple[bytes, str]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    min_side = min(w, h)
    if min_side > max_side:
        scale = max_side / float(min_side)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), RESAMPLE)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"


def validate_prompt_b_inputs(image_path: str, stage2_label: str, text_label: str, text_trigger: str, title: str) -> None:
    missing = []
    if not image_path or not os.path.exists(image_path):
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
{{"label":"<{labels_str}之一>","confidence":0-1}}"""


def build_output_record(rec: Dict[str, Any], model: str) -> Dict[str, Any]:
    _id = normalize_id(rec.get("id", ""))
    title_zh = safe_str(rec.get("title_zh"))
    title_en = safe_str(rec.get("title_en"))
    title = safe_str(rec.get("title")) or title_zh or title_en
    path_raw = safe_str(rec.get("path_raw"))
    image_path = resolve_image_path(rec)
    stage2_label = safe_str(rec.get("stage2_label"))
    ts_old = safe_str(rec.get("ts"))

    return {
        "id": _id,
        "title_zh": title_zh,
        "title_en": title_en,
        "title": title,
        "path_raw": path_raw,
        "image_path": image_path,
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
    if not safe_str(rec.get("image_path")) or not os.path.exists(safe_str(rec.get("image_path"))):
        return False
    return True


def sort_out_list(id2rec: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(s: str):
        try:
            return (0, int(s))
        except Exception:
            return (1, s)
    return [id2rec[k] for k in sorted(id2rec.keys(), key=sort_key)]


def call_fusion(client, types, model: str, image_path: str, prompt: str, max_side: int, quality: int) -> Dict[str, Any]:
    image_bytes, mime_type = image_to_bytes(image_path=image_path, max_side=max_side, quality=quality)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    text = extract_response_text(response)
    out = parse_model_json(text)
    if out.get("label") not in LABEL_SET_8:
        raise ValueError(f"label不在8类集合中：{out.get('label')}")
    conf = out.get("confidence")
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        raise ValueError(f"confidence非法：{conf}")
    return {"label": out["label"], "confidence": float(conf), "raw_output": text}


def worker_one(rec: Dict[str, Any], args, api_key: str) -> Tuple[str, Dict[str, Any]]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("请先安装 google-genai") from e

    client = genai.Client(api_key=api_key)
    model = args.model
    _id = normalize_id(rec.get("id", ""))
    base = build_output_record(rec, model=model)

    title = safe_str(base.get("title"))
    image_path = safe_str(base.get("image_path"))
    stage2_label = safe_str(base.get("stage2_label"))
    text_label = safe_str(base.get("text_label"))
    text_trigger = safe_str(base.get("text_trigger"))
    text_conf = float(base.get("text_confidence", 0.0) or 0.0)

    validate_prompt_b_inputs(
        image_path=image_path,
        stage2_label=stage2_label,
        text_label=text_label,
        text_trigger=text_trigger,
        title=title,
    )

    prompt_b = build_prompt_fusion(
        title=title,
        stage2_label=stage2_label,
        text_label=text_label,
        text_trigger=text_trigger,
        text_confidence=text_conf,
    )

    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            out = call_fusion(
                client=client,
                types=types,
                model=model,
                image_path=image_path,
                prompt=prompt_b,
                max_side=args.max_side,
                quality=args.quality,
            )
            base["stage3_label"] = out["label"]
            base["stage3_confidence"] = float(out["confidence"])
            base["error"] = ""
            base["model"] = model
            base["meta"] = {"raw_output": out.get("raw_output", "")}
            return _id, base
        except Exception as e:
            last_error = str(e)
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep)

    base["stage3_label"] = ""
    base["stage3_confidence"] = 0.0
    base["error"] = last_error
    base["model"] = model
    return _id, base


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Gemini 2.5 Pro Stage3 improve labeling")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="0 表示全量")
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--retry_sleep", type=float, default=1.5)
    ap.add_argument("--flush_every", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 错误：未检测到 GEMINI_API_KEY 或 GOOGLE_API_KEY")
        return

    items = load_items(IN_PATH)
    existing = load_existing_out(OUT_JSON)

    out_map: Dict[str, Dict[str, Any]] = {}
    input_map = {normalize_id(x.get("id", "")): x for x in items if normalize_id(x.get("id", ""))}

    for _id, rec in input_map.items():
        merged = dict(rec)
        if _id in existing:
            old = existing[_id]
            merged["stage3_label"] = old.get("stage3_label", old.get("final_label", ""))
            merged["stage3_confidence"] = old.get("stage3_confidence", old.get("final_confidence", 0.0))
            merged["error"] = old.get("error", "")
        out_map[_id] = build_output_record(merged, model=args.model)

    done_ids: Set[str] = {_id for _id, rec in out_map.items() if is_complete_success_record(rec)}
    candidates = [input_map[_id] for _id in input_map if _id not in done_ids]

    if args.limit and args.limit > 0:
        candidates = candidates[:args.limit]

    print("=====================================")
    print("Gemini Stage3 improve 运行信息")
    print(f"- PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"- 模型: {args.model}")
    print(f"- 输入总条目数: {len(items)}")
    print(f"- 已有完整成功记录(跳过): {len(done_ids)}")
    print(f"- 待新增/补跑数量: {len(candidates)}")
    print(f"- OUT_JSON: {OUT_JSON}")
    print("=====================================")

    new_ok = 0
    new_fail = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(worker_one, rec=rec, args=args, api_key=api_key): normalize_id(rec.get("id", ""))
            for rec in candidates
        }

        for idx, fut in enumerate(as_completed(futures), start=1):
            _id = futures[fut]
            try:
                rid, record = fut.result()
                out_map[rid] = record
                if safe_str(record.get("stage3_label")) in LABEL_SET_8 and not safe_str(record.get("error")):
                    new_ok += 1
                    print(f"[stage3] {idx}/{len(candidates)} | {rid} -> {record.get('stage3_label')}")
                else:
                    new_fail += 1
                    print(f"[stage3] {idx}/{len(candidates)} | {rid} -> failed")
            except Exception as e:
                new_fail += 1
                base = build_output_record(input_map[_id], model=args.model)
                base["stage3_label"] = ""
                base["stage3_confidence"] = 0.0
                base["error"] = str(e)
                out_map[_id] = base
                print(f"[error] {idx}/{len(candidates)} | {_id} -> {e}")

            if (new_ok + new_fail) % max(1, args.flush_every) == 0:
                atomic_save(sort_out_list(out_map), OUT_TMP, OUT_JSON)
                print(f"[flush] 已落盘 {new_ok + new_fail} 条 -> {OUT_JSON}")

    atomic_save(sort_out_list(out_map), OUT_TMP, OUT_JSON)

    elapsed = time.time() - start
    print("=====================================")
    print("🎯 Gemini Stage3 improve 完成")
    print(f"- 输出 JSON: {OUT_JSON}")
    print(f"- 当前总条目数: {len(out_map)}")
    print(f"- 本次新增成功: {new_ok}")
    print(f"- 本次新增失败: {new_fail}")
    print(f"- 总用时: {elapsed:.1f}s")
    print("=====================================")


if __name__ == "__main__":
    main()
