# -*- coding: utf-8 -*-
"""
8.2_Gemini25pro_run_label2_improve.py

Stage2 improve：原图(主证据) + stage2_cues(辅助证据) -> 8类情感单标签（Gemini 2.5 Pro）

特点：
1) 直接复用已有 stage2_cues.json，不重新跑证据抽取
2) 输出 JSON 到当前脚本目录下独立文件夹：result Gemini_stage2/
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
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)
DEFAULT_MODEL = "gemini-2.5-pro"
PROMPT_VERSION = "stage2_scene_assisted_v1"
RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def find_project_root(start_dir: Path) -> Path:
    candidates = [start_dir, *start_dir.parents]
    for p in candidates:
        if (p / "requirements.txt").exists() and (p / "src").exists():
            return p
    raise FileNotFoundError("未找到项目根目录：请确认脚本位于仓库内部。")


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(BASE_DIR)
METHOD_DIR = PROJECT_ROOT / "outputs" / "method_data"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images_raw"

GOLD_IDS_PATH = METHOD_DIR / "gold_item_ids.json"
CUES_PATH = METHOD_DIR / "stage2_cues.json"

OUT_DIR = BASE_DIR / "result Gemini_stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "stage2_gemini_labels4.json"
OUT_TMP = OUT_JSON.with_suffix(".json.tmp")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save(obj: Any, tmp_path: Path, final_path: Path):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def dedup_keep_order(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json(path)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and "id" in x]
    except Exception:
        pass
    return []


def build_done_id_set(existing: List[Dict[str, Any]]) -> set:
    done = set()
    for r in existing:
        iid = str(r.get("id", "")).strip()
        if iid:
            done.add(iid)
    return done


def sort_records_by_input_order(records: Dict[str, Dict[str, Any]], ids: List[str]) -> List[Dict[str, Any]]:
    ordered = [records[iid] for iid in ids if iid in records]
    extras = [records[iid] for iid in records.keys() if iid not in set(ids)]
    return ordered + extras


def find_image_file(images_dir: Path, item_id: str) -> Optional[Path]:
    exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    for ext in exts:
        p = images_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    candidates = list(images_dir.glob(f"{item_id}.*"))
    if not candidates:
        return None
    for ext in exts:
        for c in candidates:
            if c.suffix.lower() == ext:
                return c
    return candidates[0]


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


SYSTEM_PROMPT = (
    "你是一个严格的图像情绪标签器。"
    "你的任务是给图片打一个情绪单标签。"
    "你必须只输出一个标签，且标签必须来自给定的 8 类。"
)


def build_user_prompt(stage2_cues: Optional[Dict[str, Any]]) -> str:
    cues3 = pick_stage2_cues(stage2_cues)
    cues_text = "(无)"
    if cues3:
        cues_text = json.dumps(cues3, ensure_ascii=False)

    return (
        "你将给图片打一个“情绪单标签”。\n"
        "【顺序要求】先看图片本身，形成主要判断；再阅读下方 Stage2 证据做辅助校正。\n"
        "注意：证据只是辅助，不应覆盖你对图片的直接判断。\n\n"

        "【Stage2 证据（辅助，三要素）】\n"
        f"{cues_text}\n\n"

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
        "只输出标签本身，不要输出任何解释、标点、编号或其他文字。"
    )


def image_to_bytes(img_path: Path, max_side: int = 512, quality: int = 80) -> Tuple[bytes, str]:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    min_side = min(w, h)
    if min_side > max_side:
        scale = max_side / float(min_side)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), RESAMPLE)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"


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


def call_stage2_label(client, types, model: str, image_path: Path, stage2_cues: Optional[Dict[str, Any]],
                      max_side: int, quality: int) -> Tuple[str, Dict[str, Any]]:
    image_bytes, mime_type = image_to_bytes(image_path, max_side=max_side, quality=quality)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            build_user_prompt(stage2_cues),
        ],
    )
    raw_text = extract_response_text(response)
    label = normalize_label(raw_text)
    meta = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "raw_output": raw_text,
    }
    return label, meta


def worker_one(item_id: str, stage2_cues_obj: Optional[Dict[str, Any]], args, api_key: str) -> Dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("请先安装 google-genai") from e

    client = genai.Client(api_key=api_key)

    img_path = find_image_file(IMAGES_DIR, item_id)
    if not img_path:
        return {
            "id": item_id,
            "stage2_label": None,
            "status": "image_not_found",
            "image_path": None,
            "has_stage2_cues": bool(pick_stage2_cues(stage2_cues_obj)),
            "prompt_version": PROMPT_VERSION,
            "meta": {"model": args.model},
        }

    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            label, meta = call_stage2_label(
                client=client,
                types=types,
                model=args.model,
                image_path=img_path,
                stage2_cues=stage2_cues_obj,
                max_side=args.max_side,
                quality=args.quality,
            )
            if label not in LABEL_SET:
                raise ValueError(f"输出不在8类中：{label!r} | raw={meta.get('raw_output')!r}")
            return {
                "id": item_id,
                "stage2_label": label,
                "status": "ok",
                "image_path": str(img_path),
                "has_stage2_cues": bool(pick_stage2_cues(stage2_cues_obj)),
                "prompt_version": PROMPT_VERSION,
                "meta": meta,
            }
        except Exception as e:
            last_error = str(e)
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep)

    return {
        "id": item_id,
        "stage2_label": None,
        "status": "failed",
        "image_path": str(img_path),
        "has_stage2_cues": bool(pick_stage2_cues(stage2_cues_obj)),
        "prompt_version": PROMPT_VERSION,
        "meta": {
            "error": last_error,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
        },
    }


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Gemini 2.5 Pro Stage2 improve labeling")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=-1)
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

    gold_ids_data = load_json(GOLD_IDS_PATH)
    if isinstance(gold_ids_data, dict) and "ids" in gold_ids_data:
        ids = [str(x) for x in gold_ids_data["ids"]]
    elif isinstance(gold_ids_data, list):
        ids = [str(x) for x in gold_ids_data]
    else:
        raise ValueError("gold_item_ids.json 格式不符合预期：应为 [id,...] 或 {ids:[...]}")

    ids = dedup_keep_order(ids)

    cues_data = load_json(CUES_PATH)
    cues_map = build_stage2_cues_map(cues_data)

    existing_list = load_existing_results(OUT_JSON)
    result_map: Dict[str, Dict[str, Any]] = {str(x["id"]): x for x in existing_list if "id" in x}
    done_ids = build_done_id_set(existing_list)

    todo_ids = [iid for iid in ids if iid not in done_ids]
    if args.limit and args.limit > 0:
        todo_ids = todo_ids[:args.limit]

    print("=====================================")
    print("Gemini Stage2 improve 运行信息")
    print(f"- PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"- 模型: {args.model}")
    print(f"- 总 id 数(去重后): {len(ids)}")
    print(f"- 已存在输出条数: {len(existing_list)}")
    print(f"- 待处理(未跑过): {len(todo_ids)}")
    print(f"- OUT_JSON: {OUT_JSON}")
    print("=====================================")

    added = 0
    ok_count = 0
    fail_count = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(worker_one, item_id=iid, stage2_cues_obj=cues_map.get(iid), args=args, api_key=api_key): iid
            for iid in todo_ids
        }

        for idx, fut in enumerate(as_completed(futures), start=1):
            iid = futures[fut]
            try:
                record = fut.result()
                result_map[iid] = record
                added += 1
                if record.get("status") == "ok":
                    ok_count += 1
                    print(f"[stage2] {idx}/{len(todo_ids)} | {iid} -> {record.get('stage2_label')}")
                else:
                    fail_count += 1
                    print(f"[stage2] {idx}/{len(todo_ids)} | {iid} -> {record.get('status')}")
            except Exception as e:
                fail_count += 1
                result_map[iid] = {
                    "id": iid,
                    "stage2_label": None,
                    "status": "failed",
                    "image_path": None,
                    "has_stage2_cues": bool(pick_stage2_cues(cues_map.get(iid))),
                    "prompt_version": PROMPT_VERSION,
                    "meta": {"error": str(e), "model": args.model},
                }
                print(f"[error] {idx}/{len(todo_ids)} | {iid} -> {e}")

            if added % max(1, args.flush_every) == 0:
                atomic_save(sort_records_by_input_order(result_map, ids), OUT_TMP, OUT_JSON)
                print(f"[flush] 已落盘 {added} 条 -> {OUT_JSON}")

    atomic_save(sort_records_by_input_order(result_map, ids), OUT_TMP, OUT_JSON)

    elapsed = time.time() - start
    print("=====================================")
    print("🎯 Gemini Stage2 improve 完成")
    print(f"- 输出 JSON: {OUT_JSON}")
    print(f"- 当前总条数: {len(result_map)}")
    print(f"- 本次新增: {added}")
    print(f"- 本次成功: {ok_count}")
    print(f"- 本次失败: {fail_count}")
    print(f"- 总用时: {elapsed:.1f}s")
    print("=====================================")


if __name__ == "__main__":
    main()
