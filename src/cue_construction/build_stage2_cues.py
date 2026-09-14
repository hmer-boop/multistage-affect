# step2_build_stage2_cues.py
# Stage 2: Scene-level cues (entities / relations / event-or-state)

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import os
import base64
import re

import cv2
from openai import OpenAI
from tqdm import tqdm


# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_IDS_PATH = PROJECT_ROOT / "outputs" / "method_data" / "gold_item_ids.json"
IMAGES_RAW_DIR = PROJECT_ROOT / "dataset" / "images_raw"
ITEMS_MIN_PATH = PROJECT_ROOT / "dataset" / "metadata" / "items_min.jsonl"

OUT_DIR = PROJECT_ROOT / "outputs" / "method_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "stage2_cues.json"


# =========================
# VLM 配置
# =========================
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_RETRY = 2


# =========================
# 断点续跑
# =========================
def load_existing_results(out_json: Path) -> Tuple[Dict[str, Dict[str, Any]], set]:
    if not out_json.exists():
        return {}, set()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    existing_map = {}

    if isinstance(data, list):
        for rec in data:
            existing_map[str(rec["id"])] = rec
    elif isinstance(data, dict):
        for k, v in data.items():
            existing_map[str(k)] = v

    return existing_map, set(existing_map.keys())


def save_results_as_list(out_json: Path, results_map: Dict[str, Dict[str, Any]]) -> None:
    out_json.write_text(
        json.dumps(list(results_map.values()), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# =========================
# 重复检测（relations vs scene_or_event）
# =========================
import string

def _norm_zh(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    punct = set(string.punctuation) | set("，。！？；：、（）【】《》“”‘’—…· ")
    s = "".join(ch for ch in s if ch not in punct)
    return s.lower()

def dup_check(rel: str, scene: str) -> Dict[str, Any]:
    rel_n = _norm_zh(rel)
    scn_n = _norm_zh(scene)

    if not rel_n or not scn_n:
        return {"dup_flag": False, "dup_score": 0.0, "dup_reason": "empty"}

    a, b = set(rel_n), set(scn_n)
    jacc = len(a & b) / max(1, len(a | b))

    shorter, longer = (rel_n, scn_n) if len(rel_n) <= len(scn_n) else (scn_n, rel_n)
    contain = 1.0 if shorter and shorter in longer else 0.0

    score = max(jacc, contain)
    flag = (contain == 1.0 and len(shorter) >= 6) or (jacc >= 0.72)

    reason = "contain" if contain == 1.0 else ("jaccard" if jacc >= 0.72 else "ok")
    return {"dup_flag": flag, "dup_score": round(score, 3), "dup_reason": reason}


# =========================
# 图像路径解析（复用）
# =========================
def load_image_path_from_items_min(item_id: str) -> Optional[str]:
    if not ITEMS_MIN_PATH.exists():
        return None
    with ITEMS_MIN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if str(it.get("id")) == str(item_id):
                return (it.get("image_path")
                        or it.get("image")
                        or it.get("img_path")
                        or it.get("local_path")
                        or it.get("url"))
    return None


def resolve_image_path(item_id: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = IMAGES_RAW_DIR / f"{item_id}{ext}"
        if p.exists():
            return p

    s = load_image_path_from_items_min(item_id)
    if s:
        p = Path(s)
        if p.exists():
            return p
    return None


# =========================
# Prompt（严格低噪声版）
# =========================
def build_stage2_prompt() -> str:
    return """
你正在执行 Stage 2（场景线索抽取）任务。

你的目标是：基于画面本身，抽取三个【中间线索】，用于后续情感判断。
这些线索应当具体、可区分，但不包含任何情绪判断。

=====================
总体要求（必须遵守）
=====================

1）允许使用【完整句子】，但每一项最多一句话。
2）句子必须简短、具体，避免空泛描述。
3）不要使用情绪词（如：宁静、压抑、敬畏、恐惧、快乐等）。
4）不要使用风格评价或艺术评价（如：抽象、写意、象征性、超现实等）。
5）只描述画面中【可以直接观察到的内容】。

=====================
需要输出的三个字段
=====================

【entities】
- 描述画面中可指认的主要实体
- 使用名词短语，或由名词构成的简短句子
- 可以包含限定词（如：高耸的、密集的、悬浮的）
- 若不存在具体物体，可描述为结构或形态

示例：
- 高耸山体、低垂云雾、层叠远山
- 统一制服的人物与圆形徽记结构
- 放射状线条与中心色块结构

【relations】
- 用一句简短句子描述实体之间是否存在互动、对照、压迫、顺应或并置关系
- 优先描述“谁相对于谁在做什么”
- 若不存在明显互动，可描述为均衡、排列或无冲突状态

示例：
- 人物悬浮于山体之上，与地面空间形成对照
- 多个结构单元彼此并列，无明显冲突
- 人物被几何结构包围并限制活动空间

【scene_or_event】
- 若画面呈现具体事件或场景，请用一句具体句子描述“正在发生什么”
- 若不存在事件，仅呈现状态，请描述为画面所呈现的状态
- 避免风格词，避免抽象判断

示例：
- 山谷中人物凌空飞行的场景
- 夜晚水岸旁的人群聚集
- 稳定呈现的山地景观
- 静态展示的人物形象


补充约束（避免重复）：

- relations 只描述局部实体之间的作用或张力（如：谁相对于谁在上/下、包围、对照、限制等）。
- scene_or_event 只用于给整个画面命名为一个“场景”或“状态”，不要重复 relations 中的具体动作或关系。
- 如果画面是强事件型（relations 已清楚表达事件），scene_or_event 请用更概括的场景名称。

=====================
输出格式（严格 JSON）
=====================
{
  "entities": "...",
  "relations": "...",
  "scene_or_event": "..."
}

""".strip()


# =========================
# VLM 调用
# =========================
def _guess_mime(path: Path) -> str:
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if path.suffix.lower() == ".png":
        return "image/png"
    return "image/jpeg"


def _extract_json(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    l, r = t.find("{"), t.rfind("}")
    if l != -1 and r != -1:
        return t[l:r + 1]
    return t


def call_vlm(image_path: Path, prompt: str) -> Dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    data_url = f"data:{_guess_mime(image_path)};base64,{img_b64}"

    last_err = None
    for _ in range(VLM_MAX_RETRY):
        try:
            resp = client.responses.create(
                model=VLM_MODEL,
                temperature=VLM_TEMPERATURE,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }]
            )

            raw = resp.output_text or ""
            js = _extract_json(raw)
            return json.loads(js)

        except Exception as e:
            last_err = e

    print(f"⚠️ VLM 失败: {image_path.name} -> {last_err}")
    return {"entities": None, "relations": None, "scene_or_event": None}


# =========================
# 主流程
# =========================
def main():
    gold_ids = [str(x) for x in json.loads(GOLD_IDS_PATH.read_text(encoding="utf-8"))]

    existing_map, done_ids = load_existing_results(OUT_JSON)
    to_process = [gid for gid in gold_ids if gid not in done_ids]

    print(f"Stage2 total: {len(gold_ids)}, skip: {len(done_ids)}, run: {len(to_process)}")

    prompt = build_stage2_prompt()

    pbar = tqdm(to_process, desc="Stage2 scene cues", unit="img")

    new_count = 0
    SAVE_EVERY = 10  # 每 10 张落盘
    processed_count = 0  # 本次实际处理的张数（含跳过）

    for item_id in pbar:
        img_path = resolve_image_path(item_id)
        if not img_path:
            continue

        cues = call_vlm(img_path, prompt)

        # ===== 重复检测：若 relations 与 scene_or_event 高度重复，则跳过 =====
        rel = (cues or {}).get("relations")
        scn = (cues or {}).get("scene_or_event")
        dup = dup_check(rel, scn)

        processed_count += 1

        if dup["dup_flag"]:
            # 直接跳过，不写入、不计入新增
            if processed_count % SAVE_EVERY == 0:
                save_results_as_list(OUT_JSON, existing_map)
            continue

        record = {
            "id": item_id,
            "image_path": str(img_path),
            "stage": 2,
            "scene_cues": cues
        }

        existing_map[item_id] = record
        new_count += 1
        if new_count % SAVE_EVERY == 0:
            save_results_as_list(OUT_JSON, existing_map)

        if new_count % 50 == 0:
            save_results_as_list(OUT_JSON, existing_map)

    save_results_as_list(OUT_JSON, existing_map)
    print(f"Stage2 done. New: {new_count}, Total: {len(existing_map)}")
    print(f"Output -> {OUT_JSON}")
    print("\n====== Stage2 运行统计 ======")
    print(f"本次新增条数: {new_count}")
    print(f"当前总条数: {len(existing_map)}")
    print(f"本次处理图像数（含跳过）: {processed_count}")
    print(f"结果文件: {OUT_JSON}")


if __name__ == "__main__":
    main()
