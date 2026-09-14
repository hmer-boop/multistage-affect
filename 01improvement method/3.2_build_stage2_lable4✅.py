# -*- coding: utf-8 -*-
"""
Stage2: 原图(主证据) + stage2_cues(辅助证据) -> 8类情感单标签（调优后的打标）

✅ 特性：
1) RUN_LIMIT 可控：5/100/None(全量)
2) 输出 JSON（不输出 jsonl）
3) 进度显示：当前/总数、已跳过、已新增、耗时
4) 断点续跑：若 OUT_JSON 已存在，自动跳过已完成 id
5) 每 SAVE_EVERY 条落盘一次（默认每 10 条）
"""

from __future__ import annotations

import os
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# =========================
# 0) 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_IDS_PATH = PROJECT_ROOT / "01improvement method" / "method data" / "gold_item_ids.json"
IMAGES_DIR    = PROJECT_ROOT / "dataset" / "images_raw"
CUES_PATH     = PROJECT_ROOT / "01improvement method" / "method data" / "stage2_cues.json"

OUT_DIR       = PROJECT_ROOT / "01improvement method" / "method data"
OUT_JSON      = OUT_DIR / "stage2_gpt_labels4.json"

# =========================
# 0.1) 运行数量控制
# =========================
RUN_LIMIT: Optional[int] = None   # 设为 5 / 100 / None(全量)
SAVE_EVERY = 10                   # 每 10 条落盘一次

# =========================
# 1) OpenAI 配置
# =========================
MODEL_NAME = os.getenv("STAGE2_MODEL", "gpt-4o")

# =========================
# 2) 8类标签（固定枚举）
# =========================
LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)

PROMPT_VERSION = "stage2_scene_assisted_v1"

SYSTEM_PROMPT = (
    "你是一个严格的图像情绪标签器。"
    "你的任务是给图片打一个情绪单标签。"
    "你必须只输出一个标签，且标签必须来自给定的 8 类。"
)

# =========================
# 3) 工具函数：读写与找图
# =========================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def dedup_keep_order(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

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

def image_to_data_url(img_path: Path) -> str:
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(img_path.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def normalize_label(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("标签：", "").replace("标签:", "").strip()
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    t = t.strip().strip('"').strip("'").strip()
    t = t.replace("。", "").replace("，", "").replace(",", "").strip()
    return t

def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json(path)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and "id" in x]
        return []
    except Exception:
        return []

def build_done_id_set(existing: List[Dict[str, Any]]) -> set:
    done = set()
    for r in existing:
        iid = str(r.get("id"))
        if iid and iid != "None":
            done.add(iid)
    return done

# =========================
# 4) Stage2 cues 解析（兼容多种结构）
# =========================
def build_stage2_cues_map(cues_data: Any) -> Dict[str, Dict[str, Any]]:
    """
    兼容两种输入：
    A) list[{"id":..., "scene_cues":{...}}]  或 list[{"id":..., "entities":..., ...}]
    B) dict{id: {...}}
    """
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
                # 可能直接平铺了 entities/relations/scene_or_event
                cues_map[iid] = {k: v for k, v in obj.items() if k != "id"}
        return cues_map

    raise ValueError("stage2_cues.json 格式不符合预期：应为 dict 或 list[dict]")

def pick_stage2_cues(cues: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    只取 Stage2 三要素：entities / relations / scene_or_event
    若缺失则返回空 dict。
    """
    if not cues or not isinstance(cues, dict):
        return {}
    out = {}
    for k in ("entities", "relations", "scene_or_event"):
        if k in cues and cues[k] is not None:
            out[k] = cues[k]
    return out

# =========================
# 5) Prompt：先看图，再看证据（证据辅助）
# =========================
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


# =========================
# 6) OpenAI 调用（Responses API）
# =========================
def call_stage2_label(image_data_url: str, stage2_cues: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    from openai import OpenAI
    client = OpenAI()

    user_prompt = build_user_prompt(stage2_cues)

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_data_url},
                    {"type": "input_text", "text": user_prompt},
                ],
            },
        ],
    )

    out_text = getattr(resp, "output_text", "") or ""
    label = normalize_label(out_text)

    meta = {
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "raw_output": out_text,
        "response_id": getattr(resp, "id", None),
    }
    return label, meta

# =========================
# 7) 主流程
# =========================
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读 gold ids
    gold_ids_data = load_json(GOLD_IDS_PATH)
    if isinstance(gold_ids_data, dict) and "ids" in gold_ids_data:
        ids = [str(x) for x in gold_ids_data["ids"]]
    elif isinstance(gold_ids_data, list):
        ids = [str(x) for x in gold_ids_data]
    else:
        raise ValueError("gold_item_ids.json 格式不符合预期：应为 [id,...] 或 {ids:[...]}")

    ids = dedup_keep_order(ids)

    # 读 stage2 cues
    cues_data = load_json(CUES_PATH)
    cues_map = build_stage2_cues_map(cues_data)

    # 断点续跑：读已有输出
    existing_results = load_existing_results(OUT_JSON)
    done_ids = build_done_id_set(existing_results)

    total = len(ids)
    remaining = sum(1 for x in ids if x not in done_ids)

    print("=====================================")
    print("Stage2 重新打标 运行信息")
    print(f"- 模型: {MODEL_NAME}")
    print(f"- 总 id 数(去重后): {total}")
    print(f"- 已存在输出条数: {len(existing_results)}")
    print(f"- 待处理(未跑过): {remaining}")
    print(f"- RUN_LIMIT: {RUN_LIMIT} (None=全量)")
    print(f"- SAVE_EVERY: {SAVE_EVERY}")
    print("=====================================")

    added = 0
    skipped = 0
    missing_img = 0
    failed = 0

    for i, item_id in enumerate(ids, start=1):
        if item_id in done_ids:
            skipped += 1
            continue

        if RUN_LIMIT is not None and added >= RUN_LIMIT:
            break

        print(f"\n[{added+1}/{RUN_LIMIT or 'ALL'}] (scan {i}/{total}) id={item_id}")

        img_path = find_image_file(IMAGES_DIR, item_id)
        if not img_path:
            missing_img += 1
            obj = {
                "id": item_id,
                "stage2_label": None,
                "status": "image_not_found",
                "image_path": None,
                "has_stage2_cues": bool(pick_stage2_cues(cues_map.get(item_id))),
                "prompt_version": PROMPT_VERSION,
                "meta": {"model": MODEL_NAME},
            }
            existing_results.append(obj)
            done_ids.add(item_id)
            added += 1
            print("  ❌ 找不到图片，已记录 image_not_found")
        else:
            image_url = image_to_data_url(img_path)
            stage2_cues_obj = cues_map.get(item_id)

            label = None
            meta: Dict[str, Any] = {}
            status = "ok"

            # 重试：最多 3 次
            for attempt in range(1, 4):
                try:
                    label, meta = call_stage2_label(image_url, stage2_cues_obj)
                    if label not in LABEL_SET:
                        raise ValueError(f"输出不在8类中：{label!r} | raw={meta.get('raw_output')!r}")
                    break
                except Exception as e:
                    status = f"error_attempt_{attempt}"
                    meta = {
                        "error": str(e),
                        "attempt": attempt,
                        "model": MODEL_NAME,
                        "prompt_version": PROMPT_VERSION,
                    }
                    time.sleep(1.5 * attempt)

            if label in LABEL_SET:
                obj = {
                    "id": item_id,
                    "stage2_label": label,
                    "status": "ok",
                    "image_path": str(img_path),
                    "has_stage2_cues": bool(pick_stage2_cues(stage2_cues_obj)),
                    "prompt_version": PROMPT_VERSION,
                    "meta": meta,
                }
                existing_results.append(obj)
                done_ids.add(item_id)
                added += 1
                print(f"  ✅ label={label}")
            else:
                failed += 1
                obj = {
                    "id": item_id,
                    "stage2_label": None,
                    "status": "failed",
                    "image_path": str(img_path),
                    "has_stage2_cues": bool(pick_stage2_cues(stage2_cues_obj)),
                    "prompt_version": PROMPT_VERSION,
                    "meta": meta,
                }
                existing_results.append(obj)
                done_ids.add(item_id)
                added += 1
                print(f"  ❌ 失败：{meta.get('error')}")

        # 定期保存
        if added % SAVE_EVERY == 0:
            save_json(OUT_JSON, existing_results)
            elapsed = time.time() - t0
            print(f"\n💾 已保存 {len(existing_results)} 条到: {OUT_JSON}")
            print(f"   用时: {elapsed:.1f}s | 新增:{added} 跳过:{skipped} 缺图:{missing_img} 失败:{failed}")

    # 最终保存
    save_json(OUT_JSON, existing_results)

    elapsed = time.time() - t0
    print("\n=====================================")
    print("🎯 Stage2 重新打标 完成")
    print(f"- 输出 JSON: {OUT_JSON}")
    print(f"- 现有总条数: {len(existing_results)}")
    print(f"- 本次新增: {added}")
    print(f"- 本次跳过(已存在): {skipped}")
    print(f"- 本次缺图: {missing_img}")
    print(f"- 本次失败: {failed}")
    print(f"- 总用时: {elapsed:.1f}s")
    print("=====================================")

if __name__ == "__main__":
    main()
