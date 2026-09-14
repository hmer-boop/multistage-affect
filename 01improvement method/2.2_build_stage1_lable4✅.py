# -*- coding: utf-8 -*-
"""
Stage1: 原图(主证据) + stage1_cues(辅助证据) -> 8类情感单标签（短时观看模拟）

✅ 满足你的要求：
1) 数量选择“做活”：RUN_LIMIT 可设为 5 / 100 / None(全量)
2) 只输出 JSON（不输出 jsonl）
3) 有进度显示（当前/总数、已跳过、已新增、耗时）
4) 检查已有数据避免重复跑：若 OUT_JSON 已存在，会先读取并跳过已完成 id
5) 途中安全：每处理 SAVE_EVERY 条就落盘一次（避免中途断了白跑）

依赖：
- pip install openai

环境变量：
- OPENAI_API_KEY 必须已设置
- 可选：STAGE1_MODEL 例如 "gpt-4.1-mini" / "gpt-4.1" / "gpt-4o-mini"
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
CUES_PATH     = PROJECT_ROOT / "01improvement method" / "method data" / "stage1_cues.json"

OUT_DIR       = PROJECT_ROOT / "01improvement method" / "method data"
OUT_JSON      = OUT_DIR / "stage1_gpt_labels4.json"

# =========================
# 0.1) 运行数量控制（做活）
# =========================
# - 设为 5：只跑 5 条“新的（未跑过的）”
# - 设为 200：只跑 200 条“新的（未跑过的）”
# - 设为 None：全量跑完“所有新的（未跑过的）”
RUN_LIMIT: Optional[int] = None

# 每处理多少条就保存一次（防断电/中断）
SAVE_EVERY = 10

# =========================
# 1) OpenAI 配置
# =========================
MODEL_NAME = os.getenv("STAGE1_MODEL", "gpt-4o")

# =========================
# 2) 8类标签（固定枚举）
# =========================
LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)

PROMPT_VERSION = "stage1_glance_v1"

SYSTEM_PROMPT = (
    "你是一个严格的图像情绪标签器。"
    "你的任务是模拟人类在 1–2 秒内扫视图片的直觉情绪判断（System-1）。"
    "不要进行深入分析、不要解释原因、不要做文化象征或叙事解读。"
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
    # 兜底：id.*（优先常见图片后缀）
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
    # 常见多余符号清一下
    t = t.replace("。", "").replace("，", "").replace(",", "").strip()
    return t

def pick_stage1_5_cues(cues_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从 stage1_cues.json 的单条记录里抽取固定 5 条 global_cues：
    brightness_pattern / color_tone / visual_complexity / spatial_pressure / overall_dynamics
    """
    if not cues_item or not isinstance(cues_item, dict):
        return {}

    gc = cues_item.get("global_cues")
    if not isinstance(gc, dict):
        return {}

    return {
        "brightness_pattern": gc.get("brightness_pattern"),
        "color_tone": gc.get("color_tone"),
        "visual_complexity": gc.get("visual_complexity"),
        "spatial_pressure": gc.get("spatial_pressure"),
        "overall_dynamics": gc.get("overall_dynamics"),
    }


def build_user_prompt(cues_item: Optional[Dict[str, Any]]) -> str:
    top5 = pick_stage1_5_cues(cues_item)

    cues_text = "(无)"
    if top5:
        cues_text = json.dumps(top5, ensure_ascii=False)

    return (
        "你将模拟人类在 1–2 秒内扫视图片时的直觉情绪判断（Stage1）。\n"
        "【顺序要求】先看图片形成第一眼直觉；再阅读下面 5 条补充信息与软触发器做轻微校正。\n"
        "注意：补充信息/触发器都是“概率倾向”，不是硬规则；若与图片直觉冲突，以图片直觉为准。\n\n"

        "【软触发器（概率倾向）】\n"
        "1) 纹理/肌理很多、密集、粗粝、颗粒感强，整体给人不适/脏乱/黏腻感 → 更偏向：厌恶\n"
        "2) 大面积深色、压暗氛围、强烈阴影，或单色/一片深蓝（冷、暗、压迫） → 更偏向：恐惧\n"
        "3) 颜色整体明亮，高饱和度，色彩碰撞 → 更偏向：惊奇或快乐\n"
        "4) 颜色整体灰暗 → 更偏向：悲伤\n"
        "4) 大面积高饱和红、强烈红色冲突或刺激 → 更偏向：愤怒\n\n"

        "【补充信息（5条，全局感知）】\n"
        f"{cues_text}\n\n"

        "【防止默认选项】\n"
        "- 不要因为不确定就默认选择“宁静”或“敬畏”。\n"
        "- 只有当画面确实平和、舒缓、无明显冲突或刺激时，才选择：宁静。\n"
        "- 只有当画面带来宏大、庄严、神圣、崇高或压迫式震撼时，才选择：敬畏。\n\n"

        "【观看条件】\n"
        "- 不深入分析\n"
        "- 不解释原因\n"
        "- 不做象征/叙事/文化解读\n\n"

        "【任务】\n"
        "请从以下 8 个标签中选择一个最符合“第一眼直觉”的情绪：\n"
        "悲伤、恐惧、厌恶、愤怒、宁静、快乐、惊奇、敬畏\n"
        "只输出标签本身，不要输出任何解释、标点或编号。"
    )


def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json(path)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and "id" in x]
        return []
    except Exception:
        # 如果 json 损坏，宁可不读；你也可以手动修复
        return []

def build_done_id_set(existing: List[Dict[str, Any]]) -> set:
    done = set()
    for r in existing:
        iid = str(r.get("id"))
        # 只要曾经写入（成功/失败）都先视为 done，避免重复跑
        if iid and iid != "None":
            done.add(iid)
    return done

# =========================
# 4) OpenAI 调用（Responses API）
# =========================
def call_stage1_label(image_data_url: str, cues_obj: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    from openai import OpenAI
    client = OpenAI()

    user_prompt = build_user_prompt(cues_obj)

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT}
                ]
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
# 5) 主流程
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

    # 输入去重（保持顺序）
    ids = dedup_keep_order(ids)

    # 读 cues
    cues_data = load_json(CUES_PATH)
    cues_map: Dict[str, Dict[str, Any]] = {}

    # 允许：{id: cues} 或 [{"id":..., ...}, ...]
    if isinstance(cues_data, dict):
        for k, v in cues_data.items():
            cues_map[str(k)] = v if isinstance(v, dict) else {"cues": v}
    elif isinstance(cues_data, list):
        for obj in cues_data:
            if isinstance(obj, dict) and "id" in obj:
                iid = str(obj["id"])
                cues_map[iid] = {kk: vv for kk, vv in obj.items() if kk != "id"}
    else:
        raise ValueError("stage1_cues.json 格式不符合预期：应为 dict 或 list[dict]")

    # 读取已有输出（用于查重）
    existing_results = load_existing_results(OUT_JSON)
    done_ids = build_done_id_set(existing_results)

    total = len(ids)
    remaining = sum(1 for x in ids if x not in done_ids)

    print("=====================================")
    print("Stage1 运行信息")
    print(f"- 模型: {MODEL_NAME}")
    print(f"- 总 id 数(去重后): {total}")
    print(f"- 已存在输出条数: {len(existing_results)}")
    print(f"- 待处理(未跑过): {remaining}")
    print(f"- RUN_LIMIT: {RUN_LIMIT} (None=全量)")
    print("=====================================")

    # 新增结果会 append 到 existing_results
    added = 0
    skipped = 0
    missing_img = 0
    failed = 0

    # 主循环：只处理“新”的；并受 RUN_LIMIT 控制
    for i, item_id in enumerate(ids, start=1):
        if item_id in done_ids:
            skipped += 1
            continue

        if RUN_LIMIT is not None and added >= RUN_LIMIT:
            break

        # 进度（这里的 i 是遍历位置，不等于新增数）
        print(f"\n[{added+1}/{RUN_LIMIT or 'ALL'}] (scan {i}/{total}) id={item_id}")

        img_path = find_image_file(IMAGES_DIR, item_id)
        if not img_path:
            missing_img += 1
            obj = {
                "id": item_id,
                "stage1_label": None,
                "status": "image_not_found",
                "image_path": None,
                "has_cues": bool(cues_map.get(item_id)),
                "prompt_version": PROMPT_VERSION,
                "meta": {"model": MODEL_NAME},
            }
            existing_results.append(obj)
            done_ids.add(item_id)
            added += 1
            print("  ❌ 找不到图片，已记录 image_not_found")
        else:
            image_url = image_to_data_url(img_path)
            cues_obj = cues_map.get(item_id)

            label = None
            meta: Dict[str, Any] = {}
            status = "ok"

            # 简单重试：最多 3 次
            for attempt in range(1, 4):
                try:
                    label, meta = call_stage1_label(image_url, cues_obj)
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
                    "stage1_label": label,
                    "status": "ok",
                    "image_path": str(img_path),
                    "has_cues": bool(cues_obj),
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
                    "stage1_label": None,
                    "status": "failed",
                    "image_path": str(img_path),
                    "has_cues": bool(cues_obj),
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
    print("🎯 Stage1 完成")
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
