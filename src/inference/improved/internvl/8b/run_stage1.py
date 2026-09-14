# -*- coding: utf-8 -*-
"""
Stage1 Improve (InternVL 8B):
云端图 URL + stage1_cues(辅助证据) -> 8类情感单标签（短时观看模拟）

保留 improve 版的这些特性：
1) RUN_LIMIT 可设为 5 / 100 / None(全量)
2) 只输出 JSON（不输出 jsonl / txt）
3) 有进度显示（当前/总数、已跳过、已新增、耗时）
4) 检查已有数据避免重复跑：若 OUT_JSON 已存在，会先读取并跳过已完成 id
5) 途中安全：每处理 SAVE_EVERY 条就落盘一次
6) 失败重试，避免接口偶发波动白跑

注意：
- 这里不再读本地图片转 base64
- 直接调用 Hugging Face dataset 上的公开图片 URL
- 默认假设云端图片命名为 images_cloud/{item_id}.jpg
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from huggingface_hub import InferenceClient
    _HF_OK = True
except Exception:
    _HF_OK = False


# =========================
# 0) 路径配置（改成你当前 Mac 工程）
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists() and (p / "src").exists())

GOLD_IDS_PATH = PROJECT_ROOT / "outputs" / "method_data" / "gold_item_ids.json"
CUES_PATH     = PROJECT_ROOT / "outputs" / "method_data" / "stage1_cues.json"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists() and (p / "src").exists())

GOLD_IDS_PATH = PROJECT_ROOT / "outputs" / "method_data" / "gold_item_ids.json"
CUES_PATH     = PROJECT_ROOT / "outputs" / "method_data" / "stage1_cues.json"

OUT_DIR  = SCRIPT_DIR / "result internvl_stage1"
OUT_JSON = OUT_DIR / "stage1_internvl8b_labels4.json"

# =========================
# 0.1) 运行数量控制（做活）
# =========================
# - 设为 5：只跑 5 条“新的（未跑过的）”
# - 设为 200：只跑 200 条“新的（未跑过的）”
# - 设为 None：全量跑完“所有新的（未跑过的）”
RUN_LIMIT: Optional[int] = None

# 每处理多少条就保存一次（防断电/中断）
SAVE_EVERY = 10

# 失败重试次数
MAX_RETRIES = 3

# 接口参数
REQUEST_TIMEOUT = 120.0
RATE_DELAY = 0.0

# =========================
# 1) HF / InternVL 配置
# =========================
HF_DATASET_REPO = "hmer123/affectecom"
HF_DATASET_REV = "main"

# 这里不再用 OPENAI_API_KEY
# 仍然使用你当前已经跑通的：
# HF_ENDPOINT_URL / HF_TOKEN / HF_MODEL_ID
MODEL_NAME = os.getenv("HF_MODEL_ID", "OpenGVLab/InternVL3_5-8B")

# =========================
# 2) 8类标签（固定枚举）
# =========================
LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)

PROMPT_VERSION = "stage1_glance_v1_internvl"

SYSTEM_PROMPT = (
    "你是一个严格的图像情绪标签器。"
    "你的任务是模拟人类在 1–2 秒内扫视图片的直觉情绪判断（System-1）。"
    "不要进行深入分析、不要解释原因、不要做文化象征或叙事解读。"
    "你必须只输出一个标签，且标签必须来自给定的 8 类。"
)

# =========================
# 3) 工具函数：读写与输入处理
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
        # 如果 json 损坏，宁可不读；你也可以手动修复
        return []

def build_done_id_set(existing: List[Dict[str, Any]]) -> set:
    done = set()
    for r in existing:
        iid = str(r.get("id"))
        if iid and iid != "None":
            done.add(iid)
    return done

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
        "5) 大面积高饱和红、强烈红色冲突或刺激 → 更偏向：愤怒\n\n"

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

def item_id_to_hf_url(item_id: str) -> str:
    # 你当前 label2 成功日志里，云端图片就是这种命名方式：images_cloud/{item_id}.jpg
    return f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/{HF_DATASET_REV}/images_cloud/{item_id}.jpg"


# =========================
# 4) InternVL 调用
# =========================
def build_client() -> "InferenceClient":
    endpoint_url = os.getenv("HF_ENDPOINT_URL")
    hf_token = os.getenv("HF_TOKEN")
    model_id = os.getenv("HF_MODEL_ID")

    if not endpoint_url:
        raise RuntimeError("缺少环境变量 HF_ENDPOINT_URL")
    if not hf_token:
        raise RuntimeError("缺少环境变量 HF_TOKEN")
    if not model_id:
        raise RuntimeError("缺少环境变量 HF_MODEL_ID")

    client = InferenceClient(
        base_url=endpoint_url,
        token=hf_token,
        timeout=REQUEST_TIMEOUT,
    )
    return client

def call_stage1_label(client: "InferenceClient", image_url: str, cues_obj: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    user_prompt = build_user_prompt(cues_obj)
    model_id = os.getenv("HF_MODEL_ID")

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=32,
        temperature=0.0,
    )

    out_text = (resp.choices[0].message.content or "").strip()
    label = normalize_label(out_text)

    meta = {
        "model": model_id,
        "prompt_version": PROMPT_VERSION,
        "raw_output": out_text,
    }
    return label, meta


# =========================
# 5) 主流程
# =========================
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _HF_OK:
        raise RuntimeError("未安装 huggingface_hub 或导入失败，请先安装 huggingface_hub。")

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

    client = build_client()

    print("=====================================")
    print("Stage1 Improve / InternVL 运行信息")
    print(f"- 模型: {os.getenv('HF_MODEL_ID')}")
    print(f"- 总 id 数(去重后): {total}")
    print(f"- 已存在输出条数: {len(existing_results)}")
    print(f"- 待处理(未跑过): {remaining}")
    print(f"- RUN_LIMIT: {RUN_LIMIT} (None=全量)")
    print(f"- HF_ENDPOINT_URL: {os.getenv('HF_ENDPOINT_URL')}")
    print(f"- HF_DATASET_REPO: {HF_DATASET_REPO}")
    print("=====================================")

    # 先打印 3 条示例 URL，便于确认是云端图
    for iid in ids[:3]:
        print(f"[debug] cloud url sample | {iid} -> {item_id_to_hf_url(iid)}")

    added = 0
    skipped = 0
    failed = 0

    for i, item_id in enumerate(ids, start=1):
        if item_id in done_ids:
            skipped += 1
            continue

        if RUN_LIMIT is not None and added >= RUN_LIMIT:
            break

        print(f"\n[{added+1}/{RUN_LIMIT or 'ALL'}] (scan {i}/{total}) id={item_id}")

        image_url = item_id_to_hf_url(item_id)
        cues_obj = cues_map.get(item_id)

        label = None
        meta: Dict[str, Any] = {}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                label, meta = call_stage1_label(client, image_url, cues_obj)
                if label not in LABEL_SET:
                    raise ValueError(f"输出不在8类中：{label!r} | raw={meta.get('raw_output')!r}")
                break
            except Exception as e:
                meta = {
                    "error": str(e),
                    "attempt": attempt,
                    "model": os.getenv("HF_MODEL_ID"),
                    "prompt_version": PROMPT_VERSION,
                }
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * attempt)

        if label in LABEL_SET:
            obj = {
                "id": item_id,
                "stage1_label": label,
                "status": "ok",
                "image_url": image_url,
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
                "image_url": image_url,
                "has_cues": bool(cues_obj),
                "prompt_version": PROMPT_VERSION,
                "meta": meta,
            }
            existing_results.append(obj)
            done_ids.add(item_id)
            added += 1
            print(f"  ❌ 失败：{meta.get('error')}")

        if added % SAVE_EVERY == 0:
            save_json(OUT_JSON, existing_results)
            elapsed = time.time() - t0
            print(f"\n💾 已保存 {len(existing_results)} 条到: {OUT_JSON}")
            print(f"   用时: {elapsed:.1f}s | 新增:{added} 跳过:{skipped} 失败:{failed}")

        if RATE_DELAY > 0:
            time.sleep(RATE_DELAY)

    save_json(OUT_JSON, existing_results)

    elapsed = time.time() - t0
    print("\n=====================================")
    print("🎯 Stage1 Improve / InternVL 完成")
    print(f"- 输出 JSON: {OUT_JSON}")
    print(f"- 现有总条数: {len(existing_results)}")
    print(f"- 本次新增: {added}")
    print(f"- 本次跳过(已存在): {skipped}")
    print(f"- 本次失败: {failed}")
    print(f"- 总用时: {elapsed:.1f}s")
    print("=====================================")

if __name__ == "__main__":
    main()