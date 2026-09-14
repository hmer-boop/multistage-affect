# -*- coding: utf-8 -*-
"""
Stage2 Improve (LLaVA 4B Instruct):
云端图 URL + stage2_cues(辅助证据) -> 8类情感单标签（调优后的打标，LLaVA 4B）

保留 GPT improve 版的这些特性：
1) RUN_LIMIT 可控：5 / 100 / None(全量)
2) 只输出 JSON（不输出 jsonl / txt）
3) 有进度显示（当前/总数、已跳过、已新增、耗时）
4) 断点续跑：若 OUT_JSON 已存在，自动跳过已完成 id
5) 每 SAVE_EVERY 条落盘一次
6) 失败重试，避免接口偶发波动白跑

注意：
- 不再读本地原图
- 不再转 base64
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
# 0) 路径配置（按你当前 Mac 工程）
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

GOLD_IDS_PATH = PROJECT_ROOT / "01improvement method" / "method data" / "gold_item_ids.json"
CUES_PATH     = PROJECT_ROOT / "01improvement method" / "method data" / "stage2_cues.json"

OUT_DIR  = SCRIPT_DIR / "result llava_stage2"
OUT_JSON = OUT_DIR / "stage2_llava4b_labels4.json"
# =========================
# 0.1) 运行数量控制
# =========================
# 设为 5 / 100 / None(全量)
RUN_LIMIT: Optional[int] = None
SAVE_EVERY = 20
MAX_RETRIES = 3

# 请求控制
REQUEST_TIMEOUT = 120.0
RATE_DELAY = 0.0

# =========================
# 1) HF / InternVL 配置
# =========================
HF_DATASET_REPO = "hmer123/affectecom"
HF_DATASET_REV = "main"

# 仍然使用你当前已经跑通的：
# HF_ENDPOINT_URL / HF_TOKEN / HF_MODEL_ID
MODEL_NAME = os.getenv("HF_MODEL_ID", "llava-4b-instruct")

# =========================
# 2) 8类标签（固定枚举）
# =========================
LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)

PROMPT_VERSION = "stage2_scene_assisted_v1_llava4b"

SYSTEM_PROMPT = (
    "你是一个严格的图像情绪标签器。"
    "你的任务是给图片打一个情绪单标签。"
    "你必须只输出一个标签，且标签必须来自给定的 8 类。"
)

# =========================
# 3) 工具函数：读写
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
        return []

def build_done_id_set(existing: List[Dict[str, Any]]) -> set:
    done = set()
    for r in existing:
        iid = str(r.get("id"))
        if iid and iid != "None":
            done.add(iid)
    return done

def item_id_to_hf_url(item_id: str) -> str:
    # 你已跑通的 label2 日志证明这个命名方式是成立的
    return f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/{HF_DATASET_REV}/images_cloud/{item_id}.jpg"


# =========================
# 4) Stage2 cues 解析（兼容多种结构）
# =========================
def build_stage2_cues_map(cues_data: Any) -> Dict[str, Dict[str, Any]]:
    """
    兼容两种输入：
    A) list[{"id":..., "scene_cues":{...}}] 或 list[{"id":..., "entities":..., ...}]
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
# 6) InternVL 调用
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

def call_stage2_label(client: "InferenceClient", image_url: str, stage2_cues: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    user_prompt = build_user_prompt(stage2_cues)
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
# 7) 主流程
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

    ids = dedup_keep_order(ids)

    # 读 stage2 cues
    cues_data = load_json(CUES_PATH)
    cues_map = build_stage2_cues_map(cues_data)

    # 断点续跑
    existing_results = load_existing_results(OUT_JSON)
    done_ids = build_done_id_set(existing_results)

    total = len(ids)
    remaining = sum(1 for x in ids if x not in done_ids)

    client = build_client()

    print("=====================================")
    print("Stage2 Improve / LLaVA 4B 运行信息")
    print(f"- 模型: {os.getenv('HF_MODEL_ID')}")
    print(f"- 总 id 数(去重后): {total}")
    print(f"- 已存在输出条数: {len(existing_results)}")
    print(f"- 待处理(未跑过): {remaining}")
    print(f"- RUN_LIMIT: {RUN_LIMIT} (None=全量)")
    print(f"- SAVE_EVERY: {SAVE_EVERY}")
    print(f"- HF_ENDPOINT_URL: {os.getenv('HF_ENDPOINT_URL')}")
    print(f"- HF_DATASET_REPO: {HF_DATASET_REPO}")
    print("=====================================")

    # 打印 3 条示例 URL，确认走的是云端图
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
        stage2_cues_obj = cues_map.get(item_id)

        label = None
        meta: Dict[str, Any] = {}

        # 重试：最多 MAX_RETRIES 次
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                label, meta = call_stage2_label(client, image_url, stage2_cues_obj)
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
                "stage2_label": label,
                "status": "ok",
                "image_url": image_url,
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
                "image_url": image_url,
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
            print(f"   用时: {elapsed:.1f}s | 新增:{added} 跳过:{skipped} 失败:{failed}")

        if RATE_DELAY > 0:
            time.sleep(RATE_DELAY)

    # 最终保存
    save_json(OUT_JSON, existing_results)

    elapsed = time.time() - t0
    print("\n=====================================")
    print("🎯 Stage2 Improve / LLaVA 4B 完成")
    print(f"- 输出 JSON: {OUT_JSON}")
    print(f"- 现有总条数: {len(existing_results)}")
    print(f"- 本次新增: {added}")
    print(f"- 本次跳过(已存在): {skipped}")
    print(f"- 本次失败: {failed}")
    print(f"- 总用时: {elapsed:.1f}s")
    print("=====================================")

if __name__ == "__main__":
    main()