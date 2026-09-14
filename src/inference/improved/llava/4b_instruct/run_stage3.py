# -*- coding: utf-8 -*-
"""
Stage3 Improve (LLaVA 4B Instruct):
云端图 URL + 标题 + stage2_label + text_label/text_trigger/text_confidence
-> 最终 8 类情感单标签（label-only 输出，LLaVA 4B）

对应你 GPT 的 4.2_build_stage3_lable3✅ 思路，但迁移到 LLaVA 4B：
1) 使用 Hugging Face dataset 云端图片 URL，不读本地原图，不转 base64
2) 保留断点续跑 / 定期保存 / 失败重试 / 进度显示
3) 只让模型输出一个最终标签，不输出 confidence JSON
4) 融合线索沿用原来的：title + stage2_label + text_label + text_trigger + text_confidence

输入：
- PROJECT_ROOT / "outputs" / "method_data" / "stage3_all_information.json"

输出：
- 当前脚本目录 / "result llava_stage3" / "stage3_llava4b_labels3.json"
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from huggingface_hub import InferenceClient
    _HF_OK = True
except Exception:
    _HF_OK = False


# =========================
# 0) 路径配置
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists() and (p / "src").exists())

DATA_DIR = PROJECT_ROOT / "outputs" / "method_data"
IN_PATH = DATA_DIR / "stage3_all_information.json"

OUT_DIR = SCRIPT_DIR / "result llava_stage3"
OUT_JSON = OUT_DIR / "stage3_llava4b_labels3.json"


# =========================
# 0.1) 运行控制
# =========================
RUN_LIMIT: Optional[int] = None   # 5 / 100 / None(全量)
SAVE_EVERY = 20
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120.0
RATE_DELAY = 0.0


# =========================
# 1) HF / InternVL 配置
# =========================
HF_DATASET_REPO = "hmer123/affectecom"
HF_DATASET_REV = "main"
MODEL_NAME = os.getenv("HF_MODEL_ID", "llava-4b-instruct")


# =========================
# 2) 枚举与 Prompt 配置
# =========================
LABELS = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]
LABEL_SET = set(LABELS)
TEXT_TRIGGER_SET = {"none", "event", "appraisal", "explicit_emotion"}
PROMPT_VERSION = "stage3_fusion_label_only_v1_llava4b"

SYSTEM_PROMPT = (
    "你是一个严格的图像情绪单标签分类器。"
    "你必须结合图像与给定线索，输出一个最终标签。"
    "你只能输出 8 类标签之一，且只能输出标签本身。"
)


# =========================
# 3) 基础工具函数
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


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_id(x: Any) -> str:
    return safe_str(x)


def normalize_label(text: str) -> str:
    t = safe_str(text)
    t = t.replace("标签：", "").replace("标签:", "").strip()
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    t = t.strip().strip('"').strip("'").strip()
    t = t.replace("。", "").replace("，", "").replace(",", "").strip()
    return t


def dedup_keep_order(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def item_id_to_hf_url(item_id: str) -> str:
    return (
        f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
        f"/resolve/{HF_DATASET_REV}/images_cloud/{item_id}.jpg"
    )


def resolve_title(rec: Dict[str, Any]) -> str:
    title = safe_str(rec.get("title"))
    title_zh = safe_str(rec.get("title_zh"))
    title_en = safe_str(rec.get("title_en"))

    if title:
        return title
    if title_zh and title_en:
        return f"{title_zh} / {title_en}"
    return title_zh or title_en


def load_items(path: Path) -> List[Dict[str, Any]]:
    obj = load_json(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return [x for x in obj["items"] if isinstance(x, dict)]
    raise ValueError(f"输入结构不支持：{path}")


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


def is_complete_success_record(rec: Dict[str, Any]) -> bool:
    if normalize_id(rec.get("id")) == "":
        return False
    if safe_str(rec.get("stage3_label")) not in LABEL_SET:
        return False
    if safe_str(rec.get("status")) != "ok":
        return False
    if safe_str(rec.get("error")):
        return False
    return True


def build_done_id_set(existing: List[Dict[str, Any]]) -> set:
    done = set()
    for r in existing:
        if is_complete_success_record(r):
            done.add(normalize_id(r.get("id")))
    return done


def sort_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(x: Dict[str, Any]):
        s = normalize_id(x.get("id"))
        try:
            return (0, int(s))
        except Exception:
            return (1, s)
    return sorted(records, key=sort_key)


# =========================
# 4) 输入校验与 Prompt
# =========================
def validate_inputs(
    item_id: str,
    title: str,
    stage2_label: str,
    text_label: str,
    text_trigger: str,
) -> None:
    missing = []

    if not item_id:
        missing.append("id缺失")
    if not title:
        missing.append("title缺失/为空")
    if stage2_label not in LABEL_SET:
        missing.append(f"stage2_label非法/缺失: {stage2_label}")
    if text_label not in LABEL_SET:
        missing.append(f"text_label非法/缺失: {text_label}")
    if text_trigger not in TEXT_TRIGGER_SET:
        missing.append(f"text_trigger非法/缺失: {text_trigger}")

    if missing:
        raise ValueError("stage3_input_incomplete: " + "; ".join(missing))


def build_user_prompt(
    title: str,
    stage2_label: str,
    text_label: str,
    text_trigger: str,
    text_confidence: float,
) -> str:
    labels_str = "、".join(LABELS)

    return f"""你将为该作品输出最终“情绪单标签”（8选1）。

你有四个信息源：
1) stage2_label：仅看图后的图像先验标签（可能错）
2) text_label：仅看标题后的文本建议标签（可能误导）
3) text_trigger：文本强度类型
4) text_confidence：文本建议置信度（0-1）
你将同时看到清晰图像用于最终裁决。

【已知锚点（仅看图）】
stage2_label = {stage2_label}

【文本建议（仅看标题）】
text_label = {text_label}
text_trigger = {text_trigger}
text_confidence = {text_confidence}

【标题】
{title}

【融合策略（权重）】
- stage2_label 只是先验，不是最终答案
- 当 text_trigger 较强且 text_confidence 较高时，允许文本主导
- 但若图像给出强反证，则应以图像为准

【规则（冲突如何选择）】
- text_trigger=none：强制 stay（final_label = stage2_label）
- text_trigger=event：
  若画面支持该事件/行为语义，优先 text_label（尽量不跨极性）
- text_trigger=appraisal：
  若 text_confidence >= 0.75，默认 final_label = text_label；只有强反证才推翻
- text_trigger=explicit_emotion：
  默认 final_label = text_label；只有强反证才推翻

【强反证（才允许推翻文本）】
- 文本指向负向，但画面呈现明显欢乐、庆祝、温馨互动且无危险压迫线索
- 文本指向正向，但画面呈现明确暴力、死亡、恐惧追逐、哭泣崩溃或强压迫威胁线索

【最终校正规则（易混标签复核）】
- “宁静”仅作为兜底标签，不可仅因画面安静、人物少、色调克制、构图平稳而直接判为“宁静”
- 若画面虽安静，但具有宏大、庄严、神圣、深邃、崇高，或令人产生渺小感、肃穆感、被震住的感受，应优先判为“敬畏”
- 若画面虽不喧闹，但整体明朗、温暖、轻快、舒展、富有生机、愉悦或积极活力，应优先判为“快乐”
- “恐惧”侧重危险、威胁、压迫、不安和受害风险；“厌恶”侧重脏污、腐败、黏腻、病态、反胃、嫌恶和排斥
- 仅在其他更具体情绪证据不足时，才最终选择“宁静”

【输出要求（严格）】
你必须从以下 8 个标签中只选 1 个：
{labels_str}

只输出标签本身，不要输出解释、原因、标点、JSON、编号或任何其他文字。"""


# =========================
# 5) InternVL 调用
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


def call_stage3_label(
    client: "InferenceClient",
    image_url: str,
    title: str,
    stage2_label: str,
    text_label: str,
    text_trigger: str,
    text_confidence: float,
) -> Tuple[str, Dict[str, Any]]:
    user_prompt = build_user_prompt(
        title=title,
        stage2_label=stage2_label,
        text_label=text_label,
        text_trigger=text_trigger,
        text_confidence=text_confidence,
    )
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

    out_text = safe_str(resp.choices[0].message.content)
    label = normalize_label(out_text)
    meta = {
        "model": model_id,
        "prompt_version": PROMPT_VERSION,
        "raw_output": out_text,
    }
    return label, meta


# =========================
# 6) 输出记录构造
# =========================
def build_output_record(
    rec: Dict[str, Any],
    image_url: str,
    stage3_label: Optional[str],
    status: str,
    error: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    item_id = normalize_id(rec.get("id"))
    return {
        "id": item_id,
        "title_zh": safe_str(rec.get("title_zh")),
        "title_en": safe_str(rec.get("title_en")),
        "title": resolve_title(rec),
        "path_raw": safe_str(rec.get("path_raw")),
        "image_url": image_url,
        "stage2_label": safe_str(rec.get("stage2_label")),
        "text_label": safe_str(rec.get("text_label")),
        "text_trigger": safe_str(rec.get("text_trigger")),
        "text_confidence": float(rec.get("text_confidence", 0.0) or 0.0),
        "text_error": safe_str(rec.get("text_error")),
        "stage3_label": stage3_label,
        "status": status,
        "error": error,
        "prompt_version": PROMPT_VERSION,
        "meta": meta,
    }


# =========================
# 7) 主流程
# =========================
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _HF_OK:
        raise RuntimeError("未安装 huggingface_hub 或导入失败，请先安装 huggingface_hub。")
    if not IN_PATH.exists():
        raise FileNotFoundError(f"找不到输入文件：{IN_PATH}")

    items = load_items(IN_PATH)
    # 以 id 去重，保留第一次出现
    item_map: Dict[str, Dict[str, Any]] = {}
    for obj in items:
        iid = normalize_id(obj.get("id"))
        if iid and iid not in item_map:
            item_map[iid] = obj
    ids = dedup_keep_order(list(item_map.keys()))

    existing_results = load_existing_results(OUT_JSON)
    existing_map = {normalize_id(x.get("id")): x for x in existing_results if normalize_id(x.get("id"))}
    done_ids = build_done_id_set(existing_results)

    total = len(ids)
    remaining = sum(1 for x in ids if x not in done_ids)

    client = build_client()

    print("=====================================")
    print("Stage3 Improve / LLaVA 4B 运行信息")
    print(f"- 模型: {os.getenv('HF_MODEL_ID')}")
    print(f"- 输入文件: {IN_PATH}")
    print(f"- 总 id 数(去重后): {total}")
    print(f"- 已有完整成功输出条数: {len(done_ids)}")
    print(f"- 待处理(未跑过/需补跑): {remaining}")
    print(f"- RUN_LIMIT: {RUN_LIMIT} (None=全量)")
    print(f"- SAVE_EVERY: {SAVE_EVERY}")
    print(f"- 输出文件: {OUT_JSON}")
    print(f"- HF_ENDPOINT_URL: {os.getenv('HF_ENDPOINT_URL')}")
    print(f"- HF_DATASET_REPO: {HF_DATASET_REPO}")
    print("=====================================")

    for iid in ids[:3]:
        print(f"[debug] cloud url sample | {iid} -> {item_id_to_hf_url(iid)}")

    results_map: Dict[str, Dict[str, Any]] = dict(existing_map)
    added = 0
    skipped = 0
    failed = 0

    for i, item_id in enumerate(ids, start=1):
        if item_id in done_ids:
            skipped += 1
            continue

        if RUN_LIMIT is not None and added >= RUN_LIMIT:
            break

        rec = item_map[item_id]
        title = resolve_title(rec)
        stage2_label = safe_str(rec.get("stage2_label"))
        text_label = safe_str(rec.get("text_label"))
        text_trigger = safe_str(rec.get("text_trigger"))
        text_confidence = float(rec.get("text_confidence", 0.0) or 0.0)
        image_url = item_id_to_hf_url(item_id)

        print(f"\n[{added + 1}/{RUN_LIMIT or 'ALL'}] (scan {i}/{total}) id={item_id}")

        label: Optional[str] = None
        meta: Dict[str, Any] = {}
        error = ""

        try:
            validate_inputs(
                item_id=item_id,
                title=title,
                stage2_label=stage2_label,
                text_label=text_label,
                text_trigger=text_trigger,
            )

            # 按原 GPT 规则，text_trigger=none 时直接 stay，不必浪费一次模型调用
            if text_trigger == "none":
                label = stage2_label
                meta = {
                    "model": os.getenv("HF_MODEL_ID"),
                    "prompt_version": PROMPT_VERSION,
                    "shortcut": "text_trigger=none -> stage2_label",
                }
            else:
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        label, meta = call_stage3_label(
                            client=client,
                            image_url=image_url,
                            title=title,
                            stage2_label=stage2_label,
                            text_label=text_label,
                            text_trigger=text_trigger,
                            text_confidence=text_confidence,
                        )
                        if label not in LABEL_SET:
                            raise ValueError(
                                f"输出不在8类中：{label!r} | raw={meta.get('raw_output')!r}"
                            )
                        break
                    except Exception as e:
                        error = str(e)
                        meta = {
                            "error": error,
                            "attempt": attempt,
                            "model": os.getenv("HF_MODEL_ID"),
                            "prompt_version": PROMPT_VERSION,
                        }
                        if attempt < MAX_RETRIES:
                            time.sleep(1.5 * attempt)

        except Exception as e:
            error = str(e)

        if label in LABEL_SET:
            results_map[item_id] = build_output_record(
                rec=rec,
                image_url=image_url,
                stage3_label=label,
                status="ok",
                error="",
                meta=meta,
            )
            done_ids.add(item_id)
            added += 1
            print(f"  ✅ stage3_label={label}")
        else:
            failed += 1
            results_map[item_id] = build_output_record(
                rec=rec,
                image_url=image_url,
                stage3_label=None,
                status="failed",
                error=error or meta.get("error", "未知错误"),
                meta=meta,
            )
            added += 1
            print(f"  ❌ 失败：{error or meta.get('error', '未知错误')}")

        if added % SAVE_EVERY == 0:
            save_json(OUT_JSON, sort_records(list(results_map.values())))
            elapsed = time.time() - t0
            print(f"\n💾 已保存 {len(results_map)} 条到: {OUT_JSON}")
            print(f"   用时: {elapsed:.1f}s | 新增:{added} 跳过:{skipped} 失败:{failed}")

        if RATE_DELAY > 0:
            time.sleep(RATE_DELAY)

    final_results = sort_records(list(results_map.values()))
    save_json(OUT_JSON, final_results)

    ok_count = sum(1 for x in final_results if safe_str(x.get("status")) == "ok")
    fail_count = sum(1 for x in final_results if safe_str(x.get("status")) == "failed")

    elapsed = time.time() - t0
    print("\n=====================================")
    print("🎯 Stage3 Improve / LLaVA 4B 完成")
    print(f"- 输出 JSON: {OUT_JSON}")
    print(f"- 输出总条数: {len(final_results)}")
    print(f"- 成功条数: {ok_count}")
    print(f"- 失败条数: {fail_count}")
    print(f"- 本次新增: {added}")
    print(f"- 本次跳过(已存在): {skipped}")
    print(f"- 总用时: {elapsed:.1f}s")
    print("=====================================")


if __name__ == "__main__":
    main()
