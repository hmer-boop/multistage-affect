# -*- coding: utf-8 -*-
"""
4.2_build_stage3_lable3.py

Stage3：B段，图文融合最终裁决

输入：
- 4.1.3_stage3_all_information.json
- results/summary.json / results/summary_round2.json 中的 caption_zh、caption_en

输出：
- stage3_gpt_labels3.json

说明：
- 基于清晰原图 + 标题 + 图像描述 + stage2_label + title-derived text cues 做最终 stage3 label
- 已有完整结果自动跳过，避免重复跑
- 已有不完整/失败结果不会跳过，会补跑
- 输出格式与既有格式 stage3_gpt_labels3.json 保持一致；图像描述只作为 prompt 输入，不写入输出
- 结束后自动检查缺失项并打印统计
"""

import os
import json
import time
import base64
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "outputs" / "method_data")
IN_PATH = os.path.join(DATA_DIR, "stage3_all_information.json")
OUT_PATH = os.path.join(DATA_DIR, "stage3_gpt_labels3.json")
OUT_TMP = OUT_PATH + ".tmp"
SUMMARY_PATHS = [
    os.path.join(str(PROJECT_ROOT), "results", "summary.json"),
    os.path.join(str(PROJECT_ROOT), "results", "summary_round2.json"),
    os.path.join(str(PROJECT_ROOT), "results", "summary_1person.json"),
]

LIMIT = 0
SAVE_EVERY = 1
MAX_RETRIES = 5

LABELS_8 = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]
TEXT_TRIGGER_SET = {"none", "event", "appraisal", "explicit_emotion"}


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    if not s:
        return None
    return json.loads(s)


def atomic_save(obj: Any, tmp_path: str, final_path: str):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def normalize_id(x: Any) -> str:
    return str(x).strip()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_items(path: str) -> List[Dict[str, Any]]:
    obj = load_json(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
        return [x for x in obj["items"] if isinstance(x, dict)]
    raise ValueError(f"输入结构不支持：{path}")


def load_existing_out(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
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


def _safe_get(d: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def _iter_records(obj: Any):
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
    elif isinstance(obj, dict):
        if isinstance(obj.get("items"), list):
            for it in obj["items"]:
                if isinstance(it, dict):
                    yield it
        else:
            for k, v in obj.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    if "id" not in vv and "item_id" not in vv:
                        vv["id"] = k
                    yield vv


def load_caption_map(paths: List[str]) -> Dict[str, Dict[str, str]]:
    """
    从 summary 文件中按 id / item_id 读取 caption_zh、caption_en。
    这些描述只进入最终 Stage III fusion，不参与 text_label/text_trigger/text_confidence 抽取。
    """
    caption_map: Dict[str, Dict[str, str]] = {}

    for path in paths:
        obj = load_json(path)
        if obj is None:
            continue

        for it in _iter_records(obj):
            sid = normalize_id(_safe_get(it, ["id", "item_id", "wulunnage_id"], default=""))
            if not sid:
                continue

            caption_zh = safe_str(_safe_get(
                it,
                ["caption_zh", "description_zh", "desc_zh", "caption_cn", "description_cn"],
                default="",
            ))
            caption_en = safe_str(_safe_get(
                it,
                ["caption_en", "description_en", "desc_en", "caption", "description", "desc"],
                default="",
            ))

            if not caption_zh and not caption_en:
                continue

            if sid not in caption_map:
                caption_map[sid] = {"caption_zh": caption_zh, "caption_en": caption_en}
            else:
                old = caption_map[sid]
                if not old.get("caption_zh") and caption_zh:
                    old["caption_zh"] = caption_zh
                if not old.get("caption_en") and caption_en:
                    old["caption_en"] = caption_en

    return caption_map


def attach_caption(rec: Dict[str, Any], caption_map: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    out = dict(rec)
    _id = normalize_id(out.get("id", ""))
    item_id = normalize_id(out.get("item_id", ""))
    cap = caption_map.get(_id) or caption_map.get(item_id) or {}

    if not safe_str(out.get("caption_zh")):
        out["caption_zh"] = safe_str(cap.get("caption_zh"))
    if not safe_str(out.get("caption_en")):
        out["caption_en"] = safe_str(cap.get("caption_en"))
    return out


def resolve_title(rec: Dict[str, Any]) -> str:
    title_zh = safe_str(rec.get("title_zh"))
    title_en = safe_str(rec.get("title_en"))
    title = safe_str(rec.get("title"))

    if title:
        return title
    if title_zh and title_en:
        return f"{title_zh} / {title_en}"
    return title_zh or title_en


def resolve_description(rec: Dict[str, Any]) -> str:
    caption_zh = safe_str(rec.get("caption_zh"))
    caption_en = safe_str(rec.get("caption_en"))

    if caption_zh and caption_en and caption_zh != caption_en:
        return f"中文描述：{caption_zh}\n英文描述：{caption_en}"
    return caption_zh or caption_en


def resolve_image_path(rec: Dict[str, Any]) -> str:
    """
    优先使用绝对路径字段；如果 path_raw 是相对路径，则自动拼到项目根目录。
    """
    image_path = safe_str(rec.get("image_path"))
    if image_path and os.path.exists(image_path):
        return image_path

    path_raw = safe_str(rec.get("path_raw"))
    if not path_raw:
        return ""

    # 直接存在
    if os.path.exists(path_raw):
        return path_raw

    # 相对路径转绝对路径：以 PythonProject 为根
    candidate = os.path.normpath(os.path.join(PROJECT_ROOT, path_raw))
    if os.path.exists(candidate):
        return candidate

    return path_raw


def parse_model_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def openai_client():
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("未检测到 OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY, timeout=60)


def extract_output_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text
    try:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str):
                        chunks.append(txt)
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def read_image_as_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".bmp":
        mime = "image/bmp"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def validate_prompt_b_inputs(
    image_path: str,
    stage2_label: str,
    text_label: str,
    text_trigger: str,
    title: str,
    description: str,
) -> None:
    missing = []

    if not image_path or not os.path.exists(image_path):
        missing.append(f"image_path无效: {image_path}")
    if not title:
        missing.append("title缺失/为空")
    if stage2_label not in LABELS_8:
        missing.append(f"stage2_label非法/缺失: {stage2_label}")
    if text_label not in LABELS_8:
        missing.append(f"text_label非法/缺失: {text_label}")
    if text_trigger not in TEXT_TRIGGER_SET:
        missing.append(f"text_trigger非法/缺失: {text_trigger}")

    if missing:
        raise ValueError("promptB_input_incomplete: " + "; ".join(missing))


def build_prompt_fusion(
    title: str,
    description: str,
    stage2_label: str,
    text_label: str,
    text_trigger: str,
    text_confidence: float,
) -> str:
    labels_str = "、".join(LABELS_8)
    description = description or "（无描述）"
    return f"""你将为该作品输出最终“情绪单标签”（8选1）。

你有五个信息源：
1) stage2_label：仅看图的直觉标签（可能错）
2) text_label：仅由标题生成的文本建议（可能误导）
3) text_trigger：标题触发的文本强度类型
4) text_confidence：标题侧文本建议置信度（0-1）
5) description：图像内容的文本化描述，只作为视觉内容辅助，不参与 text_label/text_trigger/text_confidence 的生成
你将看到清晰图像用于最终裁决。

【已知锚点（仅看图）】
stage2_label = {stage2_label}

【标题侧文本建议（仅由标题生成）】
text_label = {text_label}
text_trigger = {text_trigger}
text_confidence = {text_confidence}

【标题】
{title}

【图像描述】
{description}

【融合策略（权重）】
- stage2_label 仅为先验（prior）
- 当 text_trigger 较强且 text_confidence 较高时，允许文本主导
- description 仅用于帮助确认图像内容与场景语义，不能替代清晰图像本身，也不能覆盖强视觉反证

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
{{"label":"<{labels_str}之一>","confidence":0-1}}
"""


def call_fusion(model: str, image_path: str, prompt: str, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    client = openai_client()
    data_url = read_image_as_data_url(image_path)
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "你是一个严格的情绪单标签分类器，只能输出合法JSON对象。"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    },
                ],
                temperature=0.2,
            )
            text = extract_output_text(resp)
            out = parse_model_json(text)

            if out.get("label") not in LABELS_8:
                raise ValueError(f"label不在8类集合中：{out.get('label')}")
            conf = out.get("confidence")
            if not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
                raise ValueError(f"confidence非法：{conf}")
            return out

        except Exception as e:
            last_err = e
            err_str = str(e).lower()

            if "timeout" in err_str or "timed out" in err_str:
                raise RuntimeError(f"fusion 超时跳过: {e}")

            if "connection" in err_str or "rate limit" in err_str or "429" in err_str:
                time.sleep(min(2 ** attempt, 20))
                continue

            time.sleep(min(2 ** attempt, 20))

    raise RuntimeError(f"fusion 调用失败: {last_err}")


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
        "stage3_confidence": float(
            rec.get("stage3_confidence", rec.get("final_confidence", 0.0)) or 0.0
        ),
        "error": safe_str(rec.get("error")),
    }


def is_complete_success_record(rec: Dict[str, Any]) -> bool:
    if safe_str(rec.get("stage3_label")) not in LABELS_8:
        return False
    if safe_str(rec.get("error")):
        return False
    if not safe_str(rec.get("id")):
        return False
    if not safe_str(rec.get("title_zh")) and not safe_str(rec.get("title_en")) and not safe_str(rec.get("title")):
        return False
    if safe_str(rec.get("stage2_label")) not in LABELS_8:
        return False
    if safe_str(rec.get("text_label")) not in LABELS_8:
        return False
    if safe_str(rec.get("text_trigger")) not in TEXT_TRIGGER_SET:
        return False
    if not safe_str(rec.get("image_path")) or not os.path.exists(safe_str(rec.get("image_path"))):
        return False
    return True


def _sorted_out_list(id2rec: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(s: str):
        try:
            return (0, int(s))
        except Exception:
            return (1, s)
    return [id2rec[k] for k in sorted(id2rec.keys(), key=sort_key)]


def validate_output_record(rec: Dict[str, Any]) -> List[str]:
    required_fields = [
        "id",
        "title_zh",
        "title_en",
        "title",
        "path_raw",
        "image_path",
        "stage2_label",
        "ts",
        "text_label",
        "text_trigger",
        "text_confidence",
        "text_error",
        "model",
        "stage3_label",
        "stage3_confidence",
        "error",
    ]
    missing = []
    for k in required_fields:
        if k not in rec:
            missing.append(k)
            continue
        if rec[k] is None:
            missing.append(k)
    return missing


def main():
    model = DEFAULT_MODEL
    items = load_items(IN_PATH)
    existing = load_existing_out(OUT_PATH)
    caption_map = load_caption_map(SUMMARY_PATHS)

    # 先用新输入标准化已有输出，避免已有记录字段不齐
    out_map: Dict[str, Dict[str, Any]] = {}
    input_map = {
        normalize_id(x.get("id", "")): attach_caption(x, caption_map)
        for x in items
        if normalize_id(x.get("id", ""))
    }

    for _id, rec in input_map.items():
        merged = dict(rec)
        if _id in existing:
            old = existing[_id]
            merged["stage3_label"] = old.get("stage3_label", old.get("final_label", ""))
            merged["stage3_confidence"] = old.get(
                "stage3_confidence", old.get("final_confidence", 0.0)
            )
            merged["error"] = old.get("error", "")
            # 保留旧输出中的最终模型字段，但本次会统一改 model
        out_map[_id] = build_output_record(merged, model=model)

    done_ids: Set[str] = set()
    for _id, rec in out_map.items():
        if is_complete_success_record(rec):
            done_ids.add(_id)

    candidates = [input_map[_id] for _id in input_map if _id not in done_ids]

    if LIMIT and LIMIT > 0:
        candidates = candidates[:LIMIT]

    print(f"[INFO] 输入总条目数: {len(items)}")
    print(f"[INFO] 可匹配 caption 的条目数: {len(caption_map)}")
    print(f"[INFO] 已有完整成功记录(跳过): {len(done_ids)}")
    print(f"[INFO] 待新增/补跑数量: {len(candidates)}")

    new_ok = 0
    new_fail = 0

    pbar = tqdm(total=len(candidates), desc="Stage3 B fusion", unit="img")

    for rec in candidates:
        _id = normalize_id(rec.get("id", ""))
        base = build_output_record(rec, model=model)

        title = safe_str(base.get("title"))
        description = resolve_description(rec)
        image_path = safe_str(base.get("image_path"))
        stage2_label = safe_str(base.get("stage2_label"))
        text_label = safe_str(base.get("text_label"))
        text_trigger = safe_str(base.get("text_trigger"))
        text_conf = float(base.get("text_confidence", 0.0) or 0.0)

        try:
            validate_prompt_b_inputs(
                image_path=image_path,
                stage2_label=stage2_label,
                text_label=text_label,
                text_trigger=text_trigger,
                title=title,
                description=description,
            )

            prompt_b = build_prompt_fusion(
                title=title,
                description=description,
                stage2_label=stage2_label,
                text_label=text_label,
                text_trigger=text_trigger,
                text_confidence=text_conf,
            )
            b = call_fusion(model=model, image_path=image_path, prompt=prompt_b)

            base["stage3_label"] = b["label"]
            base["stage3_confidence"] = float(b["confidence"])
            base["error"] = ""
            base["model"] = model
            new_ok += 1

        except Exception as e:
            base["stage3_label"] = ""
            base["stage3_confidence"] = 0.0
            base["error"] = str(e)
            base["model"] = model
            new_fail += 1

        out_map[_id] = base

        if (new_ok + new_fail) % SAVE_EVERY == 0:
            atomic_save(_sorted_out_list(out_map), OUT_TMP, OUT_PATH)

        pbar.update(1)

    pbar.close()
    atomic_save(_sorted_out_list(out_map), OUT_TMP, OUT_PATH)

    # 总结检查
    final_list = _sorted_out_list(out_map)
    total_now = len(final_list)

    missing_counter = Counter()
    missing_examples = []

    for r in final_list:
        missing_keys = validate_output_record(r)
        for k in missing_keys:
            missing_counter[f"{k} 字段不存在或为 None"] += 1

        if not safe_str(r.get("image_path")) or not os.path.exists(safe_str(r.get("image_path"))):
            reason = f"image_path无效: {safe_str(r.get('image_path'))}"
            missing_counter[reason] += 1
            if len(missing_examples) < 20:
                missing_examples.append((safe_str(r.get("id")), "image_path", reason))

        if safe_str(r.get("stage2_label")) not in LABELS_8:
            reason = f"stage2_label非法/缺失: {safe_str(r.get('stage2_label'))}"
            missing_counter[reason] += 1
            if len(missing_examples) < 20:
                missing_examples.append((safe_str(r.get("id")), "stage2_label", reason))

        if safe_str(r.get("text_label")) not in LABELS_8:
            reason = safe_str(r.get("text_error")) or f"text_label非法/缺失: {safe_str(r.get('text_label'))}"
            missing_counter[f"text_label问题: {reason}"] += 1
            if len(missing_examples) < 20:
                missing_examples.append((safe_str(r.get("id")), "text_label", reason))

        if safe_str(r.get("text_trigger")) not in TEXT_TRIGGER_SET:
            reason = f"text_trigger非法/缺失: {safe_str(r.get('text_trigger'))}"
            missing_counter[reason] += 1
            if len(missing_examples) < 20:
                missing_examples.append((safe_str(r.get("id")), "text_trigger", reason))

        if safe_str(r.get("stage3_label")) not in LABELS_8:
            reason = safe_str(r.get("error")) or "stage3_label缺失"
            missing_counter[f"stage3_label问题: {reason}"] += 1
            if len(missing_examples) < 20:
                missing_examples.append((safe_str(r.get("id")), "stage3_label", reason))

    total_missing = sum(missing_counter.values())

    print(f"[DONE] 本次新增成功: {new_ok}")
    print(f"[DONE] 本次新增失败: {new_fail}")
    print(f"[DONE] 当前输出总条目数: {total_now}")

    if total_missing == 0:
        print("[CHECK] 当前无条目缺失，所有关键字段齐全。")
    else:
        print(f"[CHECK] 发现 {total_missing} 处缺失/异常：")
        for reason, cnt in missing_counter.items():
            print(f"  - {cnt} 条：{reason}")

        if missing_examples:
            print("[CHECK] 缺失样例（最多20条）：")
            for _id, field_name, reason in missing_examples:
                print(f"  - id={_id} 缺少/异常 {field_name}，原因是：{reason}")


if __name__ == "__main__":
    main()
