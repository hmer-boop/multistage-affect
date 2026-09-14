# -*- coding: utf-8 -*-
"""
4.1.2：A段，只看标题做文本情绪判定

输入：
- stage3_titles_only.json   （由 4.1.1 生成）

输出：
- stage3_text_only_labels.json

说明：
- 只根据标题做文本建议，不看图
- 不依赖 stage2_gpt_labels.json
- 保留断点续跑 / 防重复 / 定期落盘 / 进度显示
- 启动时会先把已有输出标准化为统一格式，再处理新增条目
"""

import os
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "outputs" / "method_data")
IN_PATH = os.path.join(DATA_DIR, "stage3_1titles_only.json")
OUT_PATH = os.path.join(DATA_DIR, "stage3_text_only_labels.json")
OUT_TMP = OUT_PATH + ".tmp"

LIMIT = 0
SAVE_EVERY = 1
MAX_RETRIES = 5

LABELS_8 = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]

TEXT_TRIGGER_SCHEMA = {
    "none": {"description": "文本未提供有效情绪增量，仅为客观名词或弱信息", "strength": 0},
    "event": {"description": "文本包含客观事件或行为信息，如战争、葬礼、追逐", "strength": 1},
    "appraisal": {"description": "文本包含评价性或气氛修饰，如庄严、压抑、神圣", "strength": 2},
    "explicit_emotion": {"description": "文本直接出现明确情绪词，如恐惧、悲伤、愤怒", "strength": 3},
}

REQUIRED_FIELDS = [
    "id",
    "title_zh",
    "title_en",
    "text_label",
    "text_trigger",
    "text_confidence",
    "text_error",
    "model",
    "ts",
]


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


def load_items(path: str) -> List[Dict[str, Any]]:
    obj = load_json(path)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
        return [x for x in obj["items"] if isinstance(x, dict)]
    raise ValueError(f"输入结构不支持，必须是 list 或 {{'items': [...]}}，当前文件：{path}")


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
                # 兼容既有格式 dict 结构
                if "id" not in v:
                    v = dict(v)
                    v["id"] = normalize_id(k)
                mp[normalize_id(v["id"])] = v
    return mp


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


def build_prompt_text_only(title_zh: str, title_en: str) -> str:
    labels_str = "、".join(LABELS_8)

    title_zh = (title_zh or "").strip()
    title_en = (title_en or "").strip()

    return f"""你将仅根据“标题文字”为作品预测一个情绪单标签（8选1）。

重要限制：
1. 你不能看到图片。
2. 你不能假设任何画面内容。
3. 你只能依据下面给出的中文标题和英文标题字面信息进行判断。
4. 若标题信息很弱，也必须在8类中选1个最可能标签，并给出较低置信度。
5. 不要进行艺术阐释、象征延伸或脑补画面。

【中文标题】
{title_zh}

【英文标题】
{title_en}

【text_trigger 定义】
- none: 文本未提供有效情绪增量，仅为客观名词或弱信息
- event: 文本包含客观事件或行为信息
- appraisal: 文本包含评价性或气氛修饰
- explicit_emotion: 文本直接出现明确情绪词

【输出格式（严格）】
只输出一个JSON对象（不要Markdown，不要解释，不要多余文字）：
{{"text_label":"<{labels_str}之一>","text_trigger":"none|event|appraisal|explicit_emotion","confidence":0-1}}
"""


def call_text_only(model: str, prompt: str, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    client = openai_client()
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "你是一个严格的文本情绪单标签分类器，只能依据标题文字做判断，并且只能输出合法JSON对象。"},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
                ],
                temperature=0.2,
            )
            text = extract_output_text(resp)
            out = parse_model_json(text)

            if out.get("text_label") not in LABELS_8:
                raise ValueError(f"text_label不在8类集合中：{out.get('text_label')}")
            if out.get("text_trigger") not in TEXT_TRIGGER_SCHEMA:
                raise ValueError(f"text_trigger非法：{out.get('text_trigger')}")
            conf = out.get("confidence")
            if not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
                raise ValueError(f"confidence非法：{conf}")

            return out

        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "timeout" in err_str or "timed out" in err_str:
                raise RuntimeError(f"text-only 超时: {e}")
            time.sleep(min(2 ** attempt, 20))

    raise RuntimeError(f"text-only 调用失败: {last_err}")


def _sorted_out_list(id2rec: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(s: str):
        try:
            return (0, int(s))
        except Exception:
            return (1, s)
    return [id2rec[k] for k in sorted(id2rec.keys(), key=sort_key)]


def build_base_output_by_values(
    _id: str,
    title_zh: str,
    title_en: str,
    model: str,
    old: Dict[str, Any] = None,
) -> Dict[str, Any]:
    old = old or {}
    return {
        "id": _id,
        "title_zh": (title_zh or "").strip(),
        "title_en": (title_en or "").strip(),
        "text_label": (old.get("text_label") or "").strip(),
        "text_trigger": (old.get("text_trigger") or "none").strip() or "none",
        "text_confidence": float(old.get("text_confidence", 0.0) or 0.0),
        "text_error": (old.get("text_error") or "").strip(),
        "model": (old.get("model") or model).strip(),
        "ts": (old.get("ts") or time.strftime("%Y-%m-%d %H:%M:%S")).strip(),
    }


def normalize_existing_outputs(
    items: List[Dict[str, Any]],
    existing: Dict[str, Dict[str, Any]],
    model: str,
) -> Dict[str, Dict[str, Any]]:
    """
    用输入文件 stage3_titles_only.json 作为标题真源，
    把已有输出全部改写成统一格式，只保留规定字段。
    """
    title_map: Dict[str, Dict[str, str]] = {}
    for rec in items:
        _id = normalize_id(rec.get("id", ""))
        if not _id:
            continue
        title_map[_id] = {
            "title_zh": (rec.get("title_zh") or "").strip(),
            "title_en": (rec.get("title_en") or "").strip(),
        }

    normalized: Dict[str, Dict[str, Any]] = {}

    # 先覆盖已有输出：已有记录也强制改成目标格式
    for _id, old in existing.items():
        src_title = title_map.get(_id, {})
        normalized[_id] = build_base_output_by_values(
            _id=_id,
            title_zh=src_title.get("title_zh", old.get("title_zh", "")),
            title_en=src_title.get("title_en", old.get("title_en", "")),
            model=model,
            old=old,
        )

    # 再补齐输入里有、输出里还没有的条目，占位成统一格式
    for _id, src in title_map.items():
        if _id not in normalized:
            normalized[_id] = build_base_output_by_values(
                _id=_id,
                title_zh=src.get("title_zh", ""),
                title_en=src.get("title_en", ""),
                model=model,
                old=None,
            )

    return normalized


def validate_output_record(r: Dict[str, Any]) -> List[str]:
    missing = []

    for k in REQUIRED_FIELDS:
        if k not in r:
            missing.append(k)
            continue
        if k == "text_confidence":
            if r[k] is None:
                missing.append(k)
        else:
            if r[k] is None:
                missing.append(k)

    return missing


def main():
    model = DEFAULT_MODEL
    items = load_items(IN_PATH)
    existing = load_existing_out(OUT_PATH)

    # 关键修改：先把已有输出全部改成统一格式
    out_map = normalize_existing_outputs(items=items, existing=existing, model=model)

    # 先保存一次，确保旧内容也被覆盖成新格式
    atomic_save(_sorted_out_list(out_map), OUT_TMP, OUT_PATH)

    done_ids: Set[str] = set()
    for k, v in out_map.items():
        if isinstance(v, dict) and v.get("text_label") in LABELS_8 and not v.get("text_error"):
            done_ids.add(k)

    candidates = [x for x in items if normalize_id(x.get("id", "")) not in done_ids]

    if LIMIT and LIMIT > 0:
        candidates = candidates[:LIMIT]

    print(f"[INFO] 输入文件: {IN_PATH}")
    print(f"[INFO] 总条目数: {len(items)}")
    print(f"[INFO] 已有输出已标准化: {len(out_map)}")
    print(f"[INFO] 已完成数量: {len(done_ids)}")
    print(f"[INFO] 待处理数量: {len(candidates)}")

    pbar = tqdm(total=len(candidates), desc="Stage3 title-only", unit="img")

    new_ok = 0
    new_fail = 0
    skipped_missing_title = 0

    for rec in candidates:
        _id = normalize_id(rec.get("id", ""))
        title_zh = (rec.get("title_zh") or "").strip()
        title_en = (rec.get("title_en") or "").strip()

        base = build_base_output_by_values(
            _id=_id,
            title_zh=title_zh,
            title_en=title_en,
            model=model,
            old=out_map.get(_id, {}),
        )

        if not title_zh and not title_en:
            base["text_error"] = "title_zh和title_en都为空，未调用模型"
            out_map[_id] = base
            skipped_missing_title += 1
            new_fail += 1

            if (new_ok + new_fail) % SAVE_EVERY == 0:
                atomic_save(_sorted_out_list(out_map), OUT_TMP, OUT_PATH)

            pbar.update(1)
            continue

        try:
            prompt_a = build_prompt_text_only(title_zh=title_zh, title_en=title_en)
            a = call_text_only(model=model, prompt=prompt_a)

            base["text_label"] = a["text_label"]
            base["text_trigger"] = a["text_trigger"]
            base["text_confidence"] = float(a["confidence"])
            base["text_error"] = ""
            base["model"] = model
            base["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")

            out_map[_id] = base
            new_ok += 1

        except Exception as e:
            base["text_label"] = ""
            base["text_trigger"] = "none"
            base["text_confidence"] = 0.0
            base["text_error"] = str(e)
            base["model"] = model
            base["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")

            out_map[_id] = base
            new_fail += 1

        if (new_ok + new_fail) % SAVE_EVERY == 0:
            atomic_save(_sorted_out_list(out_map), OUT_TMP, OUT_PATH)

        pbar.update(1)

    pbar.close()
    atomic_save(_sorted_out_list(out_map), OUT_TMP, OUT_PATH)

    print(f"[DONE] 成功: {new_ok}")
    print(f"[DONE] 失败: {new_fail}")
    print(f"[DONE] 其中标题双空跳过: {skipped_missing_title}")
    print(f"[DONE] 输出文件: {OUT_PATH}")

    final_list = _sorted_out_list(out_map)
    missing_counter = Counter()
    missing_examples = []

    for r in final_list:
        missing_keys = validate_output_record(r)

        if not str(r.get("id", "")).strip():
            missing_keys.append("id")

        if r.get("text_label", "") == "":
            reason = r.get("text_error", "") or "模型未返回text_label"
            missing_counter[f"text_label缺失，原因是：{reason}"] += 1
            if len(missing_examples) < 20:
                missing_examples.append((r.get("id", ""), "text_label", reason))

        if r.get("text_trigger", "") == "":
            reason = r.get("text_error", "") or "模型未返回text_trigger"
            missing_counter[f"text_trigger缺失，原因是：{reason}"] += 1

        if r.get("model", "") == "":
            missing_counter["model缺失，原因是：model为空"] += 1

        if r.get("ts", "") == "":
            missing_counter["ts缺失，原因是：时间戳为空"] += 1

        for k in missing_keys:
            if k != "text_label":
                missing_counter[f"{k}缺失，原因是：字段不存在或值为None"] += 1

    if not missing_counter:
        print("[CHECK] 所有条目字段齐全，无缺失项。")
    else:
        total_missing = sum(missing_counter.values())
        print(f"[CHECK] 发现 {total_missing} 处缺失/异常字段：")
        for reason, cnt in missing_counter.items():
            print(f"  - {cnt} 条数据 {reason}")

        if missing_examples:
            print("[CHECK] 缺失样例（最多20条）：")
            for _id, field_name, reason in missing_examples:
                print(f"  - id={_id} 缺少 {field_name}，原因是 {reason}")


if __name__ == "__main__":
    main()
