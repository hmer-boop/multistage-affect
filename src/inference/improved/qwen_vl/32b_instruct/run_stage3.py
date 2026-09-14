import os
import json
import time
import base64
import argparse
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

from PIL import Image

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

LABELS = ["宁静", "快乐", "惊奇", "敬畏", "悲伤", "恐惧", "厌恶", "愤怒"]
LABEL_SET = set(LABELS)
TEXT_TRIGGER_SET = {"none", "event", "appraisal", "explicit_emotion"}
PROMPT_VERSION = "stage3_fusion_v1_qwen32b_improve"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists() and (p / "src").exists())
METHOD_DIR = PROJECT_ROOT / "outputs" / "method_data"

IN_PATH = METHOD_DIR / "stage3_all_information.json"
OUT_DIR = SCRIPT_DIR / ("result qwen_stage3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_OUT = OUT_DIR / "results_stage3_qwen32b_improve.txt"
JSON_OUT = OUT_DIR / "results_stage3_qwen32b_improve.json"
STATE_JSON = OUT_DIR / "results_stage3_qwen32b_improve_state.json"
REPORT_PATH = OUT_DIR / "missing_report_stage3_qwen32b_improve.json"
LOCK_PATH = OUT_DIR / ".results_stage3_qwen32b_improve.lock"


# ---------- 轻量锁 ----------
def _acquire_lock(path: Path, retry_sleep: float = 0.05):
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(retry_sleep)


def _release_lock(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


# ---------- 通用 I/O ----------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def sort_key_maybe_num(s: str):
    try:
        return (0, int(s))
    except Exception:
        return (1, s)


# ---------- 数据解析 ----------
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
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return [x for x in obj["items"] if isinstance(x, dict)]
    raise ValueError(f"输入结构不支持：{path}")


def dedup_items_keep_order(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for rec in items:
        iid = normalize_id(rec.get("id", ""))
        if not iid or iid in seen:
            continue
        seen.add(iid)
        out.append(rec)
    return out


def load_existing_state(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}

    state: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                iid = normalize_id(row.get("item_id") or row.get("id") or "")
                if iid:
                    state[iid] = row
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                iid = normalize_id(v.get("item_id") or v.get("id") or k)
                if iid:
                    state[iid] = v
    return state


def resolve_title(rec: Dict[str, Any]) -> str:
    title = safe_str(rec.get("title"))
    title_zh = safe_str(rec.get("title_zh"))
    title_en = safe_str(rec.get("title_en"))
    if title:
        return title
    if title_zh and title_en:
        return f"{title_zh} / {title_en}"
    return title_zh or title_en


def resolve_image_path(rec: Dict[str, Any]) -> Path:
    image_path = safe_str(rec.get("image_path"))
    if image_path:
        p = Path(image_path)
        if p.exists():
            return p

    path_raw = safe_str(rec.get("path_raw"))
    if not path_raw:
        return Path("")

    p_raw = Path(path_raw)
    if p_raw.exists():
        return p_raw

    candidate = PROJECT_ROOT / p_raw
    if candidate.exists():
        return candidate

    return candidate


# ---------- 图像 ----------
def pil_to_data_url(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def load_resize_to_data_url(image_path: Path, max_side: int = 1280, quality: int = 85) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    max_dim = max(w, h)
    if max_dim > max_side:
        scale = max_side / float(max_dim)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return pil_to_data_url(img, quality=quality)


# ---------- 输出解析 ----------
def normalize_label(text: str) -> str:
    if not text:
        return ""
    t = text.strip().replace("：", ":").replace("标签:", "").replace("标签：", "")
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    t = t.strip().strip('"').strip("'").strip()
    aliases = {
        "开心": "快乐", "喜悦": "快乐", "高兴": "快乐",
        "安宁": "宁静", "平静": "宁静", "安静": "宁静",
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
    return t


def extract_output_text(resp: Any) -> str:
    try:
        txt = resp.choices[0].message.content
        if isinstance(txt, str):
            return txt.strip()
    except Exception:
        pass
    return ""


def parse_model_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    obj = json.loads(t)
    if not isinstance(obj, dict):
        raise ValueError(f"模型输出不是JSON对象：{obj!r}")
    return obj


# ---------- Prompt ----------
def validate_inputs(title: str, image_path: Path, stage2_label: str, text_label: str, text_trigger: str):
    missing = []
    if not title:
        missing.append("title缺失/为空")
    if not image_path or not image_path.exists():
        missing.append(f"image_path无效: {image_path}")
    if stage2_label not in LABEL_SET:
        missing.append(f"stage2_label非法/缺失: {stage2_label}")
    if text_label not in LABEL_SET:
        missing.append(f"text_label非法/缺失: {text_label}")
    if text_trigger not in TEXT_TRIGGER_SET:
        missing.append(f"text_trigger非法/缺失: {text_trigger}")
    if missing:
        raise ValueError("promptB_input_incomplete: " + "; ".join(missing))


def build_prompt_fusion(
    title: str,
    stage2_label: str,
    text_label: str,
    text_trigger: str,
    text_confidence: float,
) -> str:
    labels_str = "、".join(LABELS)
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


# ---------- 模型调用 ----------
def call_fusion(client: OpenAI, model: str, prompt: str, image_data_url: str,
                request_timeout: float = 60.0, temperature: float = 0.2) -> Tuple[Dict[str, Any], str]:
    resp = client.with_options(timeout=request_timeout).chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "你是一个严格的情绪单标签分类器，只能输出合法JSON对象。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    raw_text = extract_output_text(resp)
    out = parse_model_json(raw_text)

    label = normalize_label(str(out.get("label", "")))
    conf = out.get("confidence")
    if label not in LABEL_SET:
        raise ValueError(f"label不在8类集合中：{label!r} | raw={raw_text!r}")
    try:
        conf = float(conf)
    except Exception:
        raise ValueError(f"confidence非法：{conf!r} | raw={raw_text!r}")
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence超范围：{conf!r} | raw={raw_text!r}")

    return {"label": label, "confidence": conf}, raw_text


# ---------- 状态管理 ----------
def build_output_record(rec: Dict[str, Any], model: str) -> Dict[str, Any]:
    iid = normalize_id(rec.get("id", ""))
    title_zh = safe_str(rec.get("title_zh"))
    title_en = safe_str(rec.get("title_en"))
    title = resolve_title(rec)
    path_raw = safe_str(rec.get("path_raw"))
    image_path = resolve_image_path(rec)
    stage2_label = safe_str(rec.get("stage2_label"))
    ts_old = safe_str(rec.get("ts"))

    text_conf_raw = rec.get("text_confidence", 0.0)
    try:
        text_conf = float(text_conf_raw or 0.0)
    except Exception:
        text_conf = 0.0

    return {
        "item_id": iid,
        "id": iid,
        "title_zh": title_zh,
        "title_en": title_en,
        "title": title,
        "path_raw": path_raw,
        "image_path": str(image_path) if image_path else "",
        "stage2_label": stage2_label,
        "ts": ts_old,
        "text_label": safe_str(rec.get("text_label")),
        "text_trigger": safe_str(rec.get("text_trigger")),
        "text_confidence": text_conf,
        "text_error": safe_str(rec.get("text_error")),
        "model": model,
        "stage3": safe_str(rec.get("stage3") or rec.get("stage3_label") or rec.get("final_label")),
        "stage3_confidence": float(rec.get("stage3_confidence", rec.get("final_confidence", 0.0)) or 0.0),
        "raw_output": safe_str(rec.get("raw_output")),
        "status": safe_str(rec.get("status")),
        "error": safe_str(rec.get("error")),
        "prompt_version": safe_str(rec.get("prompt_version")) or PROMPT_VERSION,
        "updated_at": safe_str(rec.get("updated_at")),
    }


def is_complete_success_record(rec: Dict[str, Any]) -> bool:
    image_path = Path(safe_str(rec.get("image_path"))) if safe_str(rec.get("image_path")) else Path("")
    return (
        safe_str(rec.get("item_id") or rec.get("id")) != ""
        and safe_str(rec.get("stage3")) in LABEL_SET
        and safe_str(rec.get("error")) == ""
        and safe_str(rec.get("status")) == "ok"
        and safe_str(rec.get("stage2_label")) in LABEL_SET
        and safe_str(rec.get("text_label")) in LABEL_SET
        and safe_str(rec.get("text_trigger")) in TEXT_TRIGGER_SET
        and safe_str(rec.get("title")) != ""
        and image_path.exists()
    )


def write_txt_from_state(txt_path: Path, state: Dict[str, Dict[str, Any]]):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("item_id\tstage3\tstage3_confidence\n")
        for iid in sorted(state.keys(), key=sort_key_maybe_num):
            row = state[iid]
            if row.get("status") == "ok" and safe_str(row.get("stage3")) in LABEL_SET:
                f.write(f"{iid}\t{row.get('stage3', '')}\t{row.get('stage3_confidence', 0.0)}\n")


def export_json_from_state(json_path: Path, state: Dict[str, Dict[str, Any]]):
    rows = []
    for iid in sorted(state.keys(), key=sort_key_maybe_num):
        row = state[iid]
        if row.get("status") == "ok" and safe_str(row.get("stage3")) in LABEL_SET:
            rows.append({
                "item_id": iid,
                "id": iid,
                "stage3": row.get("stage3"),
                "stage3_confidence": row.get("stage3_confidence", 0.0),
            })
    write_json(json_path, rows)


def export_state_json(state_path: Path, state: Dict[str, Dict[str, Any]]):
    rows = [state[iid] for iid in sorted(state.keys(), key=sort_key_maybe_num)]
    write_json(state_path, rows)


def write_report(report_path: Path, input_ids: List[str], state: Dict[str, Dict[str, Any]]):
    ok_ids = []
    failed_ids = []
    image_not_found_ids = []
    invalid_input_ids = []
    timeout_skipped_ids = []
    unfinished_ids = []

    all_id_set = set(input_ids)

    for iid in sorted(all_id_set, key=sort_key_maybe_num):
        row = state.get(iid)
        if not row:
            unfinished_ids.append(iid)
            continue
        status = row.get("status")
        if status == "ok":
            ok_ids.append(iid)
        elif status == "image_not_found":
            image_not_found_ids.append(iid)
        elif status == "invalid_input":
            invalid_input_ids.append(iid)
        elif status == "timeout_skipped":
            timeout_skipped_ids.append(iid)
        elif status == "failed":
            failed_ids.append(iid)
        else:
            unfinished_ids.append(iid)

    report = {
        "summary": {
            "input_count": len(input_ids),
            "input_unique_ids": len(all_id_set),
            "ok_count": len(ok_ids),
            "failed_count": len(failed_ids),
            "image_not_found_count": len(image_not_found_ids),
            "invalid_input_count": len(invalid_input_ids),
            "timeout_skipped_count": len(timeout_skipped_ids),
            "unfinished_count": len(unfinished_ids),
        },
        "ok_ids": ok_ids,
        "failed_ids": failed_ids,
        "image_not_found_ids": image_not_found_ids,
        "invalid_input_ids": invalid_input_ids,
        "timeout_skipped_ids": timeout_skipped_ids,
        "unfinished_ids": unfinished_ids,
    }
    write_json(report_path, report)


# ---------- worker ----------
def worker_stage3(item: Dict[str, Any], client: OpenAI, model: str,
                  max_side: int, quality: int,
                  rate_delay: float,
                  request_timeout: float = 60.0,
                  max_retries: int = 3,
                  temperature: float = 0.2) -> Dict[str, Any]:
    base = build_output_record(item, model=model)
    iid = base["item_id"]
    image_path = Path(base["image_path"])

    try:
        validate_inputs(
            title=base["title"],
            image_path=image_path,
            stage2_label=base["stage2_label"],
            text_label=base["text_label"],
            text_trigger=base["text_trigger"],
        )
    except Exception as e:
        base["status"] = "invalid_input"
        base["stage3"] = ""
        base["stage3_confidence"] = 0.0
        base["error"] = str(e)
        base["prompt_version"] = PROMPT_VERSION
        return base

    image_data_url = load_resize_to_data_url(image_path, max_side=max_side, quality=quality)
    prompt = build_prompt_fusion(
        title=base["title"],
        stage2_label=base["stage2_label"],
        text_label=base["text_label"],
        text_trigger=base["text_trigger"],
        text_confidence=float(base["text_confidence"]),
    )

    last_err = None
    last_raw = None
    for attempt in range(1, max_retries + 1):
        try:
            out, raw_text = call_fusion(
                client=client,
                model=model,
                prompt=prompt,
                image_data_url=image_data_url,
                request_timeout=request_timeout,
                temperature=temperature,
            )
            last_raw = raw_text
            base["status"] = "ok"
            base["stage3"] = out["label"]
            base["stage3_confidence"] = float(out["confidence"])
            base["raw_output"] = raw_text
            base["error"] = ""
            base["prompt_version"] = PROMPT_VERSION
            base["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if rate_delay > 0:
                time.sleep(rate_delay)
            return base
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(1.2 * attempt)

    base["status"] = "failed"
    base["stage3"] = ""
    base["stage3_confidence"] = 0.0
    base["raw_output"] = last_raw or ""
    base["error"] = last_err or "unknown_error"
    base["prompt_version"] = PROMPT_VERSION
    base["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return base


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Stage3 improve labeling: TXT落盘→JSON→诊断（可断点续跑）")
    ap.add_argument("--model", default="qwen3-vl-32b-instruct", help="DashScope 视觉模型，如 qwen3-vl-32b-instruct")
    ap.add_argument("--limit", type=int, default=-1, help="处理条数（-1 全量）")
    ap.add_argument("--skip", type=int, default=0, help="跳过前 N 条（默认 0）")
    ap.add_argument("--max_side", type=int, default=1280, help="图像最大边尺寸（默认 1280）")
    ap.add_argument("--quality", type=int, default=85, help="JPEG 质量（默认 85）")
    ap.add_argument("--max_workers", type=int, default=2, help="并发线程数（默认 2）")
    ap.add_argument("--rate_delay", type=float, default=0.0, help="每次请求后的延时秒数（默认 0）")
    ap.add_argument("--temperature", type=float, default=0.2, help="采样温度（默认 0.2）")
    ap.add_argument("--request_timeout", type=float, default=90.0, help="单次API请求超时（默认90s）")
    ap.add_argument("--future_timeout", type=float, default=300.0, help="每个样本总体等待上限（默认300s）")
    ap.add_argument("--flush_every", type=int, default=20, help="每 N 个结果强制落盘（默认20）")
    ap.add_argument("--skip_on_timeout", action="store_true", help="超时跳过该样本，本次不写入状态，下次可继续补跑")
    return ap.parse_args()


# ---------- 主流程 ----------
def main():
    args = parse_args()

    if not _OPENAI_OK:
        print("[warn] 未安装 openai 或导入失败，本脚本不会调用模型。")
        return

    dashscope_base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=dashscope_base_url,
    )

    items_raw = load_items(IN_PATH)
    items = dedup_items_keep_order(items_raw)

    if args.skip > 0:
        items = items[args.skip:]
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    input_ids = [normalize_id(x.get("id", "")) for x in items if normalize_id(x.get("id", ""))]
    existing_state = load_existing_state(STATE_JSON)

    # 仅跳过“完整成功记录”；失败/不完整记录会补跑
    done_ids = {iid for iid, row in existing_state.items() if is_complete_success_record(row)}

    # 先把图片不存在的条目标到 state 里，便于报告统一统计
    todo: List[Dict[str, Any]] = []
    for rec in items:
        iid = normalize_id(rec.get("id", ""))
        if not iid:
            continue
        if iid in done_ids:
            continue

        image_path = resolve_image_path(rec)
        base = build_output_record(rec, model=args.model)

        if not image_path.exists():
            base["status"] = "image_not_found"
            base["stage3"] = ""
            base["stage3_confidence"] = 0.0
            base["error"] = f"image_path无效: {image_path}"
            base["prompt_version"] = PROMPT_VERSION
            base["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            existing_state[iid] = base
            continue

        todo.append(rec)

    print("[debug] IN_PATH =", IN_PATH, "exists =", IN_PATH.exists(), "loaded =", len(items_raw))
    print(f"[index] 输入条数: {len(items)} | unique ids: {len(set(input_ids))}")
    print(f"[todo] 待处理: {len(todo)} | 已有完整成功记录: {len(done_ids)} | 已写入缺图状态: {sum(1 for v in existing_state.values() if v.get('status') == 'image_not_found')}")

    new_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [
            ex.submit(
                worker_stage3,
                rec,
                client,
                args.model,
                args.max_side,
                args.quality,
                args.rate_delay,
                args.request_timeout,
                3,
                args.temperature,
            )
            for rec in todo
        ]

        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=args.future_timeout)
                iid = res["item_id"]
                existing_state[iid] = res
                new_count += 1
                print(f"[stage3] {iid} -> {res.get('stage3') or res['status']}")

                if new_count % max(1, args.flush_every) == 0:
                    _acquire_lock(LOCK_PATH)
                    try:
                        write_txt_from_state(TXT_OUT, existing_state)
                        export_json_from_state(JSON_OUT, existing_state)
                        export_state_json(STATE_JSON, existing_state)
                        write_report(REPORT_PATH, input_ids, existing_state)
                    finally:
                        _release_lock(LOCK_PATH)
                    print(f"[flush] 已落盘 {new_count} 条到 {OUT_DIR}")

            except FuturesTimeoutError:
                if args.skip_on_timeout:
                    fut.cancel()
                    print("[timeout] 某样本处理超时，已跳过，本次不写入状态，下次可继续补跑。")
                    continue
                raise
            except Exception as e:
                print(f"[error] worker: {e}")

    _acquire_lock(LOCK_PATH)
    try:
        write_txt_from_state(TXT_OUT, existing_state)
        export_json_from_state(JSON_OUT, existing_state)
        export_state_json(STATE_JSON, existing_state)
        write_report(REPORT_PATH, input_ids, existing_state)
    finally:
        _release_lock(LOCK_PATH)

    print(f"[ok] TXT: {TXT_OUT}")
    print(f"[ok] JSON: {JSON_OUT}")
    print(f"[ok] STATE: {STATE_JSON}")
    print(f"[ok] REPORT: {REPORT_PATH}")
    print("[done] stage3 improve 完成。")


if __name__ == "__main__":
    main()
