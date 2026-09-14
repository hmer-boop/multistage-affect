import os
import json
import time
import base64
import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

from PIL import Image

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
LABEL_SET = set(LABELS)
PROMPT_VERSION = "stage1_glance_v1_qwen32b_improve"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists() and (p / "src").exists())

METHOD_DIR = PROJECT_ROOT / "outputs" / "method_data"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images_raw"
GOLD_IDS_PATH = METHOD_DIR / "gold_item_ids.json"
CUES_PATH = METHOD_DIR / "stage1_cues.json"

OUT_DIR = SCRIPT_DIR / "result qwen_stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_OUT = OUT_DIR / "results_stage1_qwen32b_improve.txt"
JSON_OUT = OUT_DIR / "results_stage1_qwen32b_improve.json"
STATE_JSON = OUT_DIR / "results_stage1_qwen32b_improve_state.json"
REPORT_PATH = OUT_DIR / "missing_report_stage1_qwen32b_improve.json"
LOCK_PATH = OUT_DIR / ".results_stage1_qwen32b_improve.lock"


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


# ---------- I/O ----------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


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
            if isinstance(row, dict) and row.get("item_id"):
                state[str(row["item_id"])] = row
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                state[str(k)] = v
    return state


def write_txt_from_state(txt_path: Path, state: Dict[str, Dict[str, Any]]):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("item_id\tstage1\n")
        for iid in sorted(state.keys()):
            row = state[iid]
            if row.get("status") == "ok":
                f.write(f"{iid}\t{row.get('stage1', '')}\n")


def export_json_from_state(json_path: Path, state: Dict[str, Dict[str, Any]]):
    rows = []
    for iid in sorted(state.keys()):
        row = state[iid]
        if row.get("status") == "ok":
            rows.append({"item_id": iid, "stage1": row.get("stage1")})
    write_json(json_path, rows)


def write_report(report_path: Path, all_ids: List[str], state: Dict[str, Dict[str, Any]]):
    ok_ids = []
    failed_ids = []
    image_not_found_ids = []
    skipped_timeout_ids = []
    unfinished_ids = []

    all_id_set = set(all_ids)

    for iid in sorted(all_id_set):
        row = state.get(iid)
        if not row:
            unfinished_ids.append(iid)
            continue
        status = row.get("status")
        if status == "ok":
            ok_ids.append(iid)
        elif status == "image_not_found":
            image_not_found_ids.append(iid)
        elif status == "timeout_skipped":
            skipped_timeout_ids.append(iid)
        elif status == "failed":
            failed_ids.append(iid)
        else:
            unfinished_ids.append(iid)

    report = {
        "summary": {
            "input_count": len(all_ids),
            "input_unique_ids": len(all_id_set),
            "ok_count": len(ok_ids),
            "failed_count": len(failed_ids),
            "image_not_found_count": len(image_not_found_ids),
            "timeout_skipped_count": len(skipped_timeout_ids),
            "unfinished_count": len(unfinished_ids),
        },
        "ok_ids": ok_ids,
        "failed_ids": failed_ids,
        "image_not_found_ids": image_not_found_ids,
        "timeout_skipped_ids": skipped_timeout_ids,
        "unfinished_ids": unfinished_ids,
    }
    write_json(report_path, report)


# ---------- 预处理 ----------
def dedup_keep_order(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in ids:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def normalize_label(text: str) -> str:
    if not text:
        return "未知"
    t = text.strip().replace("：", ":").replace(" ", "")
    t = t.replace("标签:", "").replace("标签：", "")
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    t = t.strip().strip('"').strip("'").strip()
    t = t.replace("。", "").replace("，", "").replace(",", "").strip()

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


# ---------- 图像处理 ----------
def pil_to_data_url(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def load_resize_to_data_url(image_path: Path, max_side: int = 1024, quality: int = 85) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    min_side = min(w, h)
    if min_side > max_side:
        scale = max_side / float(min_side)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return pil_to_data_url(img, quality=quality)


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


# ---------- Prompt ----------
def pick_stage1_5_cues(cues_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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


def build_stage1_prompt(cues_item: Optional[Dict[str, Any]]) -> str:
    top5 = pick_stage1_5_cues(cues_item)
    cues_text = "(无)"
    if top5:
        cues_text = json.dumps(top5, ensure_ascii=False)

    return (
        "你是一个严格的图像情绪标签器。你的任务是模拟人类在1到2秒内扫视图片时的直觉情绪判断。\n"
        "【顺序要求】先看图片形成第一眼直觉，再阅读下面5条补充信息与软触发器做轻微校正。\n"
        "注意：补充信息和触发器都是概率倾向，不是硬规则；若与图片直觉冲突，以图片直觉为准。\n\n"
        "【软触发器（概率倾向）】\n"
        "1) 纹理或肌理很多、密集、粗粝、颗粒感强，整体给人不适、脏乱或黏腻感，更偏向：厌恶\n"
        "2) 大面积深色、压暗氛围、强烈阴影，或单色、大片深蓝的冷暗压迫感，更偏向：恐惧\n"
        "3) 颜色整体明亮、高饱和、色彩碰撞明显，更偏向：惊奇或快乐\n"
        "4) 颜色整体灰暗，更偏向：悲伤\n"
        "5) 大面积高饱和红、强烈红色冲突或刺激，更偏向：愤怒\n\n"
        "【补充信息（5条，全局感知）】\n"
        f"{cues_text}\n\n"
        "【防止默认选项】\n"
        "- 不要因为不确定就默认选择‘宁静’或‘敬畏’。\n"
        "- 只有当画面确实平和、舒缓、无明显冲突或刺激时，才选择：宁静。\n"
        "- 只有当画面带来宏大、庄严、神圣、崇高或压迫式震撼时，才选择：敬畏。\n\n"
        "【观看条件】\n"
        "- 不深入分析\n"
        "- 不解释原因\n"
        "- 不做象征、叙事或文化解读\n\n"
        "【任务】\n"
        "请从以下8个标签中选择一个最符合第一眼直觉的情绪：\n"
        "悲伤、恐惧、厌恶、愤怒、宁静、快乐、惊奇、敬畏\n"
        "只输出标签本身，不要输出任何解释、标点或编号。"
    )


# ---------- 模型调用 ----------
def call_stage1_vision(
    client: OpenAI,
    model: str,
    image_data_url: str,
    cues_obj: Optional[Dict[str, Any]],
    request_timeout: float = 45.0,
) -> Dict[str, Any]:
    prompt = build_stage1_prompt(cues_obj)
    resp = client.with_options(timeout=request_timeout).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严格的图像情绪标签器，只能输出给定8类标签中的一个。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    out_text = (resp.choices[0].message.content or "").strip()
    label = normalize_label(out_text)
    return {
        "label": label,
        "raw_output": out_text,
        "finish_reason": getattr(resp.choices[0], "finish_reason", None),
    }


def worker_stage1(
    item_id: str,
    cues_obj: Optional[Dict[str, Any]],
    client: OpenAI,
    model: str,
    max_side: int,
    quality: int,
    rate_delay: float,
    request_timeout: float,
    retry_times: int,
) -> Dict[str, Any]:
    img_path = find_image_file(IMAGES_DIR, item_id)
    if not img_path:
        return {
            "item_id": item_id,
            "status": "image_not_found",
            "stage1": None,
            "image_path": None,
            "has_cues": bool(cues_obj),
            "prompt_version": PROMPT_VERSION,
            "meta": {"model": model},
        }

    last_error = None
    for attempt in range(1, retry_times + 1):
        try:
            image_data_url = load_resize_to_data_url(img_path, max_side=max_side, quality=quality)
            result = call_stage1_vision(
                client=client,
                model=model,
                image_data_url=image_data_url,
                cues_obj=cues_obj,
                request_timeout=request_timeout,
            )
            label = result["label"]
            if label not in LABEL_SET:
                raise ValueError(f"输出不在8类中：{label!r} | raw={result.get('raw_output')!r}")

            if rate_delay > 0:
                time.sleep(rate_delay)

            return {
                "item_id": item_id,
                "status": "ok",
                "stage1": label,
                "image_path": str(img_path),
                "has_cues": bool(cues_obj),
                "prompt_version": PROMPT_VERSION,
                "meta": {
                    "model": model,
                    "attempt": attempt,
                    "raw_output": result.get("raw_output"),
                    "finish_reason": result.get("finish_reason"),
                },
            }
        except Exception as e:
            last_error = str(e)
            time.sleep(1.5 * attempt)

    return {
        "item_id": item_id,
        "status": "failed",
        "stage1": None,
        "image_path": str(img_path),
        "has_cues": bool(cues_obj),
        "prompt_version": PROMPT_VERSION,
        "meta": {
            "model": model,
            "retry_times": retry_times,
            "error": last_error,
        },
    }


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Stage1 improve labeling: gold ids + cues -> TXT/JSON/STATE/REPORT（可断点续跑）")
    ap.add_argument("--model", default="qwen3-vl-32b-instruct", help="DashScope 视觉模型，如 qwen3-vl-32b-instruct")
    ap.add_argument("--limit", type=int, default=-1, help="仅处理前 N 条新样本（-1 全量）")
    ap.add_argument("--skip", type=int, default=0, help="跳过前 N 条 gold id（默认0）")
    ap.add_argument("--max_side", type=int, default=1024, help="最短边最大尺寸（默认1024）")
    ap.add_argument("--quality", type=int, default=85, help="JPEG 质量（默认85）")
    ap.add_argument("--max_workers", type=int, default=2, help="并发线程数（默认2）")
    ap.add_argument("--rate_delay", type=float, default=0.0, help="每次请求后的延时秒数（默认0）")
    ap.add_argument("--request_timeout", type=float, default=60.0, help="单次API请求超时（默认60s）")
    ap.add_argument("--future_timeout", type=float, default=240.0, help="每个样本总体等待上限（默认240s）")
    ap.add_argument("--retry_times", type=int, default=3, help="单样本最大重试次数（默认3）")
    ap.add_argument("--skip_on_timeout", action="store_true", help="超时后记为 timeout_skipped，后续可重跑")
    ap.add_argument("--flush_every", type=int, default=10, help="每 N 个结果落盘一次（默认10）")
    return ap.parse_args()


def main():
    args = parse_args()

    if not _OPENAI_OK:
        raise ImportError("未安装 openai，请先 pip install openai")
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError("未检测到 DASHSCOPE_API_KEY，请先在 PyCharm Run Configuration 或环境变量中配置")

    dashscope_base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=dashscope_base_url,
    )

    gold_ids_data = load_json(GOLD_IDS_PATH)
    if isinstance(gold_ids_data, dict) and "ids" in gold_ids_data:
        gold_ids = [str(x) for x in gold_ids_data["ids"]]
    elif isinstance(gold_ids_data, list):
        gold_ids = [str(x) for x in gold_ids_data]
    else:
        raise ValueError("gold_item_ids.json 格式不符合预期：应为 [id,...] 或 {ids:[...]}")

    gold_ids = dedup_keep_order(gold_ids)
    if args.skip > 0:
        gold_ids = gold_ids[args.skip:]

    cues_data = load_json(CUES_PATH)
    cues_map: Dict[str, Dict[str, Any]] = {}
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

    state = load_existing_state(STATE_JSON)
    done_ids: Set[str] = set(state.keys())

    todo_ids = [iid for iid in gold_ids if iid not in done_ids]
    if args.limit and args.limit > 0:
        todo_ids = todo_ids[:args.limit]

    print("=====================================")
    print("Stage1 improve 运行信息")
    print(f"- 模型: {args.model}")
    print(f"- GOLD_IDS_PATH: {GOLD_IDS_PATH}")
    print(f"- CUES_PATH: {CUES_PATH}")
    print(f"- IMAGES_DIR: {IMAGES_DIR}")
    print(f"- 输出目录: {OUT_DIR}")
    print(f"- gold id 总数(去重后): {len(gold_ids)}")
    print(f"- 已存在 state 条数: {len(state)}")
    print(f"- 待处理新样本: {len(todo_ids)}")
    print(f"- limit: {args.limit}")
    print("=====================================")

    if len(todo_ids) == 0:
        export_json_from_state(JSON_OUT, state)
        write_txt_from_state(TXT_OUT, state)
        write_report(REPORT_PATH, gold_ids, state)
        print("[skip] 无待处理样本，已直接导出 TXT / JSON / REPORT")
        print(f"[ok] TXT: {TXT_OUT}")
        print(f"[ok] JSON: {JSON_OUT}")
        print(f"[ok] STATE: {STATE_JSON}")
        print(f"[ok] REPORT: {REPORT_PATH}")
        return

    new_count = 0
    t0 = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(
                    worker_stage1,
                    iid,
                    cues_map.get(iid),
                    client,
                    args.model,
                    args.max_side,
                    args.quality,
                    args.rate_delay,
                    args.request_timeout,
                    args.retry_times,
                ): iid
                for iid in todo_ids
            }

            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    row = fut.result(timeout=args.future_timeout)
                except FuturesTimeoutError:
                    if args.skip_on_timeout:
                        row = {
                            "item_id": iid,
                            "status": "timeout_skipped",
                            "stage1": None,
                            "image_path": None,
                            "has_cues": bool(cues_map.get(iid)),
                            "prompt_version": PROMPT_VERSION,
                            "meta": {
                                "model": args.model,
                                "error": f"future timeout after {args.future_timeout}s",
                            },
                        }
                        fut.cancel()
                        print(f"[timeout] {iid} -> timeout_skipped")
                    else:
                        raise
                except Exception as e:
                    row = {
                        "item_id": iid,
                        "status": "failed",
                        "stage1": None,
                        "image_path": None,
                        "has_cues": bool(cues_map.get(iid)),
                        "prompt_version": PROMPT_VERSION,
                        "meta": {
                            "model": args.model,
                            "error": str(e),
                        },
                    }
                    print(f"[error] {iid} -> {e}")

                state[row["item_id"]] = row
                new_count += 1

                if row["status"] == "ok":
                    print(f"[stage1 improve] {row['item_id']} -> {row['stage1']}")
                else:
                    print(f"[stage1 improve] {row['item_id']} -> {row['status']}")

                if new_count % max(1, args.flush_every) == 0:
                    _acquire_lock(LOCK_PATH)
                    try:
                        write_json(STATE_JSON, list(state.values()))
                        write_txt_from_state(TXT_OUT, state)
                        export_json_from_state(JSON_OUT, state)
                        write_report(REPORT_PATH, gold_ids, state)
                    finally:
                        _release_lock(LOCK_PATH)
                    elapsed = time.time() - t0
                    print(f"[flush] 已落盘 {new_count} 条 | elapsed={elapsed:.1f}s")
    finally:
        _acquire_lock(LOCK_PATH)
        try:
            write_json(STATE_JSON, list(state.values()))
            write_txt_from_state(TXT_OUT, state)
            export_json_from_state(JSON_OUT, state)
            write_report(REPORT_PATH, gold_ids, state)
        finally:
            _release_lock(LOCK_PATH)

    elapsed = time.time() - t0
    print("=====================================")
    print("[done] stage1 improve 完成")
    print(f"[ok] TXT: {TXT_OUT}")
    print(f"[ok] JSON: {JSON_OUT}")
    print(f"[ok] STATE: {STATE_JSON}")
    print(f"[ok] REPORT: {REPORT_PATH}")
    print(f"[time] {elapsed:.1f}s")
    print("=====================================")


if __name__ == "__main__":
    main()
