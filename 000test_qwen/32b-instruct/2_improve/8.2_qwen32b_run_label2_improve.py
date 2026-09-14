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
PROMPT_VERSION = "stage2_scene_assisted_v1_qwen32b_improve"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

METHOD_DIR = PROJECT_ROOT / "01improvement method" / "method data"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images_raw"
GOLD_IDS_PATH = METHOD_DIR / "gold_item_ids.json"
CUES_PATH = METHOD_DIR / "stage2_cues.json"

OUT_DIR = SCRIPT_DIR / "result qwen_stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_OUT = OUT_DIR / "results_stage2_qwen32b_improve.txt"
JSON_OUT = OUT_DIR / "results_stage2_qwen32b_improve.json"
STATE_JSON = OUT_DIR / "results_stage2_qwen32b_improve_state.json"
REPORT_PATH = OUT_DIR / "missing_report_stage2_qwen32b_improve.json"
LOCK_PATH = OUT_DIR / ".results_stage2_qwen32b_improve.lock"


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
        f.write("item_id\tstage2\n")
        for iid in sorted(state.keys()):
            row = state[iid]
            if row.get("status") == "ok":
                f.write(f"{iid}\t{row.get('stage2', '')}\n")


def export_json_from_state(json_path: Path, state: Dict[str, Dict[str, Any]]):
    rows = []
    for iid in sorted(state.keys()):
        row = state[iid]
        if row.get("status") == "ok":
            rows.append({"item_id": iid, "stage2": row.get("stage2")})
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


def load_gold_ids(path: Path) -> List[str]:
    data = load_json(path)
    if isinstance(data, dict) and "ids" in data:
        return dedup_keep_order([str(x) for x in data["ids"]])
    if isinstance(data, list):
        return dedup_keep_order([str(x) for x in data])
    raise ValueError("gold_item_ids.json 格式不符合预期：应为 [id,...] 或 {ids:[...]}")


def build_stage2_cues_map(cues_data: Any) -> Dict[str, Dict[str, Any]]:
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
                cues_map[iid] = {k: v for k, v in obj.items() if k != "id"}
        return cues_map

    raise ValueError("stage2_cues.json 格式不符合预期：应为 dict 或 list[dict]")


def pick_stage2_cues(cues: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cues or not isinstance(cues, dict):
        return {}
    out = {}
    for k in ("entities", "relations", "scene_or_event"):
        if k in cues and cues[k] is not None:
            out[k] = cues[k]
    return out


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
def build_stage2_prompt(stage2_cues: Optional[Dict[str, Any]]) -> str:
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
        "【第二轮判定（仅当候选为‘宁静’或‘敬畏’时才执行）】\n"
        "如果你在第一轮中选择了‘宁静’或‘敬畏’，请进行一次二次确认：\n\n"
        "宁静 二次确认条件（至少满足1条强证据）\n"
        "1) 画面整体为低激活状态，无明显强动作、强对抗或强刺激\n"
        "2) 画面中不存在威胁、紧迫、怪异、压迫或冲突事件\n"
        "3) 画面更像稳定呈现的状态或景观，而非正在发生的事件\n"
        "若缺少明确强证据，请重新选择更贴近的标签。\n\n"
        "敬畏 二次确认条件（至少满足1条强证据）\n"
        "1) 明确的宏大尺度与渺小对比，人物或物体显著渺小\n"
        "2) 宗教、祭祀或庄严肃穆的仪式性场景\n"
        "3) 自然伟力或极端险峻的壮观场景，如高山、风暴、火山、宇宙等\n"
        "若缺少明确强证据，请重新选择更贴近的标签。\n\n"
        "【最终任务】\n"
        "请只输出最终确认的那个情绪标签本身，不要输出任何解释、标点、编号或其他文字。"
    )


# ---------- 模型调用 ----------
def call_vision(client: OpenAI, model: str, instruction: str, image_data_url: str,
                request_timeout: float = 30.0) -> str:
    resp = client.with_options(timeout=request_timeout).chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content.strip()


def worker_stage2(item: Dict[str, Any], client: OpenAI, model: str,
                  max_side: int, quality: int,
                  rate_delay: float,
                  request_timeout: float = 30.0,
                  max_retries: int = 3) -> Dict[str, Any]:
    iid = item["item_id"]
    image_path = item["image_path"]
    cues_obj = item.get("stage2_cues")

    data_url = load_resize_to_data_url(image_path, max_side=max_side, quality=quality)
    prompt = build_stage2_prompt(cues_obj)

    last_err = None
    last_raw = None
    for attempt in range(1, max_retries + 1):
        try:
            out = call_vision(client, model, prompt, data_url, request_timeout=request_timeout)
            last_raw = out
            label = normalize_label(out)
            if label not in LABEL_SET:
                raise ValueError(f"输出不在8类中：{label!r} | raw={out!r}")
            if rate_delay > 0:
                time.sleep(rate_delay)
            return {
                "item_id": iid,
                "status": "ok",
                "stage2": label,
                "raw_output": out,
                "image_path": str(image_path),
                "has_stage2_cues": bool(pick_stage2_cues(cues_obj)),
                "attempt": attempt,
            }
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(1.2 * attempt)

    return {
        "item_id": iid,
        "status": "failed",
        "stage2": None,
        "raw_output": last_raw,
        "error": last_err,
        "image_path": str(image_path),
        "has_stage2_cues": bool(pick_stage2_cues(cues_obj)),
        "attempt": max_retries,
    }


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Stage2 improve labeling: TXT落盘→JSON→诊断（可断点续跑）")
    ap.add_argument("--model", default="qwen3-vl-32b-instruct", help="DashScope 视觉模型，如 qwen3-vl-32b-instruct")
    ap.add_argument("--limit", type=int, default=-1, help="处理条数（-1 全量）")
    ap.add_argument("--skip", type=int, default=0, help="跳过前 N 条（默认 0）")
    ap.add_argument("--max_side", type=int, default=1024, help="最短边最大尺寸（默认 1024）")
    ap.add_argument("--quality", type=int, default=85, help="JPEG 质量（默认 85）")
    ap.add_argument("--max_workers", type=int, default=2, help="并发线程数（默认 2）")
    ap.add_argument("--rate_delay", type=float, default=0.0, help="每次请求后的延时秒数（默认 0）")
    ap.add_argument("--request_timeout", type=float, default=60.0, help="单次API请求超时（默认60s）")
    ap.add_argument("--future_timeout", type=float, default=240.0, help="每个样本总体等待上限（默认240s）")
    ap.add_argument("--skip_on_timeout", action="store_true", help="超时跳过该样本并写入 timeout_skipped，下次如需重跑需删除 state 中对应项")
    ap.add_argument("--flush_every", type=int, default=20, help="每 N 个结果强制落盘（默认20）")
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

    gold_ids = load_gold_ids(GOLD_IDS_PATH)
    cues_map = build_stage2_cues_map(load_json(CUES_PATH))

    if args.skip > 0:
        gold_ids = gold_ids[args.skip:]
    if args.limit and args.limit > 0:
        gold_ids = gold_ids[:args.limit]

    existing_state = load_existing_state(STATE_JSON)
    done_ids = set(existing_state.keys())

    todo: List[Dict[str, Any]] = []
    input_ids: List[str] = []

    for iid in gold_ids:
        input_ids.append(iid)
        if iid in done_ids:
            continue
        img_path = find_image_file(IMAGES_DIR, iid)
        if not img_path:
            existing_state[iid] = {
                "item_id": iid,
                "status": "image_not_found",
                "stage2": None,
                "image_path": None,
                "has_stage2_cues": bool(pick_stage2_cues(cues_map.get(iid))),
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            continue
        todo.append({
            "item_id": iid,
            "image_path": img_path,
            "stage2_cues": cues_map.get(iid),
        })

    print("[debug] GOLD_IDS_PATH =", GOLD_IDS_PATH, "exists =", GOLD_IDS_PATH.exists(), "loaded =", len(gold_ids))
    print("[debug] CUES_PATH =", CUES_PATH, "exists =", CUES_PATH.exists(), "loaded =", len(cues_map))
    print(f"[index] 输入条数: {len(gold_ids)} | unique ids: {len(set(input_ids))}")
    print(f"[todo] 待处理: {len(todo)} | 已完成/已有状态: {len(done_ids)}")

    new_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [
            ex.submit(
                worker_stage2,
                it,
                client,
                args.model,
                args.max_side,
                args.quality,
                args.rate_delay,
                args.request_timeout,
            )
            for it in todo
        ]

        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=args.future_timeout)
                iid = res["item_id"]
                existing_state[iid] = {
                    "item_id": iid,
                    "status": res["status"],
                    "stage2": res.get("stage2"),
                    "raw_output": res.get("raw_output"),
                    "error": res.get("error"),
                    "image_path": res.get("image_path"),
                    "has_stage2_cues": res.get("has_stage2_cues", False),
                    "attempt": res.get("attempt"),
                    "prompt_version": PROMPT_VERSION,
                    "model": args.model,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                new_count += 1
                print(f"[stage2] {iid} -> {res.get('stage2') or res['status']}")

                if new_count % max(1, args.flush_every) == 0:
                    _acquire_lock(LOCK_PATH)
                    try:
                        write_txt_from_state(TXT_OUT, existing_state)
                        export_json_from_state(JSON_OUT, existing_state)
                        write_json(STATE_JSON, list(existing_state.values()))
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
        write_json(STATE_JSON, list(existing_state.values()))
        write_report(REPORT_PATH, input_ids, existing_state)
    finally:
        _release_lock(LOCK_PATH)

    print(f"[ok] TXT: {TXT_OUT}")
    print(f"[ok] JSON: {JSON_OUT}")
    print(f"[ok] STATE: {STATE_JSON}")
    print(f"[ok] REPORT: {REPORT_PATH}")
    print("[done] stage2 improve 完成。")


if __name__ == "__main__":
    main()
