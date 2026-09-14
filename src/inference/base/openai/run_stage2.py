# -*- coding: utf-8 -*-
"""
8_run_stage2.py —— Stage2 单阶段标注（先TXT落盘→再导出JSON→诊断）
- 输入:
  - dataset/metadata/items_stage1_2.json
  - dataset/metadata/items_stage1_2_round2.json
- 输出: result gpt_stage2/
  - results_stage2.txt
  - results_stage2.json
  - missing_report_stage2.json
  - results_stage2_round2.txt
  - results_stage2_round2.json
  - missing_report_stage2_round2.json

断点续跑：各自 TXT 中已有 stage2 的 item_id 将被跳过
"""

import os
import json
import time
import base64
import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

from PIL import Image

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists() and (p / "src").exists())
META_DIR = PROJECT_ROOT / "dataset" / "metadata"

# ===== 两个输入文件 =====
INPUT_PATH = META_DIR / "items_stage1_2.json"
INPUT_PATH_round2 = META_DIR / "items_stage1_2_round2.json"

OUT_DIR = BASE_DIR / "result gpt_stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 原始输出 =====
TXT_OUT = OUT_DIR / "results_stage2.txt"
JSON_OUT = OUT_DIR / "results_stage2.json"
REPORT_PATH = OUT_DIR / "missing_report_stage2.json"
LOCK_PATH = OUT_DIR / ".results_stage2.lock"

# ===== round2 输出 =====
TXT_OUT_round2 = OUT_DIR / "results_stage2_round2.txt"
JSON_OUT_round2 = OUT_DIR / "results_stage2_round2.json"
REPORT_PATH_round2 = OUT_DIR / "missing_report_stage2_round2.json"
LOCK_PATH_round2 = OUT_DIR / ".results_stage2_round2.lock"


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
def read_json_auto(path: Path) -> List[Dict]:
    """
    同时支持：
    - JSON：文件是一个数组 [] 或对象 {}
    - JSONL：每行一个 json object
    """
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    # 尝试当作 JSON（数组/对象）
    if txt[0] in "[{":
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                # 兼容 {"items":[...]} 这种结构
                if "items" in obj and isinstance(obj["items"], list):
                    return obj["items"]
                return [obj]
        except Exception:
            pass

    # 退化为 JSONL
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def write_json(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def read_txt_index(txt_path: Path) -> Dict[str, str]:
    """
    TXT 格式（含表头）:
    item_id\tstage2
    """
    idx: Dict[str, str] = {}
    if not txt_path.exists():
        return idx
    with txt_path.open("r", encoding="utf-8") as f:
        header = True
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if header:
                header = False
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            iid, s2 = parts[0].strip(), parts[1].strip()
            if iid:
                idx[iid] = s2 or ""
    return idx


def write_txt_from_map(txt_path: Path, data_map: Dict[str, str]):
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("item_id\tstage2\n")
        for iid in sorted(data_map.keys()):
            f.write(f"{iid}\t{data_map[iid]}\n")


def export_json_from_txt(txt_path: Path, json_path: Path):
    idx = read_txt_index(txt_path)
    rows = [{"item_id": iid, "stage2": (lab or None)} for iid, lab in sorted(idx.items())]
    write_json(json_path, rows)


# ---------- 预处理 ----------
def normalize_label(text: str) -> str:
    if not text:
        return "未知"
    t = text.strip().replace("：", ":").replace(" ", "")
    aliases = {
        "开心": "快乐", "喜悦": "快乐", "高兴": "快乐",
        "满足感": "宁静", "安宁": "宁静", "平静": "宁静",
        "敬佩": "敬畏", "崇敬": "敬畏", "震撼": "敬畏", "惊叹": "敬畏",
        "害怕": "恐惧", "恐怖": "恐惧",
        "愤慨": "愤怒", "生气": "愤怒",
        "厌烦": "厌恶", "恶心": "厌恶",
        "悲痛": "悲伤", "忧伤": "悲伤",
        "惊讶": "惊奇", "诧异": "惊奇", "意外": "惊奇"
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


def load_resize_to_data_url(image_path: str, max_side: int = 1024, quality: int = 85) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    min_side = min(w, h)
    if min_side > max_side:
        scale = max_side / float(min_side)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return pil_to_data_url(img, quality=quality)


# ---------- Prompt ----------
def prompt_stage2_text() -> str:
    return f"""请模拟人在持续注视这幅艺术作品后的深层情感体验。
仅根据图像内容，从以下标签中选择 1 个：{LABELS}
输出格式：仅输出标签本身（不要解释）。"""


def call_vision(client: OpenAI, model: str, instruction: str, image_data_url: str,
                request_timeout: float = 30.0) -> str:
    content_blocks = [
        {"type": "input_text", "text": instruction},
        {"type": "input_image", "image_url": image_data_url},
    ]
    resp = client.with_options(timeout=request_timeout).responses.create(
        model=model, input=[{"role": "user", "content": content_blocks}]
    )
    return (resp.output_text or "").strip()


def worker_stage2(item: Dict, client: OpenAI, model: str,
                  max_side: int, quality: int,
                  rate_delay: float,
                  request_timeout: float = 30.0) -> Dict:
    iid = item.get("item_id")
    path_raw = item.get("path_raw")
    data_url = load_resize_to_data_url(path_raw, max_side=max_side, quality=quality)
    out = call_vision(client, model, prompt_stage2_text(), data_url, request_timeout=request_timeout)
    s2 = normalize_label(out)
    if rate_delay > 0:
        time.sleep(rate_delay)
    return {"item_id": iid, "stage2": s2}


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Stage2 labeling: TXT落盘→JSON→诊断（可断点续跑）")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--rate_delay", type=float, default=0.0)
    ap.add_argument("--request_timeout", type=float, default=30.0)
    ap.add_argument("--future_timeout", type=float, default=120.0)
    ap.add_argument("--skip_on_timeout", action="store_true")
    ap.add_argument("--flush_every", type=int, default=20)
    return ap.parse_args()


def run_one_dataset(
    items: List[Dict],
    txt_out: Path,
    json_out: Path,
    report_path: Path,
    lock_path: Path,
    args,
    dataset_name: str
):
    all_ids: Set[str] = {it.get("item_id") for it in items if it.get("item_id")}
    print(f"[index:{dataset_name}] stage2 输入条数: {len(items)} | unique ids: {len(all_ids)}")

    existing = read_txt_index(txt_out)
    done_ids = {iid for iid, lab in existing.items() if lab.strip()}

    todo = []
    for it in items:
        iid = it.get("item_id")
        if not iid or not it.get("path_raw"):
            continue
        if iid in done_ids:
            continue
        todo.append(it)

    print(f"[todo:{dataset_name}] 待处理: {len(todo)} | 已完成: {len(done_ids)}")

    if not _OPENAI_OK:
        print("[warn] 未安装 openai 或导入失败，本脚本不会调用模型。")
        return

    client = OpenAI()
    new_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [
            ex.submit(
                worker_stage2, it, client, args.model,
                args.max_side, args.quality,
                args.rate_delay, args.request_timeout
            )
            for it in todo
        ]

        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=args.future_timeout)
                iid, s2 = res["item_id"], res["stage2"]
                existing[iid] = s2
                new_count += 1
                print(f"[stage2:{dataset_name}] {iid} -> {s2}")

                if new_count % max(1, args.flush_every) == 0:
                    _acquire_lock(lock_path)
                    try:
                        write_txt_from_map(txt_out, existing)
                    finally:
                        _release_lock(lock_path)
                    print(f"[flush:{dataset_name}] 已落盘 {new_count} 条到 {txt_out}")

            except FuturesTimeoutError:
                if args.skip_on_timeout:
                    fut.cancel()
                    print(f"[timeout:{dataset_name}] 某样本处理超时，已跳过（下次会继续补）。")
                    continue
                raise
            except Exception as e:
                print(f"[error:{dataset_name}] worker: {e}")

    _acquire_lock(lock_path)
    try:
        write_txt_from_map(txt_out, existing)
    finally:
        _release_lock(lock_path)

    export_json_from_txt(txt_out, json_out)

    final_idx = read_txt_index(txt_out)
    missing = sorted([iid for iid in all_ids if not final_idx.get(iid, "").strip()])

    report = {
        "summary": {
            "input_count": len(items),
            "input_unique_ids": len(all_ids),
            "done_count": len([1 for iid in all_ids if final_idx.get(iid, "").strip()]),
            "missing_count": len(missing),
        },
        "missing_ids": missing
    }
    write_json(report_path, report)

    print(f"[ok:{dataset_name}] TXT: {txt_out}")
    print(f"[ok:{dataset_name}] JSON: {json_out}")
    print(f"[ok:{dataset_name}] REPORT: {report_path}")


def main():
    args = parse_args()

    items = read_json_auto(INPUT_PATH)
    items_round2 = read_json_auto(INPUT_PATH_round2)

    if args.skip > 0:
        items = items[args.skip:]
        items_round2 = items_round2[args.skip:]
    if args.limit and args.limit > 0:
        items = items[:args.limit]
        items_round2 = items_round2[:args.limit]

    print("[debug] INPUT_PATH =", INPUT_PATH, "exists =", INPUT_PATH.exists(), "loaded =", len(items))
    print("[debug] INPUT_PATH_round2 =", INPUT_PATH_round2, "exists =", INPUT_PATH_round2.exists(), "loaded =", len(items_round2))

    run_one_dataset(
        items=items,
        txt_out=TXT_OUT,
        json_out=JSON_OUT,
        report_path=REPORT_PATH,
        lock_path=LOCK_PATH,
        args=args,
        dataset_name="base"
    )

    run_one_dataset(
        items=items_round2,
        txt_out=TXT_OUT_round2,
        json_out=JSON_OUT_round2,
        report_path=REPORT_PATH_round2,
        lock_path=LOCK_PATH_round2,
        args=args,
        dataset_name="round2"
    )

    print("[done] stage2 两套数据均完成。")


if __name__ == "__main__":
    main()
