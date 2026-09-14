# -*- coding: utf-8 -*-
"""
8_Gemini25pro_run_label1_fast.py —— Gemini 2.5 Pro Stage1 单阶段标注（并发提速版）

用途：
1. 将原 GPT baseline 的 label1 迁移到 Gemini 2.5 Pro。
2. 输出结构与分 stage 目录一致：当前脚本只写入
   000test_gemini/gemini 2.5 pro/1_baseline/result Gemini_stage1/
3. 支持断点续跑、并发处理、定期落盘、进度打印、round2 独立输出。

输入（项目根目录，不复制）：
- dataset/metadata/items_stage1_2.json
- dataset/metadata/items_stage1_2_round2.json

输出（当前脚本目录下的独立 stage1 文件夹）：
- result Gemini_stage1/results_stage1.txt
- result Gemini_stage1/results_stage1.json
- result Gemini_stage1/missing_report_stage1.json
- result Gemini_stage1/results_stage1_round2.txt
- result Gemini_stage1/results_stage1_round2.json
- result Gemini_stage1/missing_report_stage1_round2.json
"""

import os
import json
import time
import argparse
import threading
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
DEFAULT_MODEL = "gemini-2.5-pro"
RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


# =========================================================
# 路径
# =========================================================
def find_project_root(start_dir: Path) -> Path:
    """从当前脚本目录向上寻找项目根目录（以 dataset/metadata 为标志）。"""
    candidates = [start_dir, *start_dir.parents]
    for p in candidates:
        if (p / "dataset" / "metadata").exists():
            return p
    raise FileNotFoundError(
        "未找到项目根目录：请确认当前脚本位于 PythonProject 内部，且存在 dataset/metadata/"
    )


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(BASE_DIR)
META_DIR = PROJECT_ROOT / "dataset" / "metadata"

INPUT_PATH = META_DIR / "items_stage1_2.json"
INPUT_PATH_ROUND2 = META_DIR / "items_stage1_2_round2.json"

OUT_DIR = BASE_DIR / "result Gemini_stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_OUT = OUT_DIR / "results_stage1.txt"
JSON_OUT = OUT_DIR / "results_stage1.json"
REPORT_PATH = OUT_DIR / "missing_report_stage1.json"
LOCK_PATH = OUT_DIR / ".results_stage1.lock"

TXT_OUT_ROUND2 = OUT_DIR / "results_stage1_round2.txt"
JSON_OUT_ROUND2 = OUT_DIR / "results_stage1_round2.json"
REPORT_PATH_ROUND2 = OUT_DIR / "missing_report_stage1_round2.json"
LOCK_PATH_ROUND2 = OUT_DIR / ".results_stage1_round2.lock"


# =========================================================
# 轻量锁
# =========================================================
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


# =========================================================
# I/O
# =========================================================
def read_json_auto(path: Path) -> List[Dict]:
    if not path.exists():
        return []

    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    if txt[0] in "[{":
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                if "items" in obj and isinstance(obj["items"], list):
                    return obj["items"]
                return [obj]
        except Exception:
            pass

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
    idx: Dict[str, str] = {}
    if not txt_path.exists():
        return idx

    with txt_path.open("r", encoding="utf-8") as f:
        header_seen = False
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            col1, col2 = parts[0].strip(), parts[1].strip()

            if not header_seen:
                header_seen = True
                if col1 == "item_id" and col2 == "stage1":
                    continue

            if col1:
                idx[col1] = col2 or ""

    return idx


def write_txt_from_map(txt_path: Path, data_map: Dict[str, str]):
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("item_id\tstage1\n")
        for iid in sorted(data_map.keys()):
            f.write(f"{iid}\t{data_map[iid]}\n")


def export_json_from_txt(txt_path: Path, json_path: Path):
    idx = read_txt_index(txt_path)
    rows = [{"item_id": iid, "stage1": (lab or None)} for iid, lab in sorted(idx.items())]
    write_json(json_path, rows)


# =========================================================
# 标签归一化（保留，与原 GPT baseline 一致）
# =========================================================
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


# =========================================================
# Prompt
# =========================================================
def prompt_stage1_text() -> str:
    return f"""请模拟人在快速浏览这幅艺术作品时的即时直觉反应。
仅根据图像内容，从以下标签中选择 1 个：{LABELS}
输出格式：仅输出标签本身（不要解释）。"""


# =========================================================
# 图像处理
# =========================================================
def resolve_image_path(raw_img_path: str) -> Path:
    p = Path(raw_img_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def image_to_bytes(img_path: Path, max_side: int = 512, quality: int = 80) -> Tuple[bytes, str]:
    """
    统一转为 JPEG 字节流，便于稳定传给 Gemini。
    按最短边判断是否缩放，逻辑与原 GPT 版保持一致。
    为提速，默认改为 512 + 80。
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    min_side = min(w, h)

    if min_side > max_side:
        scale = max_side / float(min_side)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), RESAMPLE)

    buf = BytesIO()
    # 提速：不再使用 optimize/progressive
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"


# =========================================================
# Gemini 响应解析
# =========================================================
def extract_response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    try:
        candidates = getattr(response, "candidates", None) or []
        parts = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(str(part_text))
        return "\n".join(parts).strip()
    except Exception:
        return ""


# =========================================================
# Gemini client（线程本地，避免多线程共享状态问题）
# =========================================================
_thread_local = threading.local()


def get_gemini_client(api_key: str):
    client = getattr(_thread_local, "client", None)
    if client is None:
        from google import genai
        client = genai.Client(api_key=api_key)
        _thread_local.client = client
    return client


# =========================================================
# worker
# =========================================================
def worker_stage1(item: Dict, args, api_key: str, types_module):
    iid = item.get("item_id")
    raw_img_path = item.get("path_raw")

    if not iid or not raw_img_path:
        raise ValueError("缺少 item_id 或 path_raw")

    img_path = resolve_image_path(raw_img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"图片路径不存在: {img_path}")

    image_bytes, mime_type = image_to_bytes(
        img_path=img_path,
        max_side=args.max_side,
        quality=args.quality,
    )

    client = get_gemini_client(api_key)
    response = client.models.generate_content(
        model=args.model,
        contents=[
            types_module.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt_stage1_text(),
        ],
    )

    raw_result = extract_response_text(response)
    s1 = normalize_label(raw_result)

    if args.rate_delay > 0:
        time.sleep(args.rate_delay)

    return {
        "item_id": iid,
        "raw_result": raw_result,
        "stage1": s1,
    }


# =========================================================
# CLI
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Gemini 2.5 Pro Stage1 labeling with TXT/JSON dual save")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Gemini 模型名，默认 gemini-2.5-pro")
    ap.add_argument("--limit", type=int, default=-1, help="处理条数（-1 表示全量）")
    ap.add_argument("--skip", type=int, default=0, help="跳过前 N 条")
    ap.add_argument("--max_side", type=int, default=512, help="最短边最大尺寸（默认 512）")
    ap.add_argument("--quality", type=int, default=80, help="JPEG 质量（默认 80）")
    ap.add_argument("--rate_delay", type=float, default=0.0, help="每次请求后的延时秒数（默认 0）")
    ap.add_argument("--flush_every", type=int, default=50, help="每 N 条强制落盘 TXT+JSON（默认 50）")
    ap.add_argument("--max_retries", type=int, default=1, help="单样本最大重试次数（默认 1）")
    ap.add_argument("--retry_sleep", type=float, default=1.0, help="失败后的重试等待秒数（默认 1.0）")
    ap.add_argument("--max_workers", type=int, default=2, help="并发线程数（默认 4）")
    return ap.parse_args()


# =========================================================
# 核心
# =========================================================
def run_one_dataset(
    items: List[Dict],
    txt_out: Path,
    json_out: Path,
    report_path: Path,
    lock_path: Path,
    args,
    dataset_name: str,
    api_key: str,
    types_module,
):
    export_json_from_txt(txt_out, json_out)

    all_ids: Set[str] = {it.get("item_id") for it in items if it.get("item_id")}
    print(f"[index:{dataset_name}] 输入条数: {len(items)} | unique ids: {len(all_ids)}")

    existing = read_txt_index(txt_out)
    done_ids = {iid for iid, lab in existing.items() if lab.strip()}

    todo = []
    for it in items:
        iid = it.get("item_id")
        raw_img_path = it.get("path_raw")
        if not iid or not raw_img_path:
            continue
        if iid in done_ids:
            continue
        todo.append(it)

    print(f"[todo:{dataset_name}] 待处理: {len(todo)} | 已完成: {len(done_ids)} | max_workers: {args.max_workers}")

    if not todo:
        export_json_from_txt(txt_out, json_out)
    else:
        new_count = 0
        total_todo = len(todo)
        flush_mark = 0

        def submit_with_retry(executor, item):
            return executor.submit(worker_stage1, item, args, api_key, types_module)

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            future_to_ctx = {}
            for idx, item in enumerate(todo, start=1):
                fut = submit_with_retry(ex, item)
                future_to_ctx[fut] = {
                    "item": item,
                    "attempt": 1,
                    "order_idx": idx,
                }

            while future_to_ctx:
                for fut in as_completed(list(future_to_ctx.keys())):
                    ctx = future_to_ctx.pop(fut)
                    item = ctx["item"]
                    attempt = ctx["attempt"]
                    order_idx = ctx["order_idx"]
                    iid = item.get("item_id")

                    try:
                        res = fut.result()
                        existing[res["item_id"]] = res["stage1"]
                        new_count += 1

                        print(
                            f"[stage1:{dataset_name}] {new_count}/{total_todo} | {res['item_id']} -> 原始:{res['raw_result']} | 归一化:{res['stage1']}"
                        )

                        if (new_count - flush_mark) >= max(1, args.flush_every):
                            _acquire_lock(lock_path)
                            try:
                                write_txt_from_map(txt_out, existing)
                            finally:
                                _release_lock(lock_path)

                            export_json_from_txt(txt_out, json_out)
                            flush_mark = new_count
                            print(f"[flush:{dataset_name}] 已累计新增 {new_count} 条 -> TXT + JSON")

                    except Exception as e:
                        if attempt < args.max_retries:
                            print(f"[retry:{dataset_name}] {iid} 第 {attempt}/{args.max_retries} 次失败: {e}")
                            if args.retry_sleep > 0:
                                time.sleep(args.retry_sleep)
                            new_fut = submit_with_retry(ex, item)
                            future_to_ctx[new_fut] = {
                                "item": item,
                                "attempt": attempt + 1,
                                "order_idx": order_idx,
                            }
                        else:
                            print(f"[error:{dataset_name}] {iid} 最终失败，保留到下次断点续跑。错误: {e}")
                    break

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
        "missing_ids": missing,
    }
    write_json(report_path, report)

    print(f"[ok:{dataset_name}] TXT: {txt_out}")
    print(f"[ok:{dataset_name}] JSON: {json_out}")
    print(f"[ok:{dataset_name}] REPORT: {report_path}")


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 错误：未检测到 GEMINI_API_KEY 或 GOOGLE_API_KEY")
        print("请确认已在 PyCharm 的环境变量或系统环境变量中设置。")
        return

    try:
        from google.genai import types
    except ImportError:
        print("❌ 错误：请先安装 google-genai")
        print("执行：pip install -U google-genai")
        return

    items = read_json_auto(INPUT_PATH)
    items_round2 = read_json_auto(INPUT_PATH_ROUND2)

    if args.skip > 0:
        items = items[args.skip:]
        items_round2 = items_round2[args.skip:]

    if args.limit and args.limit > 0:
        items = items[:args.limit]
        items_round2 = items_round2[:args.limit]

    print("[debug] PROJECT_ROOT =", PROJECT_ROOT)
    print("[debug] INPUT_PATH =", INPUT_PATH, "exists =", INPUT_PATH.exists(), "loaded =", len(items))
    print(
        "[debug] INPUT_PATH_ROUND2 =",
        INPUT_PATH_ROUND2,
        "exists =",
        INPUT_PATH_ROUND2.exists(),
        "loaded =",
        len(items_round2),
    )
    print("[debug] OUT_DIR =", OUT_DIR)
    print("[debug] MODEL =", args.model)
    print("[debug] max_workers =", args.max_workers)
    print("[debug] max_side =", args.max_side)
    print("[debug] quality =", args.quality)
    print("[debug] max_retries =", args.max_retries)
    print("[debug] flush_every =", args.flush_every)

    export_json_from_txt(TXT_OUT, JSON_OUT)
    export_json_from_txt(TXT_OUT_ROUND2, JSON_OUT_ROUND2)

    run_one_dataset(
        items=items,
        txt_out=TXT_OUT,
        json_out=JSON_OUT,
        report_path=REPORT_PATH,
        lock_path=LOCK_PATH,
        args=args,
        dataset_name="base",
        api_key=api_key,
        types_module=types,
    )

    run_one_dataset(
        items=items_round2,
        txt_out=TXT_OUT_ROUND2,
        json_out=JSON_OUT_ROUND2,
        report_path=REPORT_PATH_ROUND2,
        lock_path=LOCK_PATH_ROUND2,
        args=args,
        dataset_name="round2",
        api_key=api_key,
        types_module=types,
    )

    print("[done] stage1 两套数据均完成。")


if __name__ == "__main__":
    main()
