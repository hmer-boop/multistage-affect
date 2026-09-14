# 4_run describe.py
# 读取 dataset/metadata/items_min.jsonl（item_id, path_raw）
# 先批量生成 caption_zh -> items_with_captions.jsonl
# 然后把 caption_zh 按 item_id 合并回 dataset/metadata/items.jsonl -> items_merged.jsonl
# 可选：--overwrite 直接覆盖 items.jsonl

import os
import io
import json
import base64
import time
import argparse
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from dotenv import load_dotenv

# ================= 基本路径 =================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = next(p for p in SCRIPT_DIR.parents if (p / "requirements.txt").exists() and (p / "src").exists())
META_DIR = BASE_DIR / "dataset" / "metadata"
IN_MIN_JSONL   = META_DIR / "items_min.jsonl"               # 输入（最小字段）
CAP_OUT_JSONL  = META_DIR / "items_with_captions.jsonl"     # 中间输出（新增 caption_zh）
ITEMS_JSONL    = META_DIR / "items.jsonl"                   # 原始 items
MERGED_JSONL   = META_DIR / "items_merged.jsonl"            # 合并后的输出（默认不覆盖原始）

# ================= 模型与提示 =================
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SLEEP_BETWEEN = 0.1

PROMPT = (
    "你现在充当“图像内容记录员”。请只依据画面可见信息进行客观描述。\n"
    "输出格式：只输出中文画面描述，50–70字；不加标题、引号、项目符号或额外说明。\n"
    "写作要点：\n"
    "1-整体与结构：交代场景类型（室内/室外/展陈/自然等）、视角（平视/俯视/仰视）、构图关系（对称/三分/对角线/留白分布），可见时点出前景/中景/背景。\n"
    "2-具象场景（若存在）：明确可计数要素（人物/动物/器物/建筑等的数量）、位置关系（左/中/右、上/下、远/近）、可见动作/交互与显著细节。\n"
    "3-抽象场景（若存在）：描述形状、线条、色块、笔触、纹理，以及大小、方向性（水平/垂直/斜向/弧线/放射/旋转）、叠加与重复（覆盖/半透明/层叠/网格/条带/点状/交叉）。不要把抽象元素命名为具体物体。\n"
    "4-不做任何推测：不写情绪、故事、身份关系、年龄职业；不使用“可能/似乎/仿佛”等不确定表述。\n"
    "5-看不清的内容以“无法辨识”表述，但不要编造。\n"
    "6-严格按照我要求的中文字数，不可以超出范围。\n\n"
    "⚠️ 严禁输出任何情绪性、氛围性或心理感受类词语（如：宁静、静谧、祥和、紧张、压抑、喜悦、悲伤、氛围、感觉、气息、魅力等）。\n"
    "⚠️ 如果生成内容中包含此类词语，请立即删除，不要输出。"
)

# ================= 工具函数 =================
def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def to_jpeg_data_url(img_path: Path) -> str:
    im = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# ================= 模型调用（带重试） =================
class TransientError(Exception):
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(TransientError),
)
def call_model_with_image(data_url: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.2,
            max_tokens=180,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        m = str(e).lower()
        if any(k in m for k in ["rate", "timeout", "temporar", "overloaded", "connection"]):
            raise TransientError(e)
        raise

# ================= 主流程 =================
def generate_captions(max_items: int):
    """从 items_min.jsonl 读取，生成 caption_zh 到 items_with_captions.jsonl；支持断点续跑与条数限制。"""
    in_rows = load_jsonl(IN_MIN_JSONL)
    if not in_rows:
        print(f"未找到或为空：{IN_MIN_JSONL.resolve()}")
        return

    # 已完成集（断点续跑）
    done_ids = set()
    if CAP_OUT_JSONL.exists():
        for r in load_jsonl(CAP_OUT_JSONL):
            rid = r.get("item_id")
            if rid:
                done_ids.add(rid)

    processed_this_run = 0

    pbar = tqdm(in_rows, desc="Generating captions", unit="img")
    for row in pbar:
        if max_items != -1 and processed_this_run >= max_items:
            break

        item_id = row.get("item_id")
        path_raw = row.get("path_raw")
        if not item_id or not path_raw:
            continue
        if item_id in done_ids:
            continue

        img_path = Path(path_raw)
        out_row = {"item_id": item_id, "path_raw": path_raw}

        if not img_path.exists():
            out_row["caption_zh"] = ""
            out_row["error"] = f"Image not found: {img_path}"
            append_jsonl(CAP_OUT_JSONL, out_row)
            done_ids.add(item_id)
            processed_this_run += 1
            continue

        try:
            data_url = to_jpeg_data_url(img_path)
            content = call_model_with_image(data_url)
            out_row["caption_zh"] = content
        except Exception as e:
            out_row["caption_zh"] = ""
            out_row["error"] = str(e)

        append_jsonl(CAP_OUT_JSONL, out_row)
        done_ids.add(item_id)
        processed_this_run += 1
        time.sleep(SLEEP_BETWEEN)

    print(f"描述生成完成：本轮新增 {processed_this_run} 条。输出 -> {CAP_OUT_JSONL.resolve()}")

def merge_back(overwrite: bool = False):
    """把 items_with_captions.jsonl 的 caption_zh 合并回 items.jsonl（按 item_id）。"""
    base_rows = load_jsonl(ITEMS_JSONL)
    cap_rows = load_jsonl(CAP_OUT_JSONL)

    if not base_rows:
        print(f"未找到或为空：{ITEMS_JSONL.resolve()}")
        return
    if not cap_rows:
        print(f"未找到或为空：{CAP_OUT_JSONL.resolve()}")
        return

    # 建索引：item_id -> caption_zh
    cap_map: Dict[str, Dict[str, str]] = {}
    for r in cap_rows:
        iid = r.get("item_id")
        if not iid:
            continue
        cap_map[iid] = {
            "caption_zh": r.get("caption_zh", "")
        }

    merged: List[dict] = []
    hit, miss = 0, 0
    for r in base_rows:
        iid = r.get("item_id")
        if iid and iid in cap_map:
            r["caption_zh"] = cap_map[iid]["caption_zh"]
            hit += 1
        else:
            miss += 1
        merged.append(r)

    if overwrite:
        write_jsonl(ITEMS_JSONL, merged)
        print(f"合并完成：覆盖写回 {ITEMS_JSONL.resolve()}（命中 {hit}，未命中 {miss}）。")
    else:
        write_jsonl(MERGED_JSONL, merged)
        print(f"合并完成：输出到 {MERGED_JSONL.resolve()}（命中 {hit}，未命中 {miss}）。")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Chinese captions and merge back to items.jsonl")
    parser.add_argument(
        "--max",
        type=int,
        default=-1,
        help="本轮最多处理的图片数；-1 表示全量"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="把合并结果直接覆盖写回 items.jsonl（谨慎使用）；不加该参数则写 items_merged.jsonl"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    generate_captions(max_items=args.max)
    merge_back(overwrite=args.overwrite)

if __name__ == "__main__":
    main()
