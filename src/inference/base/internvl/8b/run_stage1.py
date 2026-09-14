import os
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

from PIL import Image

try:
    from huggingface_hub import InferenceClient
    _HF_OK = True
except Exception:
    _HF_OK = False


LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]

# ========== 这里控制测试量 ==========
# 3: 只测试 base 的前 3 张
# 0 或 -1: 跑全量（base + round2）
TEST_LIMIT = 0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists() and (p / "src").exists())

META_DIR = PROJECT_ROOT / "dataset" / "metadata"

INPUT_PATH = META_DIR / "items_stage1_2.json"
INPUT_PATH_ROUND2 = META_DIR / "items_stage1_2_round2.json"

OUT_DIR = SCRIPT_DIR / "result internvl_stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_OUT = OUT_DIR / "results_stage1_internvl8b.txt"
JSON_OUT = OUT_DIR / "results_stage1_internvl8b.json"
REPORT_PATH = OUT_DIR / "missing_report_stage1_internvl8b.json"
LOCK_PATH = OUT_DIR / ".results_stage1_internvl8b.lock"

TXT_OUT_ROUND2 = OUT_DIR / "results_stage1_internvl8b_round2.txt"
JSON_OUT_ROUND2 = OUT_DIR / "results_stage1_internvl8b_round2.json"
REPORT_PATH_ROUND2 = OUT_DIR / "missing_report_stage1_internv8bl_round2.json"
LOCK_PATH_ROUND2 = OUT_DIR / ".results_stage1_internvl8b_round2.lock"

MAX_SIDE = 1024
JPEG_QUALITY = 85
MAX_WORKERS = 1          # 第一次先单线程稳一点
RATE_DELAY = 0.0
REQUEST_TIMEOUT = 120.0
FUTURE_TIMEOUT = 180.0
SKIP_ON_TIMEOUT = True
FLUSH_EVERY = 1          # 第一次测试每条都落盘


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
    """
    TXT 格式（含表头）:
    item_id\tstage1
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
            iid, s1 = parts[0].strip(), parts[1].strip()
            if iid:
                idx[iid] = s1 or ""
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

def resolve_image_path(image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()

def load_resize_to_data_url(image_path: str, max_side: int = 1024, quality: int = 85) -> str:
    real_path = resolve_image_path(image_path)
    if not real_path.exists():
        raise FileNotFoundError(f"图片不存在: {real_path}")

    img = Image.open(real_path).convert("RGB")
    w, h = img.size
    min_side = min(w, h)
    if min_side > max_side:
        scale = max_side / float(min_side)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return pil_to_data_url(img, quality=quality)
# ---------- Prompt ----------
def prompt_stage1_text() -> str:
    return f"""请模拟人在快速浏览这幅艺术作品时的即时直觉反应。
仅根据图像内容，从以下标签中选择 1 个：{LABELS}
输出格式：仅输出标签本身（不要解释）。"""


# ---------- HF Endpoint ----------
def build_client() -> "InferenceClient":
    endpoint_url = os.getenv("HF_ENDPOINT_URL")
    hf_token = os.getenv("HF_TOKEN")

    if not endpoint_url:
        raise RuntimeError("缺少环境变量 HF_ENDPOINT_URL")
    if not hf_token:
        raise RuntimeError("缺少环境变量 HF_TOKEN")

    client = InferenceClient(
        base_url=endpoint_url,
        token=hf_token,
        timeout=REQUEST_TIMEOUT,
    )
    return client

def call_vision(client: "InferenceClient", instruction: str, image_data_url: str) -> str:
    model_id = os.getenv("HF_MODEL_ID")

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        max_tokens=32,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()

def worker_stage1(item: Dict, client: "InferenceClient",
                  max_side: int, quality: int,
                  rate_delay: float) -> Dict:
    iid = item.get("item_id")
    path_raw = item.get("path_raw")
    data_url = load_resize_to_data_url(path_raw, max_side=max_side, quality=quality)
    out = call_vision(client, prompt_stage1_text(), data_url)
    s1 = normalize_label(out)
    if rate_delay > 0:
        time.sleep(rate_delay)
    return {"item_id": iid, "stage1": s1}


def run_one_dataset(
    items: List[Dict],
    txt_out: Path,
    json_out: Path,
    report_path: Path,
    lock_path: Path,
    dataset_name: str
):
    all_ids: Set[str] = {it.get("item_id") for it in items if it.get("item_id")}
    print(f"[index:{dataset_name}] 输入条数: {len(items)} | unique ids: {len(all_ids)}")

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

    if len(todo) == 0:
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

        print(f"[skip:{dataset_name}] 无待处理样本，已直接导出 JSON / REPORT")
        print(f"[ok:{dataset_name}] TXT: {txt_out}")
        print(f"[ok:{dataset_name}] JSON: {json_out}")
        print(f"[ok:{dataset_name}] REPORT: {report_path}")
        return

    if not _HF_OK:
        print("[warn] 未安装 huggingface_hub 或导入失败，本脚本不会调用模型。")
        return

    client = build_client()
    print(f"[debug:{dataset_name}] HF_ENDPOINT_URL = {os.getenv('HF_ENDPOINT_URL')}")
    print(f"[debug:{dataset_name}] HF_MODEL_ID = {os.getenv('HF_MODEL_ID')}")

    new_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(
                worker_stage1, it, client,
                MAX_SIDE, JPEG_QUALITY,
                RATE_DELAY
            )
            for it in todo
        ]

        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=FUTURE_TIMEOUT)
                iid, s1 = res["item_id"], res["stage1"]
                existing[iid] = s1
                new_count += 1
                print(f"[stage1:{dataset_name}] {iid} -> {s1}")

                if new_count % max(1, FLUSH_EVERY) == 0:
                    _acquire_lock(lock_path)
                    try:
                        write_txt_from_map(txt_out, existing)
                    finally:
                        _release_lock(lock_path)
                    print(f"[flush:{dataset_name}] 已落盘 {new_count} 条到 {txt_out}")

            except FuturesTimeoutError:
                if SKIP_ON_TIMEOUT:
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
    items = read_json_auto(INPUT_PATH)
    items_round2 = read_json_auto(INPUT_PATH_ROUND2)

    print("[debug] INPUT_PATH =", INPUT_PATH, "exists =", INPUT_PATH.exists(), "loaded =", len(items))
    print("[debug] INPUT_PATH_ROUND2 =", INPUT_PATH_ROUND2, "exists =", INPUT_PATH_ROUND2.exists(), "loaded =", len(items_round2))

    # 测试模式：只跑 base 前 TEST_LIMIT 张
    if TEST_LIMIT > 0:
        items = items[:TEST_LIMIT]
        print(f"[test] 当前为测试模式，仅跑 base 前 {TEST_LIMIT} 张。将不会跑 round2。")

        run_one_dataset(
            items=items,
            txt_out=TXT_OUT,
            json_out=JSON_OUT,
            report_path=REPORT_PATH,
            lock_path=LOCK_PATH,
            dataset_name="base"
        )
        print("[done] base 测试完成。")
        return

    # 全量模式：跑 base + round2
    run_one_dataset(
        items=items,
        txt_out=TXT_OUT,
        json_out=JSON_OUT,
        report_path=REPORT_PATH,
        lock_path=LOCK_PATH,
        dataset_name="base"
    )

    run_one_dataset(
        items=items_round2,
        txt_out=TXT_OUT_ROUND2,
        json_out=JSON_OUT_ROUND2,
        report_path=REPORT_PATH_ROUND2,
        lock_path=LOCK_PATH_ROUND2,
        dataset_name="round2"
    )

    print("[done] stage1 两套数据均完成。")


if __name__ == "__main__":
    main()