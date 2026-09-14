# -*- coding: utf-8 -*-
# LLaVA 4B Instruct baseline version converted from the user's InternVL scripts.
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

try:
    from huggingface_hub import InferenceClient
    _HF_OK = True
except Exception:
    _HF_OK = False


LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]

# ========== 这里控制测试量 ==========
# 3: 只测试 base 的前 3 张
# 0 或 -1: 跑全量（base + 2round）
TEST_LIMIT = 0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

META_DIR = PROJECT_ROOT / "dataset" / "metadata"


def pick_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


INPUT_PATH = META_DIR / "items_stage3.json"

# 自动兼容两种命名：2round / round2
INPUT_PATH_2ROUND = pick_existing_path([
    META_DIR / "items_stage3_2round.json",
    META_DIR / "items_stage3_round2.json",
])

OUT_DIR = SCRIPT_DIR / "result llava_stage3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_OUT = OUT_DIR / "results_stage3_llava4b.txt"
JSON_OUT = OUT_DIR / "results_stage3_llava4b.json"
REPORT_PATH = OUT_DIR / "missing_report_stage3_llava4b.json"
LOCK_PATH = OUT_DIR / ".results_stage3_llava4b.lock"

TXT_OUT_2ROUND = OUT_DIR / "results_stage3_llava4b_2round.txt"
JSON_OUT_2ROUND = OUT_DIR / "results_stage3_llava4b_2round.json"
REPORT_PATH_2ROUND = OUT_DIR / "missing_report_stage3_llava4b_2round.json"
LOCK_PATH_2ROUND = OUT_DIR / ".results_stage3_llava4b_2round.lock"

MAX_WORKERS = 1
RATE_DELAY = 0.0
REQUEST_TIMEOUT = 120.0
FUTURE_TIMEOUT = 180.0
SKIP_ON_TIMEOUT = True
FLUSH_EVERY = 20

# 为了省钱：一旦出现接口配置错误/404/认证错误，立即停，不继续烧后面的图
FAIL_FAST_ON_ERROR = True

# ========== HF dataset 配置 ==========
HF_DATASET_REPO = "hmer123/affectecom"
HF_DATASET_REV = "main"

# 本地 cloud 图目录 / manifest
LOCAL_CLOUD_IMAGE_DIR = PROJECT_ROOT / "dataset_hf_upload" / "images_cloud"
CLOUD_MANIFEST_PATH = PROJECT_ROOT / "dataset_hf_upload" / "metadata_cloud" / "images_cloud_manifest.json"


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
def read_json_auto(path: Optional[Path]):
    if path is None or not path.exists():
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


def write_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def read_txt_index(txt_path: Path) -> Dict[str, str]:
    """
    TXT 格式（含表头）:
    item_id\tstage3
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
            iid, s3 = parts[0].strip(), parts[1].strip()
            if iid:
                idx[iid] = s3 or ""
    return idx


def write_txt_from_map(txt_path: Path, data_map: Dict[str, str]):
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("item_id\tstage3\n")
        for iid in sorted(data_map.keys()):
            f.write(f"{iid}\t{data_map[iid]}\n")


def export_json_from_txt(txt_path: Path, json_path: Path):
    idx = read_txt_index(txt_path)
    rows = [{"item_id": iid, "stage3": (lab or None)} for iid, lab in sorted(idx.items())]
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


# ---------- 路径 / manifest / URL ----------
def resolve_image_path(image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _looks_like_image_path(x: Any) -> bool:
    if not isinstance(x, str):
        return False
    s = x.lower()
    return s.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))


def _extract_rel_from_string(s: str, cloud_root: Path) -> Optional[str]:
    s = str(s).replace("\\", "/").strip()
    if not s:
        return None

    marker = "/images_cloud/"
    if marker in s:
        return s.split(marker, 1)[1]

    if s.startswith("images_cloud/"):
        return s[len("images_cloud/"):]

    try:
        p = Path(s).resolve()
        return p.relative_to(cloud_root.resolve()).as_posix()
    except Exception:
        pass

    if _looks_like_image_path(s):
        return s.lstrip("/")

    return None


def _search_rel_in_obj(obj: Any, cloud_root: Path) -> Optional[str]:
    """
    递归地在任意 JSON 对象里搜索一个可用的 cloud 相对路径
    """
    if obj is None:
        return None

    if isinstance(obj, str):
        return _extract_rel_from_string(obj, cloud_root)

    if isinstance(obj, list):
        for x in obj:
            rel = _search_rel_in_obj(x, cloud_root)
            if rel:
                return rel
        return None

    if isinstance(obj, dict):
        preferred_keys = [
            "dst", "path_cloud", "cloud_path", "output_path",
            "output", "saved_path", "target", "target_path", "image_cloud"
        ]
        for k in preferred_keys:
            if k in obj:
                rel = _search_rel_in_obj(obj[k], cloud_root)
                if rel:
                    return rel

        for v in obj.values():
            rel = _search_rel_in_obj(v, cloud_root)
            if rel:
                return rel
        return None

    return None


def build_raw_to_cloud_repo_path(manifest_path: Path, local_cloud_dir: Path) -> Dict[str, str]:
    """
    兼容两类 manifest:
    A. list[dict]
    B. dict[raw_path] = anything
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    mapping: Dict[str, str] = {}
    cloud_root = local_cloud_dir.resolve()

    # ---------- 情况 A：manifest 是 dict，且 key 就是 raw_path ----------
    if isinstance(obj, dict):
        for raw_path, payload in obj.items():
            if not _looks_like_image_path(raw_path):
                continue

            src_abs = str(resolve_image_path(raw_path))
            rel = _search_rel_in_obj(payload, cloud_root)

            if rel:
                mapping[src_abs] = f"images_cloud/{rel}"

    # ---------- 情况 B：manifest 是 list[dict] ----------
    elif isinstance(obj, list):
        candidate_src_keys = ["src", "path_raw", "raw_path", "input_path", "path"]

        for row in obj:
            if not isinstance(row, dict):
                continue

            src = None
            for k in candidate_src_keys:
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    src = v
                    break

            if not src:
                continue

            src_abs = str(resolve_image_path(src))
            rel = _search_rel_in_obj(row, cloud_root)

            if rel:
                mapping[src_abs] = f"images_cloud/{rel}"

    if not mapping:
        raise RuntimeError(
            f"manifest 已读取，但没有构建出任何 raw->cloud 映射: {manifest_path}\n"
            f"manifest 顶层类型: {type(obj).__name__}"
        )

    return mapping


RAW_TO_CLOUD_REPO_PATH = build_raw_to_cloud_repo_path(
    CLOUD_MANIFEST_PATH,
    LOCAL_CLOUD_IMAGE_DIR
)


def raw_path_to_hf_url(path_raw: str) -> str:
    src_abs = str(resolve_image_path(path_raw))
    repo_rel = RAW_TO_CLOUD_REPO_PATH.get(src_abs)

    if not repo_rel:
        raise KeyError(f"未在 manifest 中找到云端映射: {src_abs}")

    return f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/{HF_DATASET_REV}/{repo_rel}"


def precheck_cloud_urls(items: List[Dict], check_n: int = 3):
    sample = items[:min(len(items), check_n)]
    for it in sample:
        iid = it.get("item_id")
        path_raw = it.get("path_raw")
        if not path_raw:
            raise RuntimeError(f"样本缺少 path_raw: {iid}")
        url = raw_path_to_hf_url(path_raw)
        print(f"[debug] cloud url sample | {iid} -> {url}")


def pick_text_value(item: Dict, candidates: List[str]) -> str:
    for k in candidates:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ---------- Prompt ----------
def prompt_stage3_text() -> str:
    return f"""请结合图像内容与给定英文标题/英文描述，预测该作品的情感。
从以下标签中选择 1 个：{LABELS}
输出格式：仅输出标签本身（不要解释）禁止回答未知。"""


# ---------- HF Endpoint ----------
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


def call_vision(
    client: "InferenceClient",
    instruction: str,
    image_url: str,
    title_en: Optional[str] = None,
    caption_en: Optional[str] = None
) -> str:
    model_id = os.getenv("HF_MODEL_ID")
    if not model_id:
        raise RuntimeError("缺少环境变量 HF_MODEL_ID")

    text_context = f"Title: {title_en or ''}\nCaption: {caption_en or ''}"

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "text", "text": text_context},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=32,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def worker_stage3(item: Dict, client: "InferenceClient",
                  rate_delay: float) -> Dict:
    iid = item.get("item_id")
    path_raw = item.get("path_raw")

    # 更鲁棒地取英文标题 / 描述
    title_en = pick_text_value(item, ["title_en", "title", "title_english"])
    caption_en = pick_text_value(item, ["caption_en", "caption", "description_en", "description"])

    image_url = raw_path_to_hf_url(path_raw)
    out = call_vision(
        client=client,
        instruction=prompt_stage3_text(),
        image_url=image_url,
        title_en=title_en,
        caption_en=caption_en,
    )
    s3 = normalize_label(out)

    if rate_delay > 0:
        time.sleep(rate_delay)

    return {"item_id": iid, "stage3": s3}


def run_one_dataset(
    items: List[Dict],
    txt_out: Path,
    json_out: Path,
    report_path: Path,
    lock_path: Path,
    dataset_name: str
):
    all_ids: Set[str] = {it.get("item_id") for it in items if it.get("item_id")}
    print(f"[index:{dataset_name}] stage3 输入条数: {len(items)} | unique ids: {len(all_ids)}")

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
        raise RuntimeError("未安装 huggingface_hub 或导入失败，本脚本不会调用模型。")

    precheck_cloud_urls(todo, check_n=min(3, len(todo)))

    client = build_client()
    print(f"[debug:{dataset_name}] HF_ENDPOINT_URL = {os.getenv('HF_ENDPOINT_URL')}")
    print(f"[debug:{dataset_name}] HF_MODEL_ID = {os.getenv('HF_MODEL_ID')}")
    print(f"[debug:{dataset_name}] HF_DATASET_REPO = {HF_DATASET_REPO}")

    new_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(
                worker_stage3, it, client,
                RATE_DELAY
            )
            for it in todo
        ]

        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=FUTURE_TIMEOUT)
                iid, s3 = res["item_id"], res["stage3"]
                existing[iid] = s3
                new_count += 1
                print(f"[stage3:{dataset_name}] {iid} -> {s3}")

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
                    if FAIL_FAST_ON_ERROR:
                        raise RuntimeError(f"[{dataset_name}] Stage3 测试阶段发生超时，已主动停止，避免继续烧钱。")
                    continue
                raise
            except Exception as e:
                print(f"[error:{dataset_name}] worker: {e}")
                if FAIL_FAST_ON_ERROR:
                    raise RuntimeError(f"[{dataset_name}] Stage3 在首个错误处已中止，避免继续烧钱。原始错误: {e}")

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
    items_2round = read_json_auto(INPUT_PATH_2ROUND)

    print("[debug] INPUT_PATH =", INPUT_PATH, "exists =", INPUT_PATH.exists(), "loaded =", len(items))
    print("[debug] INPUT_PATH_2ROUND =", INPUT_PATH_2ROUND, "exists =", (INPUT_PATH_2ROUND.exists() if INPUT_PATH_2ROUND else False), "loaded =", len(items_2round))
    print("[debug] CLOUD_MANIFEST_PATH =", CLOUD_MANIFEST_PATH, "exists =", CLOUD_MANIFEST_PATH.exists())
    print("[debug] RAW_TO_CLOUD_REPO_PATH size =", len(RAW_TO_CLOUD_REPO_PATH))

    if TEST_LIMIT > 0:
        items = items[:TEST_LIMIT]
        print(f"[test] 当前为测试模式，仅跑 stage3 base 前 {TEST_LIMIT} 张。将不会跑 2round。")

        run_one_dataset(
            items=items,
            txt_out=TXT_OUT,
            json_out=JSON_OUT,
            report_path=REPORT_PATH,
            lock_path=LOCK_PATH,
            dataset_name="base"
        )
        print("[done] stage3 base 测试完成。")
        return

    run_one_dataset(
        items=items,
        txt_out=TXT_OUT,
        json_out=JSON_OUT,
        report_path=REPORT_PATH,
        lock_path=LOCK_PATH,
        dataset_name="base"
    )

    if INPUT_PATH_2ROUND and INPUT_PATH_2ROUND.exists():
        run_one_dataset(
            items=items_2round,
            txt_out=TXT_OUT_2ROUND,
            json_out=JSON_OUT_2ROUND,
            report_path=REPORT_PATH_2ROUND,
            lock_path=LOCK_PATH_2ROUND,
            dataset_name="2round"
        )
    else:
        print("[skip:2round] 未找到 stage3 round2 文件，已跳过。")

    print("[done] stage3 两套数据处理结束。")


if __name__ == "__main__":
    main()