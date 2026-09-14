import os
import io
import json
import hashlib
import argparse
import re
import shutil
import time
import unicodedata
from pathlib import Path, PureWindowsPath
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ===== 路径配置 =====
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR   = BASE_DIR / "00 add description"              # 源图片目录
DATA_ROOT = BASE_DIR / "dataset"                         # 项目内数据根目录
RAW_DIR   = DATA_ROOT / "images_raw"                     # 清晰图输出目录（JPEG）
META_DIR  = DATA_ROOT / "metadata"                       # 元数据目录
OUT_JSONL = META_DIR / "items.jsonl"                     # 输出 JSONL（追加写）
TRANSLATION_CACHE = META_DIR / "title_translation_cache.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

RAW_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
TITLE_MODEL = os.getenv("OPENAI_TITLE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
SLEEP_BETWEEN_TRANSLATIONS = float(os.getenv("TITLE_TRANSLATION_SLEEP", "0.1"))
TITLE_TRANSLATION_SAVE_EVERY = max(1, int(os.getenv("TITLE_TRANSLATION_SAVE_EVERY", "20")))
TITLE_TRANSLATION_PROGRESS_EVERY = max(1, int(os.getenv("TITLE_TRANSLATION_PROGRESS_EVERY", "5")))

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# ===== 工具函数 =====
def sha12(b: bytes) -> str:
    """对 JPEG 二进制做 SHA1 取前 12 位，作为内容 ID。"""
    return hashlib.sha1(b).hexdigest()[:12]

def to_jpeg_bytes(p: Path) -> bytes:
    """任意图片转为 RGB JPEG，并返回字节流。"""
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()

def walk_images(root: Path):
    """遍历 root 下所有图片文件。"""
    for q in root.rglob("*"):
        if q.is_file() and q.suffix.lower() in IMG_EXTS:
            yield q

def load_existing_rows():
    """加载现有的 items.jsonl，返回记录列表。"""
    rows = []
    if OUT_JSONL.exists():
        with OUT_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    rows.append(row)
                except Exception:
                    continue
        print(f"已加载 {len(rows)} 条现有记录。")
    return rows

def write_jsonl(path: Path, rows):
    """覆盖写回 JSONL。"""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def source_name_from_path(path_text: str) -> str:
    """兼容旧 Windows 路径和当前 macOS 路径，提取源文件名。"""
    path_text = str(path_text or "").strip()
    if not path_text:
        return ""
    if "\\" in path_text:
        return PureWindowsPath(path_text).name
    return Path(path_text).name

def source_name_key(name: str) -> str:
    """文件名匹配键：统一 Unicode 与大小写，避免 macOS 文件名差异。"""
    return unicodedata.normalize("NFC", name or "").casefold()

def has_chinese(text: str) -> bool:
    """粗略判断标题中是否包含中文字符。"""
    return bool(CJK_RE.search(text or ""))

def normalize_title(title: str) -> str:
    """清理从文件名获得的标题。"""
    title = (title or "").strip()
    title = re.sub(r"\s+", " ", title)
    return title

def load_translation_cache():
    if not TRANSLATION_CACHE.exists():
        return {}
    try:
        with TRANSLATION_CACHE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_translation_cache(cache: dict):
    with TRANSLATION_CACHE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def make_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY，无法自动翻译标题。请在 .env 中配置后再运行 Step 1。"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "缺少 openai Python 包，无法自动翻译标题。请先安装 openai，或使用包含 openai 的解释器运行 Step 1。"
        ) from exc
    return OpenAI()

def translate_title(client, cache: dict, title: str, target_lang: str) -> str:
    """将标题翻译为指定目标语言；只返回标题本身。"""
    title = normalize_title(title)
    if not title:
        return ""

    cache_key = f"{target_lang}::{title}"
    if cache_key in cache:
        return cache[cache_key]

    target_label = "简体中文" if target_lang == "zh" else "英文"
    source_label = "中文" if has_chinese(title) else "外文"
    prompt = (
        "你是艺术作品标题翻译助手。请把下面的作品标题从"
        f"{source_label}翻译为{target_label}。\n"
        "要求：只输出翻译后的标题，不要引号、解释、项目符号或额外说明；"
        "保留数字、罗马数字、年份、编号；人名和地名使用通行译名或自然音译；"
        "Untitled 译为“无题”，中文“无题”译为 Untitled。\n\n"
        f"标题：{title}"
    )
    resp = client.chat.completions.create(
        model=TITLE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=120,
    )
    translated = (resp.choices[0].message.content or "").strip()
    translated = translated.strip("\"'“”‘’").strip()
    cache[cache_key] = translated
    time.sleep(SLEEP_BETWEEN_TRANSLATIONS)
    return translated

def bilingual_titles(original_title: str, client, cache: dict):
    """根据原始标题生成 title_zh/title_en，以及原始语言标记。"""
    original_title = normalize_title(original_title)
    if has_chinese(original_title):
        title_zh = original_title
        title_en = translate_title(client, cache, original_title, "en")
        original_lang = "zh"
    else:
        title_en = original_title
        title_zh = translate_title(client, cache, original_title, "zh")
        original_lang = "foreign"
    return title_zh, title_en, original_lang

def infer_original_title(row: dict) -> str:
    """从旧记录中恢复原始标题，用于补齐双语字段。"""
    if row.get("title_original"):
        return normalize_title(row["title_original"])
    if row.get("source_abs_path"):
        return normalize_title(Path(row["source_abs_path"]).stem)
    if row.get("title_zh") and not has_chinese(row.get("title_zh", "")):
        return normalize_title(row["title_zh"])
    if row.get("title_zh"):
        return normalize_title(row["title_zh"])
    if row.get("title_en"):
        return normalize_title(row["title_en"])
    return ""

def ensure_existing_bilingual_titles(rows, client, cache: dict) -> int:
    """给旧 items.jsonl 记录补齐 title_zh/title_en。"""
    fix_targets = []
    for row in rows:
        original_title = infer_original_title(row)
        if not original_title:
            continue

        title_zh = row.get("title_zh", "")
        title_en = row.get("title_en", "")
        needs_fix = (
            not title_zh
            or not title_en
            or not has_chinese(title_zh)
            or has_chinese(title_en)
        )
        if needs_fix:
            fix_targets.append((row, original_title))

    total = len(fix_targets)
    if not total:
        print("旧记录双语标题已完整，无需补齐。")
        return 0

    print(f"开始补齐旧记录双语标题：{total} 条")
    updated = 0
    for idx, (row, original_title) in enumerate(fix_targets, start=1):
        try:
            title_zh, title_en, original_lang = bilingual_titles(original_title, client, cache)
        except Exception:
            save_translation_cache(cache)
            print(
                f"标题翻译中断：已处理 {idx - 1}/{total} 条，"
                f"缓存已保存 -> {TRANSLATION_CACHE.resolve()}"
            )
            raise

        row["title_original"] = original_title
        row["title_original_lang"] = original_lang
        row["title_zh"] = title_zh
        row["title_en"] = title_en
        updated += 1

        if updated % TITLE_TRANSLATION_PROGRESS_EVERY == 0 or updated == total:
            print(f"标题翻译进度：{updated}/{total}")
        if updated % TITLE_TRANSLATION_SAVE_EVERY == 0:
            save_translation_cache(cache)

    return updated

def parse_args():
    parser = argparse.ArgumentParser(description="清理新增图片、生成 item 记录，并为新增标题生成双语字段")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=SRC_DIR,
        help="源图片目录；默认读取项目目录下的 00 add description"
    )
    parser.add_argument(
        "--backfill-existing-titles",
        action="store_true",
        help="补齐旧 items.jsonl 记录里的 title_zh/title_en；默认不补，只处理新增图片"
    )
    parser.add_argument(
        "--no-fast-source-skip",
        action="store_true",
        help="关闭按源文件名快速跳过；仅在同名文件被替换且需要重新入库时使用"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    src_dir = args.src_dir.expanduser()
    if not src_dir.is_absolute():
        src_dir = (BASE_DIR / src_dir).resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"源图片目录不存在：{src_dir}")

    print(f"源图片目录：{src_dir}")
    items = []
    existing_rows = load_existing_rows()
    existing_item_ids = {r.get("item_id") for r in existing_rows if r.get("item_id")}
    existing_source_names = {
        source_name_key(source_name_from_path(r.get("source_abs_path", "")))
        for r in existing_rows
        if source_name_from_path(r.get("source_abs_path", ""))
    }
    cache = load_translation_cache()
    client = None

    if args.backfill_existing_titles:
        client = make_openai_client()
        updated_existing = ensure_existing_bilingual_titles(existing_rows, client, cache)
        if updated_existing:
            backup = OUT_JSONL.with_suffix(".jsonl.before_title_translation.bak")
            shutil.copy2(OUT_JSONL, backup)
            write_jsonl(OUT_JSONL, existing_rows)
            print(f"已补齐旧记录双语标题 {updated_existing} 条；备份：{backup}")
    else:
        print("跳过旧记录双语标题补齐；本次只处理新增图片。")

    source_images = list(walk_images(src_dir))
    print(f"源图片数量：{len(source_images)}")
    if not args.no_fast_source_skip:
        before = len(source_images)
        source_images = [
            src for src in source_images
            if source_name_key(src.name) not in existing_source_names
        ]
        print(f"按源文件名快速跳过：{before - len(source_images)} 张；待处理新增：{len(source_images)} 张")

    for src in source_images:
        try:
            # 1) 转 JPEG 并计算内容哈希 ID
            jpeg = to_jpeg_bytes(src)
            item_id = sha12(jpeg)

            # 如果该 item_id 已经处理过，跳过
            if item_id in existing_item_ids:
                print(f"跳过已处理的图片：{src.name}")
                continue

            # 2) 写入 images_raw
            raw_out = RAW_DIR / f"{item_id}.jpg"
            with open(raw_out, "wb") as f:
                f.write(jpeg)

            # 3) 标题：原文件名若是外文，则翻译成中文；若是中文，则翻译成英文。
            title_original = normalize_title(src.stem)
            if client is None:
                client = make_openai_client()
            title_zh, title_en, title_original_lang = bilingual_titles(title_original, client, cache)

            # 4) 写入记录：同时保留中文与外文标题
            rec = {
                "item_id": item_id,
                "path_raw": str(raw_out.as_posix()),
                "source_abs_path": str(src.resolve()),
                "title_original": title_original,
                "title_original_lang": title_original_lang,
                "title_zh": title_zh,
                "title_en": title_en,
            }
            items.append(rec)
            existing_item_ids.add(item_id)
            print(f"OK {item_id} ← {src.name} | zh: {title_zh} | en: {title_en}")

        except Exception as e:
            print(f"ERR {src} : {e}")

    # 5) 只追加写入新增记录
    if items:
        with OUT_JSONL.open("a", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"\n完成：新增 {len(items)} 条 -> {OUT_JSONL.resolve()}")
    else:
        print("没有新增记录，跳过写入。")

    save_translation_cache(cache)
    print(f"标题翻译缓存：{TRANSLATION_CACHE.resolve()}")

if __name__ == "__main__":
    main()
