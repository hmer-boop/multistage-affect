# step1_build_stage1_cues.py
# pip install opencv-python numpy

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import os
import base64
import re
from openai import OpenAI
from tqdm import tqdm


# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_IDS_PATH = PROJECT_ROOT / "01improvement method" / "method data" / "gold_item_ids.json"
IMAGES_RAW_DIR = PROJECT_ROOT / "dataset" / "images_raw"
ITEMS_MIN_PATH = PROJECT_ROOT / "dataset" / "metadata" / "items_min.jsonl"

OUT_DIR = PROJECT_ROOT / "01improvement method" / "method data"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # 目录本来就有也无所谓

OUT_JSON = OUT_DIR / "stage1_cues.json"  # ✅ 单一 JSON 文件


# =========================
# 枚举集合（用于校验）
# =========================
SPATIAL_ENUM = {"open", "balanced", "crowded", "compressed", "enclosed"}
DYNAMICS_ENUM = {"static", "mild_dynamic", "dynamic", "high_tension", "chaotic"}

# =========================
# VLM 配置
# =========================
VLM_MODEL = "gpt-4o"  #
VLM_TEMPERATURE = 0.0       # 枚举任务建议 0
VLM_MAX_RETRY = 2           # 解析失败时最多重试次数

# =========================
# 断点续跑：读取已有 stage1_cues.json
# =========================
def load_existing_results(out_json: Path) -> Tuple[Dict[str, Dict[str, Any]], set]:
    """
    读取已有输出，返回：
      existing_map: dict[id_str] = record
      done_ids: set(id_str)

    兼容两种格式：
      1) list: [{"id":"123", ...}, ...]   （你当前脚本就是这种）
      2) dict: {"123": {...}, ...}        （以防你之前保存成这种）
    """
    if not out_json.exists():
        return {}, set()

    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ 已有 {out_json} 但读取失败：{e}，将当作空结果重新生成。")
        return {}, set()

    existing_map: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict) and "id" in rec:
                existing_map[str(rec["id"])] = rec
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                existing_map[str(k)] = v
    else:
        print(f"⚠️ 已有 {out_json} 格式不识别（{type(data)}），将当作空结果重新生成。")
        return {}, set()

    done_ids = set(existing_map.keys())
    return existing_map, done_ids


def save_results_as_list(out_json: Path, results_map: Dict[str, Dict[str, Any]]) -> None:
    """统一保存为 list（与你原脚本一致），避免重复："""
    results_list = list(results_map.values())
    out_json.write_text(
        json.dumps(results_list, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# =========================
# 本地三项特征（枚举）
# =========================
def compute_brightness_pattern(gray: np.ndarray) -> str:
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    dark_ratio = float(np.mean(gray < 60))
    bright_ratio = float(np.mean(gray > 195))

    if dark_ratio > 0.45:
        return "dominant_dark_region"
    if bright_ratio > 0.45:
        return "dominant_bright_region"
    if std > 70:
        return "high_contrast"
    if std < 35:
        return "low_contrast"
    if mean < 95:
        return "globally_dark"
    if mean > 160:
        return "globally_bright"
    return "normal"


def compute_color_tone(img_bgr: np.ndarray) -> str:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    s_mean = float(np.mean(s))
    if s_mean < 0.12:
        return "monochrome"
    if s_mean > 0.60:
        return "highly_saturated"
    if s_mean < 0.22:
        return "desaturated"

    mask = (s > 0.15) & (v > 0.15)
    h_mean = float(np.mean(h[mask])) if np.any(mask) else float(np.mean(h))

    if (0 <= h_mean < 45) or (165 <= h_mean <= 179):
        return "warm"
    if 75 <= h_mean < 135:
        return "cool"
    return "neutral"


def compute_visual_complexity(gray: np.ndarray) -> str:
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.mean(edges > 0))

    if edge_ratio < 0.03:
        return "very_simple"
    if edge_ratio < 0.06:
        return "simple"
    if edge_ratio < 0.11:
        return "moderate"
    if edge_ratio < 0.18:
        return "complex"
    return "very_complex"


# =========================
# items_min.jsonl 找图路径（兜底）
# =========================
def load_image_path_from_items_min(item_id: str) -> Optional[str]:
    if not ITEMS_MIN_PATH.exists():
        return None
    with ITEMS_MIN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if str(it.get("id")) == str(item_id):
                return (it.get("image_path")
                        or it.get("image")
                        or it.get("img_path")
                        or it.get("local_path")
                        or it.get("url"))
    return None


def resolve_image_path(item_id: str) -> Optional[Path]:
    # 优先 images_raw/<id>.jpg（也支持 png/jpeg）
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = IMAGES_RAW_DIR / f"{item_id}{ext}"
        if p.exists():
            return p

    # 兜底 items_min.jsonl
    s = load_image_path_from_items_min(item_id)
    if not s:
        return None
    p = Path(s)
    if p.exists():
        return p

    # 如果 items_min.jsonl 里是相对路径或 URL，这里你可自行扩展
    return None


# =========================
# 模型补全两项（占位）
# =========================
def build_prompt_for_spatial_and_dynamics(local_cues: Dict[str, Any]) -> str:
    return f"""
你正在执行 Stage 1（短时观看）的感知补全任务。

你的目标是：基于对图像的直接视觉观察，判断画面的
- 空间感（spatial_pressure）
- 动态感（overall_dynamics）

你只能输出以下两个字段，并且必须从给定枚举中选择：
- spatial_pressure ∈ {sorted(list(SPATIAL_ENUM))}
- overall_dynamics ∈ {sorted(list(DYNAMICS_ENUM))}

=====================
枚举含义说明（仅用于理解，不用于输出）
=====================

spatial_pressure（空间感）：
- open：画面留白明显，空间关系舒展，整体有“呼吸感”
- balanced：元素分布均衡，空间不显拥挤也不空旷
- crowded：画面被大量元素填满，留白很少
- compressed：主要视觉元素贴近画面边界，空间被明显挤压
- enclosed：画面被前景或边界包围，视觉出口受限

overall_dynamics（动态感）：
- static：画面稳定，几乎无运动或张力暗示
- mild_dynamic：存在轻微方向性、姿态变化或节奏感
- dynamic：画面呈现明显的运动趋势或结构张力
- high_tension：运动感或构图张力很强，接近爆发状态
- chaotic：方向多样且无序，整体难以形成稳定结构

=====================
重要约束（必须遵守）
=====================

1) 最终判断必须基于你对图像的直接观察。
2) 下方提供的“本地线索”仅用于轻量校准，不能替代看图，也不能仅由线索推导结论。
3) 不要描述具体物体、人物身份、事件因果、故事背景或文字语义。
4) 不要输出解释性文字、不要输出多余字段、不要使用 Markdown。
5) 只返回严格 JSON。

=====================
本地线索（仅供参考，不可替代看图）
=====================
- brightness_pattern = {local_cues["brightness_pattern"]}
- color_tone = {local_cues["color_tone"]}
- visual_complexity = {local_cues["visual_complexity"]}

=====================
输出格式（严格遵守）
=====================
{{"spatial_pressure":"...","overall_dynamics":"..."}}
""".strip()



def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def _extract_json(text: str) -> str:
    """
    允许模型偶尔包一层 ```json ... ```，这里把 JSON 拎出来。
    """
    if not text:
        return ""
    t = text.strip()

    # 去掉 ```json ``` 外壳
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()

    # 如果还有多余文字，尝试截取第一个 {...} 区间
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        return t[l:r + 1]
    return t


def call_vlm(image_path: Path, prompt: str) -> Dict[str, str]:
    """
    调用 VLM：输入图片 + prompt，只允许返回：
      {"spatial_pressure":"...","overall_dynamics":"..."}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到环境变量 OPENAI_API_KEY，请先在终端设置后再运行。")

    client = OpenAI(api_key=api_key)

    mime = _guess_mime(image_path)
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime};base64,{img_b64}"

    last_err = None

    for attempt in range(1, VLM_MAX_RETRY + 1):
        try:
            # 官方“图像输入”格式：input_text + input_image（Responses API）
            # 参考 OpenAI Images & Vision 文档的 Python 示例结构
            resp = client.responses.create(
                model=VLM_MODEL,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
                temperature=VLM_TEMPERATURE,
            )

            raw = (resp.output_text or "").strip()
            js = _extract_json(raw)
            obj = json.loads(js)

            spatial = obj.get("spatial_pressure")
            dynamics = obj.get("overall_dynamics")

            # 严格枚举校验（不合规就触发重试）
            if spatial not in SPATIAL_ENUM or dynamics not in DYNAMICS_ENUM:
                raise ValueError(f"枚举不合规: spatial={spatial}, dynamics={dynamics}")

            return {"spatial_pressure": spatial, "overall_dynamics": dynamics}

        except Exception as e:
            last_err = e
            # 第一次失败后，第二次给更强硬的“只输出 JSON”提示，减少跑偏
            prompt = prompt + "\n\n再次强调：只输出严格 JSON，且必须从枚举中选择，不得输出任何其他内容。"

    # 多次失败：返回空，让主流程写 None（你后面会看到哪些需要复查）
    print(f"⚠️ VLM 失败（已重试 {VLM_MAX_RETRY} 次）: {image_path.name} -> {last_err}")
    return {"spatial_pressure": None, "overall_dynamics": None}



def main():
    gold_ids = json.loads(GOLD_IDS_PATH.read_text(encoding="utf-8"))
    gold_ids = [str(x) for x in gold_ids]  # 统一成 str，避免 123 vs "123"
    print(f"Loaded gold ids: {len(gold_ids)}")

    # ✅ 读取已有结果：如果已存在就跳过，避免浪费 API/算力
    existing_map, done_ids = load_existing_results(OUT_JSON)

    to_process = [gid for gid in gold_ids if gid not in done_ids]

    print("========== Stage1 断点续跑统计 ==========")
    print(f"📌 gold_ids 总数: {len(gold_ids)}")
    print(f"✅ 已有结果数: {len(done_ids)}")
    print(f"⏩ 本次跳过: {len(gold_ids) - len(to_process)}")
    print(f"🚀 本次新增待跑: {len(to_process)}")
    print("========================================")

    if len(to_process) == 0:
        print("🎉 所有ID都已有结果，无需重复跑。")
        print(f"📄 输出文件保持不变 -> {OUT_JSON}")
        return

    new_count = 0

    pbar = tqdm(
        enumerate(to_process, 1),
        total=len(to_process),
        desc="Stage1 (VLM) processing",
        unit="img"
    )

    for idx, item_id in pbar:
        # ===== 你原来的循环内容不变 =====
        img_path = resolve_image_path(item_id)
        if not img_path or not img_path.exists():
            # 可选：更新进度条后缀显示
            pbar.set_postfix_str(f"MISS {item_id}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            pbar.set_postfix_str(f"READ_ERR {item_id}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        local = {
            "brightness_pattern": compute_brightness_pattern(gray),
            "color_tone": compute_color_tone(img),
            "visual_complexity": compute_visual_complexity(gray),
        }

        prompt = build_prompt_for_spatial_and_dynamics(local)
        llm = call_vlm(img_path, prompt)

        spatial = llm.get("spatial_pressure")
        dynamics = llm.get("overall_dynamics")
        spatial = spatial if spatial in SPATIAL_ENUM else None
        dynamics = dynamics if dynamics in DYNAMICS_ENUM else None

        record = {
            "id": item_id,
            "image_path": str(img_path),
            "stage": 1,
            "global_cues": {
                **local,
                "spatial_pressure": spatial,
                "overall_dynamics": dynamics
            }
        }

        existing_map[item_id] = record
        new_count += 1

        # 每 50 张增量保存一次（你原来有的话可保留）
        if new_count % 50 == 0:
            save_results_as_list(OUT_JSON, existing_map)

        # ✅ 进度条右侧显示“当前总数/新增数”
        pbar.set_postfix(total=len(existing_map), new=new_count)



    # ✅ 最终写入 JSON（覆盖写，但内容是“旧+新”的合并）
    save_results_as_list(OUT_JSON, existing_map)

    print("\n========== Stage1 本次运行结果 ==========")
    print(f"🆕 本次新加: {new_count} 张")
    print(f"📦 当前总计: {len(existing_map)} 张（stage1_cues.json 内）")
    print(f"📄 输出文件 -> {OUT_JSON}")
    print("========================================\n")


if __name__ == "__main__":
    main()
