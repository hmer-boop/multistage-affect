# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "dataset"
RAW_DIR = DATA_ROOT / "images_raw"
META_DIR = DATA_ROOT / "metadata"
ITEMS_JSONL = META_DIR / "items_merged.jsonl"

RESULT_DIR = BASE_DIR / "results"
RESULT_FILE = RESULT_DIR / "results.jsonl"
SUMMARY_JSON = RESULT_DIR / "summary.json"
SHARDS_DIR = RESULT_DIR / "split"

ROUND2_RESULT_FILE = RESULT_DIR / "results_round2.jsonl"
ROUND2_SUMMARY_JSON = RESULT_DIR / "summary_round2.json"
ROUND2_GOLD_JSON = RESULT_DIR / "gold_label_round2.json"
ROUND2_DISCARDED_JSON = RESULT_DIR / "discarded_round2.json"

USER_IDS = ["id1", "id2", "id3", "id4", "id5"]
LABELS = ["悲伤", "恐惧", "厌恶", "愤怒", "宁静", "快乐", "惊奇", "敬畏"]
SHARD_SIZE = 500
STAGE1_SECONDS = 3.0

RESULT_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.secret_key = "replace-this-with-a-random-secret-5raters-stage1-3s"
app.config["SESSION_COOKIE_NAME"] = "session_5raters_stage1_3s"


def format_seconds(seconds: float) -> str:
    return str(int(seconds)) if float(seconds).is_integer() else str(seconds)


def load_items() -> List[dict]:
    items: List[dict] = []
    seen = set()
    if not ITEMS_JSONL.exists():
        return items
    with ITEMS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            item_id = rec.get("item_id") or rec.get("id")
            if not item_id or item_id in seen:
                continue
            item_id = str(item_id)
            rec["item_id"] = item_id
            rec.setdefault("path_raw", str((RAW_DIR / f"{item_id}.jpg").as_posix()))
            items.append(rec)
            seen.add(item_id)
    return items


def items_sorted() -> List[dict]:
    return sorted(load_items(), key=lambda x: x["item_id"])


def iter_results() -> Iterable[dict]:
    if not RESULT_FILE.exists():
        return
    with RESULT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def append_result(rec: dict) -> None:
    with RESULT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def iter_json_array_ids(path: Path) -> Iterable[str]:
    if not path.exists():
        return
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    rows = obj.get("items") if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        return
    for row in rows:
        if isinstance(row, dict):
            item_id = row.get("item_id") or row.get("id")
            if item_id:
                yield str(item_id)


def iter_jsonl_ids(path: Path) -> Iterable[str]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            item_id = row.get("item_id") or row.get("id")
            if item_id:
                yield str(item_id)


def round2_done_ids() -> set[str]:
    ids: set[str] = set()
    for path in (ROUND2_SUMMARY_JSON, ROUND2_GOLD_JSON, ROUND2_DISCARDED_JSON):
        ids.update(iter_json_array_ids(path) or [])
    ids.update(iter_jsonl_ids(ROUND2_RESULT_FILE) or [])
    return ids


def active_items() -> List[dict]:
    skip_ids = round2_done_ids()
    return [rec for rec in items_sorted() if rec["item_id"] not in skip_ids]


def active_item_ids() -> set[str]:
    return {rec["item_id"] for rec in active_items()}


def done_set_for_rater(rater_id: str) -> set[str]:
    per_item_stages: Dict[str, set[str]] = {}
    active_ids = active_item_ids()
    for rec in iter_results() or []:
        if rec.get("rater_id") != rater_id:
            continue
        item_id = str(rec.get("item_id") or "")
        if item_id not in active_ids:
            continue
        stage = str(rec.get("stage") or "")
        if stage:
            per_item_stages.setdefault(item_id, set()).add(stage)
    return {item_id for item_id, stages in per_item_stages.items() if {"1", "2", "3"}.issubset(stages)}


def progress_text(rater_id: str) -> str:
    return f"{len(done_set_for_rater(rater_id))}/{len(active_items())}"


def next_item_for_rater(rater_id: str) -> Optional[dict]:
    done_ids = done_set_for_rater(rater_id)
    for rec in active_items():
        if rec["item_id"] not in done_ids:
            return rec
    return None


def global_index(item_id: str) -> int:
    for idx, rec in enumerate(items_sorted(), start=1):
        if rec["item_id"] == item_id:
            return idx
    return 0


def shard_path_for(item_id: str) -> Path:
    idx = global_index(item_id) or 1
    shard_num = (idx - 1) // SHARD_SIZE + 1
    return SHARDS_DIR / f"shard_{shard_num:04d}.json"


def load_summary_map(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        arr = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(arr, list):
        return {}
    out = {}
    for row in arr:
        if isinstance(row, dict) and row.get("item_id"):
            out[str(row["item_id"])] = row
    return out


def write_pretty_array(path: Path, mapping: dict) -> None:
    arr = [mapping[item_id] for item_id in sorted(mapping.keys())]
    with path.open("w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)


def write_summary_array_in_dataset_order(path: Path, mapping: dict) -> None:
    order_map = {rec["item_id"]: i for i, rec in enumerate(load_items()) if rec.get("item_id")}

    def sort_key(item_id: str):
        if item_id in order_map:
            return (0, order_map[item_id], "")
        return (1, 10**12, item_id)

    arr = [mapping[item_id] for item_id in sorted(mapping.keys(), key=sort_key)]
    with path.open("w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)


def update_summary_and_shards(rater_id: str, item_id: str, l1: str, l2: str, l3: str, now_ts: str) -> None:
    summary_map = load_summary_map(SUMMARY_JSON)
    base = next((x for x in items_sorted() if x["item_id"] == item_id), {})
    obj = summary_map.get(item_id) or {
        "item_id": item_id,
        "title_zh": base.get("title_zh", ""),
        "title_en": base.get("title_en", ""),
        "caption_zh": base.get("caption_zh", ""),
        "caption_en": base.get("caption_en", ""),
        "path_raw": base.get("path_raw"),
        "raters": {},
    }
    obj.setdefault("raters", {})
    obj["raters"][rater_id] = {"stage1": l1 or "", "stage2": l2 or "", "stage3": l3 or ""}
    if l1 and l2 and l3 and "first_complete_ts_any_rater" not in obj:
        obj["first_complete_ts_any_rater"] = now_ts
        obj["first_complete_by_rater"] = rater_id

    summary_map[item_id] = obj
    write_summary_array_in_dataset_order(SUMMARY_JSON, summary_map)

    shard_path = shard_path_for(item_id)
    shard_map = load_summary_map(shard_path)
    shard_map[item_id] = obj
    write_pretty_array(shard_path, shard_map)


@app.route("/rules", methods=["GET", "POST"])
def rules():
    if request.method == "POST":
        uid = (request.form.get("uid") or "").strip()
        if uid in USER_IDS:
            session["rater_id"] = uid
            return redirect(url_for("annotate"))
        return render_template(
            "rules_5raters_stage1_3s.html",
            user_ids=USER_IDS,
            error="请选择有效的账号ID",
            current=session.get("rater_id"),
            stage1_display_label=format_seconds(STAGE1_SECONDS),
        )
    return render_template(
        "rules_5raters_stage1_3s.html",
        user_ids=USER_IDS,
        error=None,
        current=session.get("rater_id"),
        stage1_display_label=format_seconds(STAGE1_SECONDS),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    return rules()


@app.route("/logout")
def logout():
    session.pop("rater_id", None)
    return redirect(url_for("rules"))


@app.before_request
def ensure_rater():
    if request.path in ("/healthz", "/login", "/logout", "/", "/rules") or request.path.startswith("/dataset/"):
        return
    if "rater_id" in session:
        return
    return redirect(url_for("rules"))


@app.route("/dataset/<path:subpath>")
def serve_dataset(subpath):
    return send_from_directory(DATA_ROOT, subpath)


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/", methods=["GET"])
def root():
    return redirect(url_for("rules"))


@app.route("/annotate", methods=["GET"])
def annotate():
    rater_id = session.get("rater_id")
    item = next_item_for_rater(rater_id)
    return render_template(
        "annotate_5raters_stage1_3s.html",
        item=item,
        emotions=LABELS,
        uid=rater_id,
        progress=progress_text(rater_id),
        stage1_display_seconds=STAGE1_SECONDS,
        stage1_display_label=format_seconds(STAGE1_SECONDS),
        finished=(item is None),
    )


@app.route("/submit", methods=["POST"])
def submit():
    data = request.form or request.json or {}
    rater_id = session.get("rater_id")
    item_id = str(data.get("item_id", "")).strip()
    l1 = (data.get("label_stage1") or "").strip()
    l2 = (data.get("label_stage2") or "").strip()
    l3 = (data.get("label_stage3") or "").strip()

    if not (rater_id and item_id):
        return jsonify({"status": "error", "msg": "缺少 rater_id 或 item_id"}), 400
    if not (l1 and l2 and l3):
        return jsonify({"status": "error", "msg": "三个阶段都需要选择标签"}), 400

    now = datetime.now().isoformat(timespec="seconds")
    append_result({"ts": now, "rater_id": rater_id, "item_id": item_id, "stage": "1", "label": l1})
    append_result({"ts": now, "rater_id": rater_id, "item_id": item_id, "stage": "2", "label": l2})
    append_result({"ts": now, "rater_id": rater_id, "item_id": item_id, "stage": "3", "label": l3})
    update_summary_and_shards(rater_id, item_id, l1, l2, l3, now_ts=now)

    nxt = next_item_for_rater(rater_id)
    if not nxt:
        return jsonify({"status": "done", "progress": progress_text(rater_id)})

    next_payload = {
        "item_id": nxt.get("item_id"),
        "path_raw": nxt.get("path_raw"),
        "title_zh": nxt.get("title_zh", ""),
        "title_en": nxt.get("title_en", ""),
        "caption_zh": nxt.get("caption_zh", ""),
        "caption_en": nxt.get("caption_en", ""),
    }
    return jsonify({"status": "ok", "next": next_payload, "progress": progress_text(rater_id)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)
