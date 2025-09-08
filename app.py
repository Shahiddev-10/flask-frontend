import re, os, io, time, json, zipfile, threading
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List

import pandas as pd
import requests
from flask import Flask, request, jsonify, send_file, render_template
from flask_socketio import SocketIO

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "dev"
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize SocketIO with eventlet async mode
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

PROCESS = {
    "thread": None,
    "paused": False,
    "stopped": False,
    "eta": None,
    "rate": None,
    "start_time": None,
    "processed": 0,
    "total": 0,
    "current_group": "-",
    "results": [],       # list of dicts: index, group, prompt, response, error
    "format": "csv",     # output format selected
    "output_dir": "outputs",
}
lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file provided"}), 400

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{ts}_{f.filename}"
    path = os.path.join("uploads", safe_name)
    f.save(path)

    ext = (f.filename.rsplit(".", 1)[-1] or "").lower()
    try:
        if ext == "csv":
            df = pd.read_csv(path)
        elif ext == "json":
            text = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
            if text.startswith("["):
                data = json.loads(text)  # JSON array
                df = pd.DataFrame(data if isinstance(data, list) else [data])
            else:
                # NDJSON fallback
                records = [json.loads(line) for line in text.splitlines() if line.strip()]
                df = pd.DataFrame(records)
        elif ext == "txt":
            lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
            df = pd.DataFrame({"text": lines})
        else:
            return jsonify({"error": f"Unsupported file type: .{ext}"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {e}"}), 400

    cols = list(df.columns)
    return jsonify({"filename": safe_name, "rows": len(df), "columns": cols})


@app.post("/start_processing")
def start_processing():
    with lock:
        if PROCESS["thread"] and PROCESS["thread"].is_alive():
            return jsonify({"error": "Processing already running"}), 400

        cfg = request.get_json(force=True)
        if not cfg:
            return jsonify({"error": "Missing config"}), 400
        data_path = os.path.join("uploads", cfg["data_file"])
        if not os.path.exists(data_path):
            return jsonify({"error": "Data file not found"}), 400

        # Load dataframe
        ext = cfg["data_file"].rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = pd.read_csv(data_path)
        elif ext == "json":
            data = json.load(open(data_path, "r", encoding="utf-8"))
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            lines = open(data_path, "r", encoding="utf-8", errors="ignore").read().splitlines()
            df = pd.DataFrame({"text": lines})

        # Reset process state
        PROCESS.update({
            "paused": False,
            "stopped": False,
            "eta": None,
            "rate": None,
            "start_time": time.time(),
            "processed": 0,
            "total": len(df),
            "current_group": "-",
            "results": [],
            "format": cfg["output"]["format"],
            "output_dir": cfg["output"]["directory"].rstrip("/"),
        })

        # Spawn worker thread
        PROCESS["thread"] = threading.Thread(
            target=run_batch,
            args=(df, cfg),
            daemon=True
        )
        PROCESS["thread"].start()

    return jsonify({"status": "started"})

@app.post("/pause_processing")
def pause_processing():
    with lock:
        PROCESS["paused"] = not PROCESS["paused"]
        status = "paused" if PROCESS["paused"] else "resumed"
    return jsonify({"status": status})

@app.post("/stop_processing")
def stop_processing():
    with lock:
        PROCESS["stopped"] = True
    return jsonify({"status": "stopping"})


@app.get("/export_results")
def export_results():
    # Build files according to chosen format
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        rows = PROCESS["results"]
        if not rows:
            # Always include an empty CSV for clarity
            z.writestr("results.csv", "index,group,prompt,response,error\n")

        # Consolidated CSV
        csv_lines = ["index,group,prompt,response,error"]
        for r in rows:
            def esc(s): 
                if s is None: return ""
                s = str(s).replace('"', '""')
                return f'"{s}"'
            csv_lines.append(f'{r["index"]},{esc(r["group"])},{esc(r["prompt"])},{esc(r["response"])},{esc(r.get("error",""))}')
        z.writestr("results.csv", "\n".join(csv_lines))

        # Consolidated JSON
        z.writestr("results.json", json.dumps(rows, ensure_ascii=False, indent=2))

        # Optional individual files
        if PROCESS["format"] in ("individual", "both"):
            for r in rows:
                name = f'{r["index"]:05d}_{sanitize_filename(r["group"] or "row")}.txt'
                z.writestr(f"individual/{name}", r.get("response", "") or "")

    buf.seek(0)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"ai_batch_{ts}.zip")


@app.get("/get_status")
def get_status():
    with lock:
        return jsonify({
            "eta": PROCESS["eta"],
            "rate": PROCESS["rate"],
            "processed": PROCESS["processed"],
            "total": PROCESS["total"],
            "current_group": PROCESS["current_group"]
        })


def sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in s)[:64]

def run_batch(df: pd.DataFrame, cfg: Dict[str, Any]):
    svc = cfg["ai_config"]["service"]
    model = cfg["ai_config"]["model"]
    api_key = cfg["ai_config"]["api_key"]
    temperature = float(cfg["ai_config"]["params"]["temperature"])
    max_tokens = int(cfg["ai_config"]["params"]["max_tokens"])
    rpm = max(int(cfg["ai_config"]["rate_limit"]), 1)
    delay = 60.0 / rpm

    group_by = cfg["mapping"]["group_by"]
    group_by = None if (not group_by or group_by == "None") else group_by
    system_prompt = (cfg.get("prompt_template", {}) or {}).get("system", "").strip()
    main_template = (cfg.get("prompt_template", {}) or {}).get("main", "").strip()

    include_prompt = bool(cfg["output"]["include_prompt"])

    # Conversation state per group
    conversations: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    start = time.time()
    errors = 0

    for i, row in df.iterrows():
        with lock:
            if PROCESS["stopped"]:
                break
        # Pause handling
        while True:
            with lock:
                if PROCESS["stopped"]:
                    break
                paused = PROCESS["paused"]
            if not paused:
                break
            cooperative_sleep(0.2)


        # Build variables dict from row
        vars_map = {str(k): ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}
        prompt = safe_format_blank_missing(main_template, vars_map)

        # Determine group key (or use per-row)
        group = str(vars_map.get(group_by)) if group_by else f"row_{i+1}"
        with lock:
            PROCESS["current_group"] = group

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if group_by and conversations[group]:
            # reuse history
            messages.extend(conversations[group])
        messages.append({"role": "user", "content": prompt})

        # Call model with retries
        response_text, err = call_llm(
            service=svc,
            model=model,
            api_key=api_key,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=int(cfg.get("retryAttempts", 3))
        )

        if err:
            errors += 1
            socketio.emit("item_error", {"index": i, "group": group, "error": err})
        else:
            # Update conversation with assistant reply if grouping
            if group_by:
                conversations[group].extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text or ""}
                ])
            socketio.emit("item_completed", {"index": i, "group": group})

        # Save result row
        PROCESS["results"].append({
            "index": i + 1,
            "group": group if group_by else "",
            "prompt": prompt if include_prompt else "",
            "response": response_text or "",
            "error": err or ""
        })

        # Progress + ETA
        with lock:
            PROCESS["processed"] += 1
            processed = PROCESS["processed"]
            total = PROCESS["total"]
        elapsed = time.time() - start
        rate_items_per_min = processed / max(elapsed / 60.0, 1e-9)
        remaining = total - processed
        eta_seconds = (remaining / max(rate_items_per_min, 1e-9)) * 60.0

        with lock:
            PROCESS["rate"] = rate_items_per_min
            PROCESS["eta"] = eta_seconds

        socketio.emit("progress_update", {"current": processed, "total": total, "group": group})

        # Simple rate-limit sleep
        cooperative_sleep(delay)

    socketio.emit("batch_completed", {"total_processed": PROCESS["processed"], "total_errors": errors})


def safe_format_blank_missing(template: str, variables: Dict[str, Any]) -> str:
    return re.sub(r"\{([^{}]+)\}", lambda m: str(variables.get(m.group(1), "")), template)


def call_llm(service, model, api_key, messages, temperature, max_tokens, retries=3):
    """
    Minimal blocking HTTP calls for MVP.
    NOTE: swap to the exact vendor APIs you use. This is intentionally simple.
    """
    for attempt in range(retries):
        try:
            if service == "openai":
                # Classic Chat Completions for MVP
                r = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=90
                )
                if r.status_code >= 400:
                    raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                return content, None

            elif service == "anthropic":
                # Minimal Anthropic Messages API
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [
                            {"role": m["role"], "content": m["content"]} for m in messages
                            if m["role"] in ("user", "assistant")  # Anthropic doesn't use "system" same way
                        ]
                    },
                    timeout=90
                )
                if r.status_code >= 400:
                    raise RuntimeError(f"Anthropic error {r.status_code}: {r.text}")
                data = r.json()
                content = "".join(block.get("text", "") for block in data.get("content", []))
                return content, None

            else:
                return None, f"Unknown service: {service}"

        except Exception as e:
            if attempt == retries - 1:
                return None, str(e)
            socketio.sleep(1.5 * (attempt + 1))  # simple backoff


def cooperative_sleep(total_seconds):
    end = time.time() + total_seconds
    while time.time() < end:
        with lock:
            if PROCESS["stopped"] or PROCESS["paused"]:
                return
        socketio.sleep(0.1)



if __name__ == "__main__":
    # For MVP we stick to threading; no eventlet/gevent required.
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
