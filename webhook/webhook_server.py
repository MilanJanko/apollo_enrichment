from fastapi import FastAPI, Request
from datetime import datetime
import json
import os

app = FastAPI(title="Apollo Phone Webhook")

DATA_DIR = "data"
EVENTS_FILE = os.path.join(DATA_DIR, "phone_events.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)


@app.post("/apollo/phone-webhook")
async def apollo_phone_webhook(request: Request):
    payload = await request.json()

    event = {
        "received_at": datetime.utcnow().isoformat(),
        "payload": payload,
    }

    with open(EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}
