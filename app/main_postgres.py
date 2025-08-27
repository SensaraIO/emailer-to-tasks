import os, re, json, base64, datetime as dt
from typing import Optional, List, Dict
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import psycopg
import httpx

# Google
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI")

GMAIL_LABEL_IDS = [x.strip() for x in (os.getenv("GMAIL_LABEL_IDS","").split(",")) if x.strip()]
GMAIL_TOPIC = os.getenv("GMAIL_TOPIC")
PUBSUB_VERIFICATION_TOKEN = os.getenv("PUBSUB_VERIFICATION_TOKEN", "")

app = FastAPI(title="Gmail → AI → TaskBoard")

# -------------------- DB helpers --------------------
def db():
    return psycopg.connect(DATABASE_URL, autocommit=True)

def slugify(s: str) -> str:
    import re
    return re.sub(r'[^a-z0-9\-]+','-', re.sub(r'\s+','-', s.strip().lower())).strip('-')

# -------------------- Subject parsing --------------------
BC_SUBJECT_RE = re.compile(
    r"""
    (?:^|\s)\(
    \s*\$coding-\s*
    (?P<app>.+?)\s*
    \(
      (?P<client>.+?)
    \)
    \)\s*
    (?P<thread>.+?)\s*$
    """, re.IGNORECASE | re.VERBOSE
)

CLIENT_ONLY_RE = re.compile(
    r"""^\(\s*(?P<client>[^)]+)\s*\)\s*(?P<thread>.+?)\s*$""",
    re.IGNORECASE | re.VERBOSE
)

def parse_subject(subject: str):
    subject = (subject or "").strip()
    m = BC_SUBJECT_RE.search(subject)
    if m:
        app = m.group("app").strip()
        client = m.group("client").strip()
        thread = m.group("thread").strip()
        project_name = f"$coding- {app} ({client})"
        return {"kind":"coding", "project_name": project_name, "app": app, "client": client, "thread": thread}
    m2 = CLIENT_ONLY_RE.search(subject)
    if m2:
        client = m2.group("client").strip()
        thread = m2.group("thread").strip()
        return {"kind":"client", "project_name": client, "app": None, "client": client, "thread": thread}
    return {"kind":"client", "project_name": subject or "General", "app": None, "client": None, "thread": "General"}

# -------------------- AI extraction --------------------
class ExtractedTask(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Optional[str] = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    due_on: Optional[str] = None
    assignee_hint: Optional[str] = None

class AIExtractResponse(BaseModel):
    tasks: List[ExtractedTask] = []

SYSTEM_PROMPT = """You are a task extraction engine.
Extract zero or more actionable tasks from an email. 
Priority: low|normal|high|urgent. Dates: use YYYY-MM-DD or omit.
"""

def build_user_prompt(subject: str, body: str) -> str:
    return f"""Subject: {subject}

Email:
{body}

Return strictly this JSON schema:
{AIExtractResponse.schema_json(indent=2)}
"""

async def ai_extract_tasks(subject: str, body_text: str) -> AIExtractResponse:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type":"json_object"},
        "messages": [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": build_user_prompt(subject, (body_text or "")[:8000])}
        ],
        "temperature": 0.1
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return AIExtractResponse(**json.loads(data["choices"][0]["message"]["content"]))
    except Exception:
        return AIExtractResponse(tasks=[])

# -------------------- OAuth & Gmail client --------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.modify"]

def save_tokens(email:str, creds:Credentials):
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            insert into google_oauth (email, access_token, refresh_token, token_expiry, scope)
            values (%s,%s,%s,%s,%s)
            on conflict (email) do update set
              access_token=excluded.access_token,
              refresh_token=excluded.refresh_token,
              token_expiry=excluded.token_expiry,
              scope=excluded.scope,
              updated_at=now()
        """, (email, creds.token, creds.refresh_token or "", creds.expiry, " ".join(creds.scopes or [])))

def load_tokens(email:str) -> Optional[Credentials]:
    with db() as conn, conn.cursor() as cur:
        cur.execute("select access_token, refresh_token, token_expiry, scope from google_oauth where email=%s", (email,))
        row = cur.fetchone()
        if not row: return None
        access, refresh, expiry, scope = row
        creds = Credentials(
            token=access,
            refresh_token=refresh,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=(scope or "").split()
        )
        # set expiry
        if isinstance(expiry, dt.datetime):
            creds.expiry = expiry
        return creds

def gmail_service(creds: Credentials):
    if not creds.valid and creds.refresh_token:
        # google library will auto-refresh when used via build()
        pass
    return build("gmail", "v1", credentials=creds, cache_discovery=False)

# -------------------- OAuth routes --------------------
@app.get("/oauth/google/start")
def oauth_start():
    flow = Flow.from_client_config(
        {
          "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [OAUTH_REDIRECT_URI]
          }
        },
        scopes=SCOPES
    )
    flow.redirect_uri = OAUTH_REDIRECT_URI
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    # store state in a short-lived cookie if you like; for simplicity return it
    return JSONResponse({"auth_url": auth_url, "state": state})

@app.get("/oauth/google/callback")
def oauth_callback(code: str, state: Optional[str] = None):
    flow = Flow.from_client_config(
        {
          "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [OAUTH_REDIRECT_URI]
          }
        },
        scopes=SCOPES
    )
    flow.redirect_uri = OAUTH_REDIRECT_URI
    flow.fetch_token(code=code)
    creds = flow.credentials
    # determine the account's email
    svc = gmail_service(creds)
    profile = svc.users().getProfile(userId="me").execute()
    email_addr = profile["emailAddress"]
    save_tokens(email_addr, creds)
    return RedirectResponse(url=f"/gmail/watch?email={email_addr}")

# -------------------- Start/stop Gmail watch --------------------
@app.get("/gmail/watch")
def gmail_watch(email: str):
    creds = load_tokens(email)
    if not creds: raise HTTPException(400, "Connect Gmail first.")
    svc = gmail_service(creds)

    body = {"topicName": GMAIL_TOPIC}
    if GMAIL_LABEL_IDS:
        body["labelIds"] = GMAIL_LABEL_IDS
    resp = svc.users().watch(userId="me", body=body).execute()

    history_id = resp.get("historyId")
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
          insert into gmail_state (email, history_id, watch_active, updated_at)
          values (%s,%s,true,now())
          on conflict (email) do update set history_id=excluded.history_id, watch_active=true, updated_at=now()
        """, (email, history_id))
    return {"ok": True, "email": email, "history_id": history_id}

@app.get("/gmail/stop")
def gmail_stop(email: str):
    creds = load_tokens(email)
    if not creds: raise HTTPException(400, "Connect Gmail first.")
    svc = gmail_service(creds)
    svc.users().stop(userId="me").execute()
    with db() as conn, conn.cursor() as cur:
        cur.execute("update gmail_state set watch_active=false, updated_at=now() where email=%s", (email,))
    return {"ok": True}

# -------------------- Pub/Sub push endpoint --------------------
class PubSubPush(BaseModel):
    message: Dict
    subscription: Optional[str] = None

def verify_pubsub(request: Request):
    if PUBSUB_VERIFICATION_TOKEN:
        token = request.headers.get("X-Goog-Resource-State", "")  # not actually a token; alt custom scheme:
        # Simpler: use a querystring ?token=..., or header "X-PubSub-Token"
    return True

@app.post("/gmail/push")
async def gmail_push(request: Request):
    # (Optional) add a token check in query/header if you like.
    body = await request.json()
    msg = body.get("message", {})
    data_b64 = msg.get("data", "")
    if not data_b64:
        return {"ok": True}

    data = json.loads(base64.b64decode(data_b64))
    # data has {emailAddress, historyId}
    email = data.get("emailAddress")
    history_id = int(data.get("historyId", 0))
    if not email or not history_id:
        return {"ok": True}

    # Continue processing in-line (simple); or enqueue to a worker.
    await process_history(email, history_id)
    return {"ok": True}

# -------------------- Process Gmail history & create tasks --------------------
async def process_history(email: str, new_history_id: int):
    creds = load_tokens(email)
    if not creds: return
    svc = gmail_service(creds)

    # find last history id
    with db() as conn, conn.cursor() as cur:
      cur.execute("select history_id from gmail_state where email=%s", (email,))
      row = cur.fetchone()
      last_history = int(row[0]) if row and row[0] else None

    # If we have no cursor, just set it; avoid backfilling everything.
    if not last_history:
        with db() as conn, conn.cursor() as cur:
            cur.execute("update gmail_state set history_id=%s, updated_at=now() where email=%s", (new_history_id, email))
        return

    page_token = None
    while True:
        hist = svc.users().history().list(
            userId="me",
            startHistoryId=last_history,
            historyTypes=["messageAdded"],
            pageToken=page_token
        ).execute()
        for h in hist.get("history", []):
            for added in h.get("messagesAdded", []):
                msg_id = added["message"]["id"]
                await handle_message(svc, msg_id)
        page_token = hist.get("nextPageToken")
        if not page_token: break

    # advance cursor
    with db() as conn, conn.cursor() as cur:
        cur.execute("update gmail_state set history_id=%s, updated_at=now() where email=%s",
                    (new_history_id, email))

def decode_mime_b64(s: str) -> str:
    try:
        return base64.urlsafe_b64decode(s.encode("utf-8")).decode("utf-8", errors="ignore")
    except Exception:
        return ""

async def handle_message(svc, msg_id: str):
    msg = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()

    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
    subject = headers.get("subject", "")
    from_raw = headers.get("from", "")
    from_name, from_email = parse_from(from_raw)
    received_ts = int(msg.get("internalDate", "0"))/1000.0
    received_at = dt.datetime.utcfromtimestamp(received_ts)

    # Body (plain/text preferred; else html)
    body_text, body_html = extract_bodies(msg.get("payload", {}))

    # Parse subject → project/thread
    pieces = parse_subject(subject)

    # Write to DB + AI extract
    with db() as conn, conn.cursor() as cur:
        project_id = upsert_project(cur, pieces["project_name"], pieces["app"], pieces["client"], pieces["kind"])
        thread_id  = upsert_thread(cur, project_id, pieces["thread"])
        cur.execute("""insert into emails (thread_id, gmail_message_id, from_name, from_email, subject, body_html, body_text, received_at)
                       values (%s,%s,%s,%s,%s,%s,%s,%s) returning id""",
                    (thread_id, msg_id, from_name, from_email, subject, body_html, body_text, received_at))
        email_row_id = cur.fetchone()[0]

    # AI extract → tasks
    extraction = await ai_extract_tasks(subject, body_text or "")
    with db() as conn, conn.cursor() as cur:
        for t in extraction.tasks:
            due = None
            if t.due_on:
                try: due = dt.date.fromisoformat(t.due_on)
                except: pass
            cur.execute("""insert into tasks (thread_id, title, description, priority, due_on, assignee_hint, source_email_id)
                           values (%s,%s,%s,%s,%s,%s,%s)""",
                        (thread_id, t.title, t.description, (t.priority or "normal"), due, t.assignee_hint, email_row_id))

def upsert_project(cur, name: str, app: Optional[str], client: Optional[str], kind: str):
    nk = slugify(name)
    cur.execute("""insert into projects (name, normalized_key, app_name, client_name, kind)
                   values (%s,%s,%s,%s,%s)
                   on conflict (normalized_key) do update set name=excluded.name
                   returning id""",
                (name, nk, app, client, kind))
    return cur.fetchone()[0]

def upsert_thread(cur, project_id, subject: str):
    cur.execute("""insert into threads (project_id, subject)
                   values (%s,%s)
                   on conflict (project_id, subject) do nothing
                   returning id""",
                (project_id, subject))
    row = cur.fetchone()
    if row: return row[0]
    cur.execute("select id from threads where project_id=%s and subject=%s", (project_id, subject))
    return cur.fetchone()[0]

def parse_from(from_header: str):
    import email.utils as eu
    name, addr = eu.parseaddr(from_header)
    return (name or None, addr or None)

def extract_bodies(payload) -> (str, str):
    mimeType = payload.get("mimeType")
    if mimeType == "text/plain":
        body_text = decode_mime_b64(payload.get("body", {}).get("data", ""))
        return body_text, None
    if mimeType == "text/html":
        body_html = decode_mime_b64(payload.get("body", {}).get("data", ""))
        return None, body_html
    if mimeType and mimeType.startswith("multipart/"):
        parts = payload.get("parts", []) or []
        text, html = None, None
        for p in parts:
            t, h = extract_bodies(p)
            text = text or t
            html = html or h
        return text, html
    return None, None