import os, re, json, base64, datetime as dt
from typing import Optional, List, Dict
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
import httpx
from bson import ObjectId

# Google
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", os.getenv("DATABASE_URL"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI")

GMAIL_LABEL_IDS = [x.strip() for x in (os.getenv("GMAIL_LABEL_IDS","").split(",")) if x.strip()]
GMAIL_TOPIC = os.getenv("GMAIL_TOPIC")
PUBSUB_VERIFICATION_TOKEN = os.getenv("PUBSUB_VERIFICATION_TOKEN", "")

app = FastAPI(title="Gmail → AI → TaskBoard")

# -------------------- MongoDB Setup --------------------
client = AsyncIOMotorClient(MONGODB_URI)
db = client.emailer_tasks

# Collections
projects_col = db.projects
threads_col = db.threads
emails_col = db.emails
tasks_col = db.tasks
google_oauth_col = db.google_oauth
gmail_state_col = db.gmail_state

# Create indexes on startup
@app.on_event("startup")
async def create_indexes():
    # Unique indexes
    await projects_col.create_index("normalized_key", unique=True)
    await threads_col.create_index([("project_id", 1), ("subject", 1)], unique=True)
    await google_oauth_col.create_index("email", unique=True)
    await gmail_state_col.create_index("email", unique=True)
    
    # Regular indexes for queries
    await emails_col.create_index("thread_id")
    await tasks_col.create_index("thread_id")
    await emails_col.create_index("gmail_message_id")
    await tasks_col.create_index([("client_name", 1), ("side", 1), ("status", 1), ("created_at", -1)])
    await emails_col.create_index([("client_name", 1), ("side", 1), ("received_at", -1)])

def slugify(s: str) -> str:
    import re
    return re.sub(r'[^a-z0-9\-]+','-', re.sub(r'\s+','-', s.strip().lower())).strip('-')

# -------------------- Subject parsing --------------------
BC_SUBJECT_RE = re.compile(
    r"""
    (?:^|\s)\(
    \s*\$\s*coding\s*[-–—]?\s*
    (?P<app>.+?)\s*
    \(
      (?P<client>.+?)
    \)
    \)\s*
    (?P<thread>.+?)\s*$
    """, re.IGNORECASE | re.VERBOSE
)

CLIENT_ONLY_RE = re.compile(
    r"""^(?:re:|fwd?:)?\s*\(\s*(?P<client>[^)]+)\s*\)\s*(?P<thread>.+?)\s*$""",
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

 
# -------------------- Subject hygiene & heuristics --------------------
NON_PROJECT_SUBJECT_PATTERNS = [
    r"delivery status notification",
    r"mail delivery (?:subsystem|failed)",
    r"out of office",
    r"auto(?:matic)? reply",
    r"added you to a project in basecamp",
    r"invitation",
    r"newsletter",
]

def looks_like_non_project(subject: str) -> bool:
    s = (subject or "").lower()
    return any(re.search(p, s) for p in NON_PROJECT_SUBJECT_PATTERNS)

PREFIX_RE = re.compile(r"^(re:|fwd?:)\s*", re.IGNORECASE)

def clean_subject(s: str) -> str:
    s = (s or "").strip()
    # strip repeated RE:/FWD:
    while True:
        m = PREFIX_RE.match(s)
        if not m:
            break
        s = s[m.end():].strip()
    return s

CLIENT_NAME_NORMALIZER = re.compile(r"\s+\(|\)\s*$")

def normalize_client_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip()
    # collapse multiple spaces and strip stray parentheses
    n = re.sub(r"\s+", " ", n)
    n = CLIENT_NAME_NORMALIZER.sub("", n)
    return n

# -------------------- AI extraction --------------------
class ExtractedTask(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Optional[str] = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    due_on: Optional[str] = None
    assignee_hint: Optional[str] = None

class AIExtractResponse(BaseModel):
    tasks: List[ExtractedTask] = []

# -------------------- AI classification model --------------------
class EmailClassification(BaseModel):
    is_project_email: bool = True
    client_name: Optional[str] = None
    side: Optional[str] = Field(default=None, pattern=r"^(client|dev)$")
    thread: Optional[str] = None
    addressed_to_zac: bool = False
    coding_app_name: Optional[str] = None

CLASSIFY_SYSTEM_PROMPT = (
    "You are an expert at reading Basecamp email notifications and normal emails "
    "and classifying them for a task system. "
    "Infer the canonical CLIENT name (e.g., 'John James'). "
    "If the subject contains a pattern like '($Coding- Project Name (Client)) ...', set side='dev' and coding_app_name='Project Name'. "
    "If the subject contains '(Client) ...' without '$Coding-', set side='client'. "
    "Extract a short thread title (like 'Designs', 'Internal Review'). "
    "Set addressed_to_zac=true if the To/CC list includes any variant of Zachary/Zac Cheshire. "
    "Set is_project_email=false for bounces, OOO, newsletters, or Basecamp meta emails like 'added you to a project'. "
    "Return only the JSON for EmailClassification."
)

def build_classify_prompt(subject: str, body: str, headers: Dict[str, str]) -> str:
    to = headers.get("to", "")
    cc = headers.get("cc", "")
    frm = headers.get("from", "")
    return (
        f"Subject: {subject}\n\n"
        f"From: {frm}\nTo: {to}\nCC: {cc}\n\n"
        f"Body (first 2k):\n{(body or '')[:2000]}\n\n"
        "Return strictly this JSON schema (fields may be null):\n" + EmailClassification.schema_json(indent=2)
    )

async def ai_classify_email(subject: str, body_text: str, headers: Dict[str, str]) -> EmailClassification:
    # quick local guardrails first
    if looks_like_non_project(subject):
        return EmailClassification(is_project_email=False)

    url = "https://api.openai.com/v1/chat/completions"
    headers_http = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": build_classify_prompt(subject, body_text or "", headers)},
        ],
        "temperature": 0.0,
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(url, headers=headers_http, json=payload)
            r.raise_for_status()
            data = r.json()
        parsed = json.loads(data["choices"][0]["message"]["content"]) or {}
        return EmailClassification(**parsed)
    except Exception:
        # Fallback: attempt regex-based classification
        s = clean_subject(subject)
        m = BC_SUBJECT_RE.search(s)
        if m:
            client_name = normalize_client_name(m.group("client").strip())
            thread = m.group("thread").strip()
            return EmailClassification(is_project_email=True, client_name=client_name, side="dev", thread=thread, coding_app_name=m.group("app").strip())
        m2 = CLIENT_ONLY_RE.search(s)
        if m2:
            client_name = normalize_client_name(m2.group("client").strip())
            thread = m2.group("thread").strip()
            return EmailClassification(is_project_email=True, client_name=client_name, side="client", thread=thread)
        return EmailClassification(is_project_email=False)

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

async def save_tokens(email:str, creds:Credentials):
    token_data = {
        "email": email,
        "access_token": creds.token,
        "refresh_token": creds.refresh_token or "",
        "token_expiry": creds.expiry,
        "scope": " ".join(creds.scopes or []),
        "updated_at": dt.datetime.utcnow()
    }
    await google_oauth_col.update_one(
        {"email": email},
        {"$set": token_data},
        upsert=True
    )

async def load_tokens(email:str) -> Optional[Credentials]:
    doc = await google_oauth_col.find_one({"email": email})
    if not doc:
        return None
    
    creds = Credentials(
        token=doc["access_token"],
        refresh_token=doc.get("refresh_token", ""),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=(doc.get("scope", "") or "").split()
    )
    if doc.get("token_expiry"):
        creds.expiry = doc["token_expiry"]
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
    return JSONResponse({"auth_url": auth_url, "state": state})

@app.get("/oauth/google/callback")
async def oauth_callback(code: str, state: Optional[str] = None):
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
    await save_tokens(email_addr, creds)
    return RedirectResponse(url=f"/gmail/watch?email={email_addr}")

# -------------------- Start/stop Gmail watch --------------------
@app.get("/gmail/watch")
async def gmail_watch(email: str):
    creds = await load_tokens(email)
    if not creds: 
        raise HTTPException(400, "Connect Gmail first.")
    svc = gmail_service(creds)

    body = {"topicName": GMAIL_TOPIC}
    if GMAIL_LABEL_IDS:
        body["labelIds"] = GMAIL_LABEL_IDS
    resp = svc.users().watch(userId="me", body=body).execute()

    history_id = resp.get("historyId")
    await gmail_state_col.update_one(
        {"email": email},
        {"$set": {
            "history_id": int(history_id) if history_id else None,
            "watch_active": True,
            "updated_at": dt.datetime.utcnow()
        }},
        upsert=True
    )
    return {"ok": True, "email": email, "history_id": history_id}

@app.get("/gmail/stop")
async def gmail_stop(email: str):
    creds = await load_tokens(email)
    if not creds: 
        raise HTTPException(400, "Connect Gmail first.")
    svc = gmail_service(creds)
    svc.users().stop(userId="me").execute()
    
    await gmail_state_col.update_one(
        {"email": email},
        {"$set": {"watch_active": False, "updated_at": dt.datetime.utcnow()}}
    )
    return {"ok": True}

# -------------------- Pub/Sub push endpoint --------------------
class PubSubPush(BaseModel):
    message: Dict
    subscription: Optional[str] = None

@app.post("/gmail/push")
async def gmail_push(request: Request):
    body = await request.json()
    msg = body.get("message", {})
    data_b64 = msg.get("data", "")
    if not data_b64:
        return {"ok": True}

    data = json.loads(base64.b64decode(data_b64))
    email = data.get("emailAddress")
    history_id = int(data.get("historyId", 0))
    if not email or not history_id:
        return {"ok": True}

    await process_history(email, history_id)
    return {"ok": True}

# -------------------- Process Gmail history & create tasks --------------------
async def process_history(email: str, new_history_id: int):
    creds = await load_tokens(email)
    if not creds: 
        return
    svc = gmail_service(creds)

    # find last history id
    state_doc = await gmail_state_col.find_one({"email": email})
    last_history = state_doc.get("history_id") if state_doc else None

    # If we have no cursor, just set it; avoid backfilling everything.
    if not last_history:
        await gmail_state_col.update_one(
            {"email": email},
            {"$set": {"history_id": new_history_id, "updated_at": dt.datetime.utcnow()}},
            upsert=True
        )
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
        if not page_token: 
            break

    # advance cursor
    await gmail_state_col.update_one(
        {"email": email},
        {"$set": {"history_id": new_history_id, "updated_at": dt.datetime.utcnow()}}
    )

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

    to_raw = headers.get("to", "")
    cc_raw = headers.get("cc", "")

    # Clean noisy prefixes for better downstream parsing
    subject_clean = clean_subject(subject)

    # AI classification to unify projects by CLIENT and side (client/dev)
    classification = await ai_classify_email(subject_clean, body_text or body_html or "", headers)

    # If it's clearly not a real project email, skip processing entirely
    if not classification.is_project_email:
        return

    # Body (plain/text preferred; else html)
    body_text, body_html = extract_bodies(msg.get("payload", {}))

    # Parse subject (regex fallback) and merge with AI classification
    pieces = parse_subject(subject_clean)

    client_name = normalize_client_name(classification.client_name or pieces.get("client"))
    side = classification.side or ("dev" if pieces.get("kind") == "coding" else "client")
    thread_title = (classification.thread or pieces.get("thread") or subject_clean).strip()
    coding_app = classification.coding_app_name or pieces.get("app")

    if not client_name:
        # If we still can't determine a client, do not create a project/thread
        return

    # Upsert a single unified project per client (name = client display name)
    project_id = await upsert_project(
        name=client_name,
        app=coding_app,
        client=client_name,
        kind="unified",
    )

    # Upsert thread under that unified project
    thread_id = await upsert_thread(project_id, thread_title)

    # Insert email
    email_doc = {
        "thread_id": thread_id,
        "gmail_message_id": msg_id,
        "from_name": from_name,
        "from_email": from_email,
        "subject": subject,
        "body_html": body_html,
        "body_text": body_text,
        "received_at": received_at,
        "created_at": dt.datetime.utcnow()
    }
    email_doc.update({
        "to": to_raw,
        "cc": cc_raw,
        "client_name": client_name,
        "side": side,
        "addressed_to_me": bool(re.search(r"\b(zac(hary)?\s+cheshire)\b", f"{to_raw} {cc_raw}", re.IGNORECASE)),
    })
    email_result = await emails_col.insert_one(email_doc)
    email_id = email_result.inserted_id

    # AI extract → tasks
    extraction = await ai_extract_tasks(subject, body_text or "")
    for t in extraction.tasks:
        due = None
        if t.due_on:
            try: 
                due = dt.datetime.fromisoformat(t.due_on)
            except: 
                pass
        
        task_doc = {
            "thread_id": thread_id,
            "title": t.title,
            "description": t.description,
            "priority": t.priority or "normal",
            "due_on": due,
            "assignee_hint": t.assignee_hint,
            "status": "open",
            "source_email_id": email_id,
            "created_at": dt.datetime.utcnow()
        }
        task_doc.update({
            "side": side,  # client or dev
            "client_name": client_name,
            "addressed_to_me": email_doc.get("addressed_to_me", False),
        })
        await tasks_col.insert_one(task_doc)

async def upsert_project(name: str, app: Optional[str], client: Optional[str], kind: str):
    nk = slugify(name)
    project_doc = {
        "name": name,
        "normalized_key": nk,
        "app_name": app,
        "client_name": client,
        "kind": kind,
        "created_at": dt.datetime.utcnow()
    }
    
    result = await projects_col.update_one(
        {"normalized_key": nk},
        {"$set": project_doc},
        upsert=True
    )
    
    if result.upserted_id:
        return result.upserted_id
    else:
        doc = await projects_col.find_one({"normalized_key": nk})
        return doc["_id"]

async def upsert_thread(project_id, subject: str):
    thread_doc = {
        "project_id": project_id,
        "subject": subject,
        "created_at": dt.datetime.utcnow()
    }
    
    try:
        result = await threads_col.insert_one(thread_doc)
        return result.inserted_id
    except DuplicateKeyError:
        doc = await threads_col.find_one({"project_id": project_id, "subject": subject})
        return doc["_id"]

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

# -------------------- API endpoints for tasks --------------------
@app.get("/api/tasks")
async def get_tasks():
    pipeline = [
        {
            "$lookup": {
                "from": "threads",
                "localField": "thread_id",
                "foreignField": "_id",
                "as": "thread"
            }
        },
        {"$unwind": "$thread"},
        {
            "$lookup": {
                "from": "projects",
                "localField": "thread.project_id",
                "foreignField": "_id",
                "as": "project"
            }
        },
        {"$unwind": "$project"},
        {
            "$project": {
                "task_id": "$_id",
                "title": 1,
                "description": 1,
                "priority": 1,
                "due_on": 1,
                "status": 1,
                "assignee_hint": 1,
                "created_at": 1,
                "thread_id": "$thread._id",
                "thread_subject": "$thread.subject",
                "project_id": "$project._id",
                "project_name": "$project.name",
                "kind": "$project.kind",
                "client_name": 1,
                "app_name": "$project.app_name",
                "side": 1,
                "addressed_to_me": 1,
            }
        },
        {
            "$sort": {
                "project_name": 1,
                "thread_subject": 1,
                "created_at": -1
            }
        }
    ]
    
    tasks = []
    async for doc in tasks_col.aggregate(pipeline):
        # Convert ObjectId to string for JSON serialization
        doc["task_id"] = str(doc["task_id"])
        doc["thread_id"] = str(doc["thread_id"])
        doc["project_id"] = str(doc["project_id"])
        if doc.get("due_on"):
            doc["due_on"] = doc["due_on"].isoformat()
        if doc.get("created_at"):
            doc["created_at"] = doc["created_at"].isoformat()
        tasks.append(doc)
    
    return tasks