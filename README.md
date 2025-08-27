# Gmail to AI Task Board

This project converts Gmail emails into actionable tasks using AI, organizing them by project and thread.

## Features

- **Gmail Integration**: OAuth-based Gmail connection with push notifications
- **Smart Subject Parsing**: Recognizes Basecamp-style subjects like `($coding- App (Client)) Thread`
- **AI Task Extraction**: Uses OpenAI to extract actionable tasks from emails
- **Project Organization**: Groups tasks by projects and threads
- **Real-time Updates**: Pub/Sub push notifications for instant email processing
- **Task Board UI**: Clean Next.js interface to view and manage tasks

## Setup Instructions

### Prerequisites

1. Google Cloud Project with:
   - Gmail API enabled
   - Pub/Sub API enabled
   - OAuth 2.0 credentials (Web application type)

2. PostgreSQL database (Neon, Supabase, or any PostgreSQL instance)

3. OpenAI API key

### 1. Database Setup

Run the SQL migrations:

```bash
psql $DATABASE_URL < migrations.sql
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_KEY`: Your OpenAI API key
- `GOOGLE_CLIENT_ID`: OAuth client ID from Google Cloud Console
- `GOOGLE_CLIENT_SECRET`: OAuth client secret
- `OAUTH_REDIRECT_URI`: Must match authorized redirect URI in Google Console
- `GMAIL_TOPIC`: Pub/Sub topic name (format: `projects/YOUR_PROJECT/topics/gmail-push`)

### 3. Google Cloud Setup

1. **Create OAuth Client**:
   - Type: Web application
   - Authorized redirect URI: `https://YOUR_API_HOST/oauth/google/callback`

2. **Create Pub/Sub Topic**:
   - Name: `gmail-push`
   - Grant Gmail service account Publisher role

3. **Create Push Subscription**:
   - Topic: `gmail-push`
   - Push endpoint: `https://YOUR_API_HOST/gmail/push`

### 4. Install Dependencies

Backend (Python):
```bash
pip install -r requirements.txt
```

Frontend (Next.js):
```bash
npm install
```

### 5. Run the Application

Start the FastAPI backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Start the Next.js frontend (in another terminal):
```bash
npm run dev
```

### 6. Connect Gmail

1. Navigate to `http://localhost:3000`
2. Click "Connect Gmail Account"
3. Authorize the application
4. The Gmail watch will be activated automatically

## How It Works

1. **Email Reception**: Gmail sends push notifications via Pub/Sub when new emails arrive
2. **Subject Parsing**: Extracts project, client, app, and thread from email subjects
3. **AI Processing**: OpenAI extracts actionable tasks with priorities and due dates
4. **Database Storage**: Organizes emails and tasks by project and thread
5. **Task Board**: Next.js UI displays tasks grouped by project and thread

## Subject Format Examples

- **Coding Project**: `($coding- MediApp (Jason Bill)) Credentials`
  - Project: "$coding- MediApp (Jason Bill)"
  - Thread: "Credentials"
  
- **Client Project**: `(Acme Corp) Budget Review`
  - Project: "Acme Corp"
  - Thread: "Budget Review"

## API Endpoints

- `GET /oauth/google/start`: Initiate OAuth flow
- `GET /oauth/google/callback`: OAuth callback
- `GET /gmail/watch`: Activate Gmail push notifications
- `GET /gmail/stop`: Deactivate Gmail push
- `POST /gmail/push`: Receive Pub/Sub notifications
- `GET /api/tasks`: Fetch all tasks (Next.js API)

## Notes

- Gmail watch expires after 7 days; set up a cron job to refresh
- Tasks are extracted only from new emails after setup
- The system avoids backfilling historical emails by default