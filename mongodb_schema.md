# MongoDB Collections Schema

## Collections Overview

The application uses MongoDB with the following collections:

### 1. `projects`
Stores project information parsed from email subjects.

```javascript
{
  _id: ObjectId,
  name: String,              // Full project name
  normalized_key: String,    // Unique slug for deduplication
  app_name: String,          // App name for coding projects
  client_name: String,       // Client name
  kind: String,              // "coding" or "client"
  created_at: Date
}
```

**Indexes:**
- `normalized_key` (unique)

### 2. `threads`
Email threads within projects.

```javascript
{
  _id: ObjectId,
  project_id: ObjectId,      // Reference to projects collection
  subject: String,           // Thread subject
  created_at: Date
}
```

**Indexes:**
- `[project_id, subject]` (compound unique)

### 3. `emails`
Individual email messages.

```javascript
{
  _id: ObjectId,
  thread_id: ObjectId,       // Reference to threads collection
  gmail_message_id: String,  // Gmail message ID
  from_name: String,
  from_email: String,
  subject: String,
  body_html: String,
  body_text: String,
  received_at: Date,
  created_at: Date
}
```

**Indexes:**
- `thread_id`
- `gmail_message_id`

### 4. `tasks`
AI-extracted tasks from emails.

```javascript
{
  _id: ObjectId,
  thread_id: ObjectId,       // Reference to threads collection
  title: String,
  description: String,
  priority: String,          // "low", "normal", "high", "urgent"
  due_on: Date,
  assignee_hint: String,
  status: String,            // "open", "in_progress", "done", "blocked"
  source_email_id: ObjectId, // Reference to emails collection
  created_at: Date
}
```

**Indexes:**
- `thread_id`

### 5. `google_oauth`
OAuth tokens for Gmail access.

```javascript
{
  _id: ObjectId,
  email: String,             // User email (unique)
  access_token: String,
  refresh_token: String,
  token_expiry: Date,
  scope: String,
  updated_at: Date
}
```

**Indexes:**
- `email` (unique)

### 6. `gmail_state`
Gmail sync state tracking.

```javascript
{
  _id: ObjectId,
  email: String,             // User email (unique)
  history_id: Number,        // Last synced Gmail history ID
  watch_active: Boolean,
  updated_at: Date
}
```

**Indexes:**
- `email` (unique)

## Data Flow

1. **Gmail Push → Email Processing**
   - Gmail sends push notification with history_id
   - System fetches new messages since last history_id
   - Emails are parsed and stored

2. **Email → Project/Thread Organization**
   - Subject is parsed for project and thread info
   - Projects are upserted by normalized_key
   - Threads are created under projects

3. **Email → Task Extraction**
   - AI processes email content
   - Tasks are created and linked to threads
   - Tasks maintain reference to source email

## Migration from PostgreSQL

The MongoDB implementation maintains the same logical structure as the PostgreSQL version but uses:
- ObjectId instead of UUID
- Document embedding where appropriate
- Aggregation pipeline for complex queries
- Async motor driver for non-blocking operations