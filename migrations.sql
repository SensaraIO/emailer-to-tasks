-- Projects/threads/emails/tasks (from our earlier design)
create extension if not exists pgcrypto;

create table if not exists projects (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  normalized_key text not null unique,
  client_name text,
  app_name text,
  kind text check (kind in ('coding','client')) default 'coding',
  created_at timestamptz default now()
);

create table if not exists threads (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references projects(id) on delete cascade,
  subject text not null,
  created_at timestamptz default now(),
  unique (project_id, subject)
);

create table if not exists emails (
  id uuid primary key default gen_random_uuid(),
  thread_id uuid not null references threads(id) on delete cascade,
  gmail_message_id text,
  from_name text,
  from_email text,
  subject text,
  body_html text,
  body_text text,
  received_at timestamptz default now()
);

create table if not exists tasks (
  id uuid primary key default gen_random_uuid(),
  thread_id uuid not null references threads(id) on delete cascade,
  title text not null,
  description text,
  priority text check (priority in ('low','normal','high','urgent')) default 'normal',
  due_on date,
  assignee_hint text,
  status text check (status in ('open','in_progress','done','blocked')) default 'open',
  source_email_id uuid references emails(id) on delete set null,
  created_at timestamptz default now()
);

-- Store Google tokens & Gmail sync cursor
create table if not exists google_oauth (
  id bigint primary key generated always as identity,
  email text not null unique,
  access_token text not null,
  refresh_token text not null,
  token_expiry timestamptz not null,
  scope text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists gmail_state (
  id bigint primary key generated always as identity,
  email text not null unique,
  history_id bigint,      -- last synced historyId
  watch_active boolean default false,
  updated_at timestamptz default now()
);