import { NextResponse } from "next/server";
import { Pool } from "pg";

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

export async function GET() {
  const { rows } = await pool.query(`
    select 
      p.id as project_id, p.name as project_name, p.kind, p.client_name, p.app_name,
      th.id as thread_id, th.subject as thread_subject,
      t.id as task_id, t.title, t.description, t.priority, t.due_on, t.status, t.assignee_hint, t.created_at
    from tasks t
    join threads th on th.id = t.thread_id
    join projects p on p.id = th.project_id
    order by p.name, th.subject, t.created_at desc
  `);
  return NextResponse.json(rows);
}