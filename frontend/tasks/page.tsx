"use client";
import useSWR from "swr";
const fetcher = (u:string)=>fetch(u).then(r=>r.json());

export default function TasksPage() {
  const { data } = useSWR("/api/tasks", fetcher, { refreshInterval: 15000 });

  const grouped = (data || []).reduce((acc:any, row:any)=>{
    acc[row.project_id] ||= { name: row.project_name, kind: row.kind, threads: {} };
    acc[row.project_id].threads[row.thread_id] ||= { subject: row.thread_subject, tasks: [] };
    acc[row.project_id].threads[row.thread_id].tasks.push(row);
    return acc;
  }, {});

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Task Board</h1>
      {Object.entries(grouped).map(([pid, proj]: any)=>(
        <div key={pid} className="bg-white rounded-2xl shadow p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-bold">{proj.name}</h2>
            <span className="text-xs px-2 py-1 rounded bg-gray-100">{proj.kind}</span>
          </div>
          <div className="grid md:grid-cols-3 gap-4">
            {Object.entries(proj.threads).map(([tid, th]: any)=>(
              <div key={tid} className="border rounded-xl p-3">
                <div className="font-semibold mb-2">{th.subject}</div>
                <div className="space-y-2">
                  {th.tasks.map((t:any)=>(
                    <div key={t.task_id} className="border rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <div className="font-medium">{t.title}</div>
                        <span className="text-xs">{t.priority}</span>
                      </div>
                      {t.due_on && <div className="text-xs mt-1">Due: {t.due_on}</div>}
                      {t.assignee_hint && <div className="text-xs">Assignee hint: {t.assignee_hint}</div>}
                      {t.description && <p className="text-sm mt-2 opacity-80">{t.description}</p>}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}