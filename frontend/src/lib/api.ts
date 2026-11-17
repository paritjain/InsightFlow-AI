const getApiBase = () => {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== 'undefined') {
    const host = (window.location.hostname === 'localhost' || !window.location.hostname)
      ? '127.0.0.1'
      : window.location.hostname;
    return `http://${host}:8000`;
  }
  return 'http://127.0.0.1:8000';
};

const API_BASE = getApiBase();

export async function fetchProjects() {
  const res = await fetch(`${API_BASE}/api/projects`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch projects');
  return res.json();
}

export async function triggerDemoProject() {
  const res = await fetch(`${API_BASE}/api/projects/demo`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to start demo project');
  return res.json();
}

export async function fetchProjectDetails(projId: string) {
  const res = await fetch(`${API_BASE}/api/project/${projId}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch project details');
  return res.json();
}

export async function applyCleaningRules(projId: string, approvedRules: any[]) {
  const res = await fetch(`${API_BASE}/api/project/${projId}/clean`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved_rules: approvedRules }),
  });
  if (!res.ok) throw new Error('Failed to apply cleaning rules');
  return res.json();
}

export async function askYourData(projId: string, question: string, confirmedField?: string) {
  const payload: any = { question };
  if (confirmedField) payload.confirmed_field = confirmedField;
  const res = await fetch(`${API_BASE}/api/project/${projId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Failed to query data');
  return res.json();
}

export async function overrideSemanticRole(projId: string, columnName: string, semanticRole?: string, semanticSubtype?: string) {
  const res = await fetch(`${API_BASE}/api/project/${projId}/semantic-override`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ column_name: columnName, semantic_role: semanticRole, semantic_subtype: semanticSubtype }),
  });
  if (!res.ok) throw new Error('Failed to override semantic role');
  return res.json();
}

export async function updateActionStatus(
  projId: string,
  actionId: string,
  payloadOrStatus: string | {
    status?: string;
    owner?: string;
    due_date?: string;
    target?: string;
    completion_notes?: string;
    verification_notes?: string;
  }
) {
  const body = typeof payloadOrStatus === 'string'
    ? { action_id: actionId, status: payloadOrStatus }
    : { action_id: actionId, ...payloadOrStatus };
  const res = await fetch(`${API_BASE}/api/project/${projId}/action-status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error('Failed to update action');
  return res.json();
}

export async function getMasterPreview(projId: string, mode: string = 'master', limit: number = 1000) {
  const res = await fetch(`${API_BASE}/api/project/${projId}/master-preview?mode=${encodeURIComponent(mode)}&limit=${limit}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch master dataset preview');
  return res.json();
}

export function getExcelExportUrl(projId: string) {
  return `${API_BASE}/api/project/${projId}/export/excel`;
}

export function getPdfExportUrl(projId: string) {
  return `${API_BASE}/api/project/${projId}/export/pdf`;
}

export async function uploadProjectFiles(files: FileList | File[]) {
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/projects/upload`, {
      method: 'POST',
      body: formData,
    });
  } catch (netErr) {
    // Fallback through Next.js rewrite proxy if direct connection encounters network issues
    res = await fetch('/api/projects/upload', {
      method: 'POST',
      body: formData,
    });
  }

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed with server status ${res.status}`);
  }

  return res.json();
}

export async function sendChatMessage(projId: string, question: string, confirmedField?: string, selectionMode?: string) {
  const payload: any = { question };
  if (confirmedField) payload.confirmed_field = confirmedField;
  if (selectionMode) payload.selection_mode = selectionMode;
  const res = await fetch(`${API_BASE}/api/project/${projId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Failed to send chat message');
  return res.json();
}

export async function fetchChatHistory(projId: string) {
  const res = await fetch(`${API_BASE}/api/project/${projId}/chat-history`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch chat history');
  return res.json();
}

export async function clearChatHistory(projId: string) {
  const res = await fetch(`${API_BASE}/api/project/${projId}/chat-history`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear chat history');
  return res.json();
}
