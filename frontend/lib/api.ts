const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function getApiKey(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem("skredx_edtech_api_key") ?? "";
}

export function setApiKey(key: string) {
  if (typeof window !== "undefined") {
    localStorage.setItem("skredx_edtech_api_key", key);
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "GET",
    headers: {
      "X-API-Key": getApiKey(),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": getApiKey(),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: {
      "X-API-Key": getApiKey(),
    },
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `Upload failed: ${res.status}`);
  }
  return res.json();
}
