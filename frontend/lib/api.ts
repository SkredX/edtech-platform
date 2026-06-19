const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function getApiKey(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem("speedlabs_api_key") ?? "";
}

export function setApiKey(key: string) {
  if (typeof window !== "undefined") {
    localStorage.setItem("speedlabs_api_key", key);
  }
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
