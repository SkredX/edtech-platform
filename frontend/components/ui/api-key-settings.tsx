"use client";

import { useEffect, useState } from "react";
import { Settings, X, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getApiKey, setApiKey } from "@/lib/api";

/**
 * Floating settings widget so anyone using the deployed site (or anyone
 * who deploys this for a different institute) can paste their backend's
 * API key without opening devtools/localStorage manually.
 *
 * The key is the same one configured server-side in TENANT_API_KEYS
 * (e.g. TENANT_API_KEYS=mySecretDevKey123:demo-tenant -> paste
 * "mySecretDevKey123" here). It's stored in this browser's localStorage
 * only — never sent anywhere except as the X-API-Key header on requests
 * to your backend.
 */
export function ApiKeySettings() {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState(false);
  const [hasKey, setHasKey] = useState(true);

  useEffect(() => {
    const existing = getApiKey();
    setValue(existing);
    setHasKey(Boolean(existing));
    if (!existing) setOpen(true); // nudge first-time visitors to set it
  }, []);

  function handleSave() {
    setApiKey(value.trim());
    setHasKey(Boolean(value.trim()));
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {open && (
        <div className="mb-3 w-80 rounded-xl border border-border bg-card p-4 shadow-xl">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-sm font-medium text-card-foreground">API Key</p>
            <button
              onClick={() => setOpen(false)}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <p className="mb-3 text-xs text-muted-foreground">
            Enter the API key your institute was given. This is stored
            only in this browser.
          </p>
          <Label htmlFor="api-key-input" className="sr-only">API Key</Label>
          <Input
            id="api-key-input"
            type="password"
            placeholder="Paste your API key"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            className="mb-3"
          />
          <Button onClick={handleSave} className="w-full" size="sm">
            {saved ? (
              <span className="flex items-center gap-1">
                <Check className="h-4 w-4" /> Saved
              </span>
            ) : (
              "Save"
            )}
          </Button>
        </div>
      )}
      <Button
        size="icon"
        variant={hasKey ? "outline" : "default"}
        className="rounded-full shadow-lg"
        onClick={() => setOpen((o) => !o)}
        aria-label="API key settings"
      >
        <Settings className="h-4 w-4" />
      </Button>
    </div>
  );
}
