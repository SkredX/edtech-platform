// frontend/app/admin/ingest/page.tsx
"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { apiPost, apiUpload, apiGet } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ClusterTopic {
  cluster_id: number;
  topic: string;
  summary: string;
  chunk_count: number;
}

interface IngestResult {
  document_name: string;
  ingested_at_utc: string;
  chunks_ingested: number;
  clusters: number;
  groq_calls_used: number;
  topics: ClusterTopic[];
}

interface JobStatus {
  job_id: string;
  status: "running" | "done" | "error";
  step: string;
  progress: number;
  detail: string;
  result: IngestResult | null;
}

interface BackfillResponse {
  backfilled_documents: string[];
  already_registered: string[];
  total_distinct_documents_in_pinecone: number;
}

// ── Progress bar (no external dep) ───────────────────────────────────────────

function ProgressBar({ value, className = "" }: { value: number; className?: string }) {
  return (
    <div className={`h-2 w-full rounded-full bg-muted overflow-hidden ${className}`}>
      <div
        className="h-full rounded-full bg-primary transition-all duration-500 ease-out"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function IngestPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Upload / progress state
  const [uploading, setUploading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Backfill state
  const [backfilling, setBackfilling] = useState(false);
  const [backfillResult, setBackfillResult] = useState<BackfillResponse | null>(null);

  // ── Polling ────────────────────────────────────────────────────────────────

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearTimeout(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(
    async (id: string) => {
      try {
        const status = await apiGet<JobStatus>(`/ingest/status/${id}`);
        setJobStatus(status);

        if (status.status === "done") {
          setUploading(false);
          stopPolling();
          toast.success(
            `Ingested ${status.result?.chunks_ingested} chunks across ${status.result?.clusters} topic clusters.`
          );
        } else if (status.status === "error") {
          setUploading(false);
          stopPolling();
          toast.error(`Ingestion failed: ${status.detail}`);
        } else {
          // Still running — poll again in 2 s. No faster: Redis + network
          // round-trip takes ~200ms and the long step (embedding) takes 30-90s.
          pollingRef.current = setTimeout(() => pollStatus(id), 2000);
        }
      } catch {
        // Transient network error — keep polling; don't surface to the user
        // unless it persists (the error branch above handles fatal failures).
        pollingRef.current = setTimeout(() => pollStatus(id), 3000);
      }
    },
    [stopPolling]
  );

  // Clean up polling on unmount
  useEffect(() => () => stopPolling(), [stopPolling]);

  // ── Upload handler ─────────────────────────────────────────────────────────

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      toast.error("Only PDF files are supported.");
      return;
    }

    stopPolling();
    setUploading(true);
    setJobId(null);
    setJobStatus(null);

    try {
      const { job_id } = await apiUpload<{ job_id: string }>("/ingest/async", file);
      setJobId(job_id);
      // Kick off polling immediately
      pollStatus(job_id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed.");
      setUploading(false);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // ── Backfill handler ───────────────────────────────────────────────────────

  const handleBackfill = async () => {
    setBackfilling(true);
    setBackfillResult(null);
    try {
      const data = await apiPost<BackfillResponse>("/ingest/backfill-registry", {});
      setBackfillResult(data);
      if (data.backfilled_documents.length > 0) {
        toast.success(
          `Recovered ${data.backfilled_documents.length} chapter${data.backfilled_documents.length > 1 ? "s" : ""} from Pinecone.`
        );
      } else {
        toast.success("Everything in Pinecone is already in the chapter picker — nothing to recover.");
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Resync failed.");
    } finally {
      setBackfilling(false);
    }
  };

  // ── Derived render values ─────────────────────────────────────────────────

  const isRunning = uploading && jobStatus?.status === "running";
  const isDone = jobStatus?.status === "done";
  const isError = jobStatus?.status === "error";
  const result = jobStatus?.result ?? null;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-xl mx-auto p-6 space-y-6">
      <header>
        <h1 className="text-lg font-semibold">Document Ingestion</h1>
        <p className="text-sm text-muted-foreground">
          Upload course material (PDF). Chunked, locally embedded, clustered, and summarized —
          only one Groq call per topic cluster, not per chunk.
        </p>
      </header>

      {/* ── Upload card ── */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Upload PDF</CardTitle>
          <CardDescription>300+ page documents are supported.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            disabled={uploading}
            className="block w-full text-sm file:mr-4 file:rounded-lg file:border-0 file:bg-primary file:px-4 file:py-2 file:text-sm file:font-medium file:text-primary-foreground hover:file:bg-primary/90 disabled:opacity-50"
          />

          {/* Progress section — shown while running or on error */}
          {(isRunning || isError || (uploading && !jobStatus)) && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">
                  {jobStatus?.step ?? "Uploading…"}
                </span>
                {jobStatus && (
                  <span className="tabular-nums font-medium text-xs text-muted-foreground">
                    {jobStatus.progress}%
                  </span>
                )}
              </div>
              <ProgressBar value={jobStatus?.progress ?? 0} />
              {jobStatus?.detail && (
                <p className="text-xs text-muted-foreground">{jobStatus.detail}</p>
              )}
              {isError && (
                <p className="text-xs text-destructive">
                  {jobStatus?.detail ?? "Ingestion failed."}
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── Result card — shown once done ── */}
      {isDone && result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{result.document_name}</CardTitle>
            <CardDescription>
              Ingested {new Date(result.ingested_at_utc).toLocaleString()}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {/* Stats row */}
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Chunks</p>
                <p className="text-lg font-medium">{result.chunks_ingested}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Topic clusters</p>
                <p className="text-lg font-medium">{result.clusters}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Groq calls used</p>
                <p className="text-lg font-medium">{result.groq_calls_used}</p>
              </div>
            </div>

            {/* Cluster topic preview */}
            {result.topics && result.topics.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Extracted topic clusters</p>
                <p className="text-xs text-muted-foreground">
                  Review these to spot mis-parsed or garbled sections before adding to the question bank.
                </p>
                <div className="max-h-72 overflow-y-auto rounded-md border divide-y text-sm">
                  {result.topics.map((t) => (
                    <div key={t.cluster_id} className="px-3 py-2 space-y-0.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium truncate">{t.topic}</span>
                        <span className="shrink-0 text-xs text-muted-foreground">
                          {t.chunk_count} chunk{t.chunk_count !== 1 ? "s" : ""}
                        </span>
                      </div>
                      {t.summary && (
                        <p className="text-xs text-muted-foreground line-clamp-2">{t.summary}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-3 pt-1">
              <Button asChild className="flex-1">
                <Link href="/chat">Ask a doubt about this</Link>
              </Button>
              <Button asChild variant="outline" className="flex-1">
                <Link href="/admin/clonegen">Generate questions</Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Backfill card (unchanged) ── */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Resync chapter picker</CardTitle>
          <CardDescription>
            If a chapter exists in Pinecone but isn&apos;t showing up on the{" "}
            <Link href="/chat" className="underline">
              Ask a Doubt
            </Link>{" "}
            page — for example, chapters uploaded before this picker existed — run this to
            recover them. Reads metadata already in Pinecone; no Groq or embedding calls.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button onClick={handleBackfill} disabled={backfilling} variant="outline">
            {backfilling ? "Scanning Pinecone…" : "Resync from Pinecone"}
          </Button>
          {backfillResult && (
            <div className="text-sm space-y-1">
              <p>
                <span className="text-muted-foreground">Distinct chapters found in Pinecone: </span>
                <span className="font-medium">{backfillResult.total_distinct_documents_in_pinecone}</span>
              </p>
              <p>
                <span className="text-muted-foreground">Already in picker: </span>
                <span className="font-medium">{backfillResult.already_registered.length}</span>
              </p>
              {backfillResult.backfilled_documents.length > 0 ? (
                <div>
                  <p className="text-muted-foreground">Recovered just now:</p>
                  <ul className="ml-4 list-disc">
                    {backfillResult.backfilled_documents.map((name) => (
                      <li key={name}>{name}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-muted-foreground">Nothing missing — the picker is already in sync.</p>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}