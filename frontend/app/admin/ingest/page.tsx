"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { apiPost, apiUpload } from "@/lib/api";

interface IngestResponse {
  document_name: string;
  ingested_at_utc: string;
  chunks_ingested: number;
  clusters: number;
  groq_calls_used: number;
}

interface BackfillResponse {
  backfilled_documents: string[];
  already_registered: string[];
  total_distinct_documents_in_pinecone: number;
}

export default function IngestPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [backfilling, setBackfilling] = useState(false);
  const [backfillResult, setBackfillResult] = useState<BackfillResponse | null>(null);

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

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      toast.error("Only PDF files are supported.");
      return;
    }

    setUploading(true);
    setResult(null);
    try {
      const data = await apiUpload<IngestResponse>("/ingest", file);
      setResult(data);
      toast.success(`Ingested ${data.chunks_ingested} chunks across ${data.clusters} topics.`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Ingestion failed.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6 space-y-6">
      <header>
        <h1 className="text-lg font-semibold">Document Ingestion</h1>
        <p className="text-sm text-muted-foreground">
          Upload course material (PDF). Chunked, locally embedded, clustered, and summarized —
          only one Groq call per topic cluster, not per chunk.
        </p>
      </header>

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
          {uploading && <p className="text-sm text-muted-foreground">Processing — this can take a few minutes for large documents…</p>}
        </CardContent>
      </Card>

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

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{result.document_name}</CardTitle>
            <CardDescription>Ingested {new Date(result.ingested_at_utc).toLocaleString()}</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-3 gap-4 text-sm">
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
          </CardContent>
          <CardContent className="flex gap-3 pt-0">
            <Button asChild className="flex-1">
              <Link href="/chat">Ask a doubt about this</Link>
            </Button>
            <Button asChild variant="outline" className="flex-1">
              <Link href="/admin/clonegen">Generate questions</Link>
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
