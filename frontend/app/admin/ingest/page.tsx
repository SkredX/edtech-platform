"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { apiUpload } from "@/lib/api";

interface IngestResponse {
  document_name: string;
  ingested_at_utc: string;
  chunks_ingested: number;
  clusters: number;
  groq_calls_used: number;
}

export default function IngestPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);

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
