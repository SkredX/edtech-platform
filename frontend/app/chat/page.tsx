"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { ChatInput, ChatInputSubmit, ChatInputTextArea } from "@/components/ui/chat-input";
import { Card, CardContent } from "@/components/ui/card";
import { MarkdownMessage } from "@/components/ui/markdown-message";
import { apiGet, apiPost } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  escalate?: boolean;
}

interface ChatApiResponse {
  answer: string;
  escalate: boolean;
  from_cache: boolean;
}

interface DocumentInfo {
  document_name: string;
  grade: string | null;
  subject_code: string | null;
  subject_label: string | null;
  chapter: string | null;
  label: string;
  chunk_count: number;
}

export default function ChatPage() {
  const [value, setValue] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());

  // Prevent the auto-backfill from firing twice in React Strict Mode
  const backfillRan = useRef(false);

  useEffect(() => {
    if (backfillRan.current) return;
    backfillRan.current = true;

    /**
     * On every page load:
     * 1. POST /ingest/backfill-registry — silently recovers any chapters
     *    that were uploaded before the Redis registry existed (or after a
     *    Redis flush). This is idempotent (already-registered docs are
     *    untouched), costs zero Groq/embedding calls, and is fast enough
     *    (~9 Pinecone fetch calls for a ~900-chunk corpus) to run quietly
     *    in the background before the GET /ingest/documents resolves.
     * 2. GET /ingest/documents — populates the chapter picker with the
     *    now-complete registry.
     *
     * Running backfill first means the chapter list is always in sync with
     * Pinecone even for documents uploaded before this registry existed,
     * without requiring any admin action.
     */
    const loadDocuments = async () => {
      try {
        // Step 1: silent background backfill (no toast on success)
        await apiPost("/ingest/backfill-registry", {});
      } catch {
        // Non-fatal: the chapter picker still works for documents that
        // were registered normally; missing ones just won't appear yet.
      }

      // Step 2: fetch the (now-complete) document list
      try {
        const docs = await apiGet<DocumentInfo[]>("/ingest/documents");
        setDocuments(docs);
      } catch {
        // Silent: chat still works without a chapter picker
      } finally {
        setDocsLoading(false);
      }
    };

    loadDocuments();
  }, []);

  const toggleDoc = (documentName: string) => {
    setSelectedDocs((prev) => {
      const next = new Set(prev);
      if (next.has(documentName)) {
        next.delete(documentName);
      } else {
        next.add(documentName);
      }
      return next;
    });
  };

  const clearDocs = () => setSelectedDocs(new Set());

  const handleSubmit = async () => {
    if (!value.trim()) return;
    const userMsg: Message = { role: "user", content: value };
    setMessages((m) => [...m, userMsg]);
    setValue("");
    setLoading(true);

    try {
      const data = await apiPost<ChatApiResponse>("/chat", {
        message: userMsg.content,
        document_names: selectedDocs.size > 0 ? Array.from(selectedDocs) : null,
      });
      setMessages((m) => [
        ...m,
        { role: "assistant", content: data.answer, escalate: data.escalate },
      ]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4 gap-4">
      <header className="pb-2 border-b">
        <h1 className="text-lg font-semibold">Doubt Assistant</h1>
        <p className="text-sm text-muted-foreground">Answers grounded in your course material only.</p>
      </header>

      {documents.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 pb-1">
          <span className="text-xs text-muted-foreground mr-1">Chapters:</span>
          {documents.map((doc) => {
            const isSelected = selectedDocs.has(doc.document_name);
            // Use the clean parsed chapter name when available, fall back to
            // the full label (which is already better than the raw filename).
            const displayName = doc.chapter ?? doc.label;
            return (
              <button
                key={doc.document_name}
                type="button"
                onClick={() => toggleDoc(doc.document_name)}
                title={`${doc.chunk_count} chunks`}
                aria-pressed={isSelected}
                className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background ${
                  isSelected
                    ? "bg-success text-success-foreground hover:bg-success/90"
                    : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
                }`}
              >
                {displayName}
                {isSelected && <span aria-hidden="true">×</span>}
              </button>
            );
          })}
          {selectedDocs.size > 0 && (
            <button
              type="button"
              onClick={clearDocs}
              className="text-xs text-muted-foreground underline-offset-2 hover:underline ml-1"
            >
              Clear ({selectedDocs.size})
            </button>
          )}
        </div>
      )}

      <div className="flex-1 overflow-y-auto space-y-3">
        {messages.length === 0 && !docsLoading && documents.length === 0 && (
          <p className="text-sm text-muted-foreground py-8 text-center">
            Ask anything about your syllabus to get started.
          </p>
        )}
        {messages.length === 0 && documents.length > 0 && (
          <p className="text-sm text-muted-foreground py-8 text-center">
            {selectedDocs.size > 0
              ? `Searching ${selectedDocs.size} selected chapter${selectedDocs.size > 1 ? "s" : ""} — ask away, or clear the selection to widen the search.`
              : "Pick one or more chapters above to narrow your search, or ask anything to search everything."}
          </p>
        )}
        {messages.map((m, i) => (
          <Card
            key={i}
            className={m.role === "user" ? "ml-auto bg-primary/10 max-w-[80%]" : "max-w-[80%]"}
          >
            <CardContent className="p-3">
              <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                {m.role === "user" ? "You" : "Doubt Assistant"}
              </p>
              {m.role === "assistant" ? (
                <MarkdownMessage content={m.content} />
              ) : (
                <p className="text-sm leading-relaxed">{m.content}</p>
              )}
              {m.escalate && (
                <p className="mt-2 text-xs text-amber-600">
                  This was escalated — your teacher will follow up.
                </p>
              )}
            </CardContent>
          </Card>
        ))}
        {loading && (
          <p className="text-xs text-muted-foreground">Thinking…</p>
        )}
      </div>

      <ChatInput
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onSubmit={handleSubmit}
        loading={loading}
      >
        <ChatInputTextArea
          placeholder={
            selectedDocs.size > 0
              ? `Ask a doubt about ${selectedDocs.size} selected chapter${selectedDocs.size > 1 ? "s" : ""}...`
              : "Ask a doubt about your course..."
          }
        />
        <ChatInputSubmit />
      </ChatInput>
    </div>
  );
}
