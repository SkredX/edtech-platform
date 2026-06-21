"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { ChatInput, ChatInputSubmit, ChatInputTextArea } from "@/components/ui/chat-input";
import { Card, CardContent } from "@/components/ui/card";
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

  useEffect(() => {
    apiGet<DocumentInfo[]>("/ingest/documents")
      .then(setDocuments)
      .catch(() => {
        // Silent: chat still works searching the whole corpus without a
        // chapter picker if this fails (e.g. no documents ingested yet).
      })
      .finally(() => setDocsLoading(false));
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
                {doc.label}
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
            <CardContent className="p-3 text-sm">
              {m.content}
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
