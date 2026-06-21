"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

interface MarkdownMessageProps {
  content: string;
  className?: string;
}

/**
 * Renders chat message content as Markdown instead of raw text.
 *
 * Why this exists: the Groq chat model is prompted (see backend
 * chatbot/router.py SYSTEM_PROMPT) to answer with real Markdown structure
 * — bold key terms, bullet lists for multi-part answers, a trailing
 * "Sources:" line. Without this component that Markdown was rendered as
 * literal asterisks and dashes in a plain <CardContent> div, which is why
 * answers looked like an unstructured wall of text. This component is the
 * other half of that fix: it actually renders the structure the model
 * is now reliably producing.
 *
 * The final "Sources: ..." line is detected and styled distinctly (smaller,
 * muted, separated by a hairline) so citations read as metadata rather
 * than part of the answer's body text.
 */
export function MarkdownMessage({ content, className }: MarkdownMessageProps) {
  const sourcesMatch = content.match(/\n?Sources?:\s*(.+)$/i);
  const body = sourcesMatch ? content.slice(0, sourcesMatch.index).trim() : content;
  const sourcesLine = sourcesMatch ? sourcesMatch[1].trim() : null;

  return (
    <div className={cn("text-sm leading-relaxed", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
          ul: ({ children }) => (
            <ul className="mb-2 ml-4 list-disc space-y-1 last:mb-0">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-2 ml-4 list-decimal space-y-1 last:mb-0">{children}</ol>
          ),
          li: ({ children }) => <li className="pl-0.5">{children}</li>,
          strong: ({ children }) => (
            <strong className="font-semibold text-foreground">{children}</strong>
          ),
          em: ({ children }) => <em className="italic">{children}</em>,
          h1: ({ children }) => <h3 className="mb-1.5 mt-2 font-semibold first:mt-0">{children}</h3>,
          h2: ({ children }) => <h3 className="mb-1.5 mt-2 font-semibold first:mt-0">{children}</h3>,
          h3: ({ children }) => <h4 className="mb-1 mt-2 font-semibold first:mt-0">{children}</h4>,
          code: ({ children }) => (
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>
          ),
          table: ({ children }) => (
            <div className="mb-2 overflow-x-auto">
              <table className="w-full border-collapse text-xs">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-border px-2 py-1 text-left font-semibold">{children}</th>
          ),
          td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
        }}
      >
        {body}
      </ReactMarkdown>

      {sourcesLine && (
        <p className="mt-2 border-t border-border/60 pt-1.5 text-xs italic text-muted-foreground">
          Sources: {sourcesLine}
        </p>
      )}
    </div>
  );
}
