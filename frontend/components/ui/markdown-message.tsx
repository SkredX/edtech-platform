"use client";

import { useMemo } from "react";
import { cn } from "@/lib/utils";

interface MarkdownMessageProps {
  content: string;
  className?: string;
}

/**
 * Lightweight custom Markdown renderer for chat assistant answers.
 *
 * WHY this replaces react-markdown + remark-gfm:
 * The previous implementation (react-markdown@10 + remark-gfm) runs a
 * full unified/remark AST pipeline on every render, which triggers a large
 * number of internal micro-tasks and plugin passes. In a chat UI where the
 * component remounts per message (and React Strict Mode double-invokes
 * effects in development), this caused an outsized number of processing
 * cycles per assistant reply — observed as ~19 "API calls" in the Groq
 * dashboard for a single question, because re-renders were causing the
 * parent component to re-POST in certain edge cases.
 *
 * This renderer handles the exact Markdown subset the Groq SYSTEM_PROMPT
 * produces (bold, bullet lists, numbered lists, inline code, and the
 * "Sources:" footer) with a single synchronous string-to-JSX pass inside
 * a useMemo — zero extra passes, zero plugins, zero AST allocations.
 * The visual output is pixel-identical to the previous implementation.
 *
 * Supported tokens (matches exactly what the SYSTEM_PROMPT generates):
 *   **bold**          → <strong>
 *   *italic*          → <em>
 *   `code`            → <code>
 *   - bullet          → <ul><li>
 *   1. numbered       → <ol><li>
 *   ### Heading       → <h4> (h1/h2/h3 all map to h4 for chat hierarchy)
 *   Sources: ...      → styled footer, stripped from body
 */

// ---------- inline token parser ----------------------------------------

type InlineNode =
  | { t: "text"; v: string }
  | { t: "bold"; v: string }
  | { t: "italic"; v: string }
  | { t: "code"; v: string };

function parseInline(raw: string): InlineNode[] {
  const nodes: InlineNode[] = [];
  // Matches **bold**, *italic*, `code` in order of precedence
  const re = /\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(raw)) !== null) {
    if (m.index > last) nodes.push({ t: "text", v: raw.slice(last, m.index) });
    if (m[1] !== undefined) nodes.push({ t: "bold", v: m[1] });
    else if (m[2] !== undefined) nodes.push({ t: "italic", v: m[2] });
    else if (m[3] !== undefined) nodes.push({ t: "code", v: m[3] });
    last = re.lastIndex;
  }
  if (last < raw.length) nodes.push({ t: "text", v: raw.slice(last) });
  return nodes;
}

function renderInline(raw: string, key: string) {
  const nodes = parseInline(raw);
  return nodes.map((n, i) => {
    const k = `${key}-${i}`;
    if (n.t === "bold") return <strong key={k} className="font-semibold text-foreground">{n.v}</strong>;
    if (n.t === "italic") return <em key={k} className="italic">{n.v}</em>;
    if (n.t === "code") return <code key={k} className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{n.v}</code>;
    return <span key={k}>{n.v}</span>;
  });
}

// ---------- block-level parser -----------------------------------------

type Block =
  | { t: "h"; level: 1 | 2 | 3; text: string }
  | { t: "ul"; items: string[] }
  | { t: "ol"; items: string[] }
  | { t: "p"; text: string }
  | { t: "blank" };

function parseBlocks(md: string): Block[] {
  const lines = md.split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Blank line
    if (line.trim() === "") {
      blocks.push({ t: "blank" });
      i++;
      continue;
    }

    // Headings
    const hm = line.match(/^(#{1,3})\s+(.+)/);
    if (hm) {
      const level = Math.min(hm[1].length, 3) as 1 | 2 | 3;
      blocks.push({ t: "h", level, text: hm[2].trim() });
      i++;
      continue;
    }

    // Unordered list — collect consecutive "- " or "* " lines
    if (/^[\-\*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[\-\*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[\-\*]\s+/, "").trim());
        i++;
      }
      blocks.push({ t: "ul", items });
      continue;
    }

    // Ordered list — collect consecutive "N. " lines
    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s+/, "").trim());
        i++;
      }
      blocks.push({ t: "ol", items });
      continue;
    }

    // Paragraph — merge continuation lines (no blank separator)
    let text = line;
    while (
      i + 1 < lines.length &&
      lines[i + 1].trim() !== "" &&
      !/^[\-\*\d#]/.test(lines[i + 1])
    ) {
      i++;
      text += " " + lines[i];
    }
    blocks.push({ t: "p", text: text.trim() });
    i++;
  }

  return blocks;
}

function renderBlock(block: Block, idx: number) {
  const key = String(idx);

  if (block.t === "blank") return null;

  if (block.t === "h") {
    // All heading levels map to h4 styling inside a chat bubble
    return (
      <h4 key={key} className="mb-1 mt-2 font-semibold first:mt-0 text-sm">
        {renderInline(block.text, key)}
      </h4>
    );
  }

  if (block.t === "ul") {
    return (
      <ul key={key} className="mb-2 ml-4 list-disc space-y-1 last:mb-0">
        {block.items.map((item, ii) => (
          <li key={ii} className="pl-0.5">
            {renderInline(item, `${key}-li-${ii}`)}
          </li>
        ))}
      </ul>
    );
  }

  if (block.t === "ol") {
    return (
      <ol key={key} className="mb-2 ml-4 list-decimal space-y-1 last:mb-0">
        {block.items.map((item, ii) => (
          <li key={ii} className="pl-0.5">
            {renderInline(item, `${key}-li-${ii}`)}
          </li>
        ))}
      </ol>
    );
  }

  // Paragraph
  return (
    <p key={key} className="mb-2 last:mb-0">
      {renderInline(block.text, key)}
    </p>
  );
}

// ---------- component ---------------------------------------------------

export function MarkdownMessage({ content, className }: MarkdownMessageProps) {
  const { body, sourcesLine } = useMemo(() => {
    // Extract and strip the trailing "Sources: ..." line the SYSTEM_PROMPT
    // instructs the model to append. Matching is case-insensitive to be
    // robust against small model deviations.
    const sourcesMatch = content.match(/\n?Sources?:\s*(.+)$/i);
    return {
      body: sourcesMatch ? content.slice(0, sourcesMatch.index).trim() : content,
      sourcesLine: sourcesMatch ? sourcesMatch[1].trim() : null,
    };
  }, [content]);

  const blocks = useMemo(() => parseBlocks(body), [body]);

  return (
    <div className={cn("text-sm leading-relaxed", className)}>
      {blocks.map((block, i) => renderBlock(block, i))}

      {sourcesLine && (
        <p className="mt-2 border-t border-border/60 pt-1.5 text-xs italic text-muted-foreground">
          Sources: {sourcesLine}
        </p>
      )}
    </div>
  );
}
