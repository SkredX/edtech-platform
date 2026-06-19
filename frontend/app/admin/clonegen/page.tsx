"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { apiPost } from "@/lib/api";

interface Clone {
  question: string;
  options: Record<string, string>;
  correct: string;
  explanation: string;
}

interface CloneResponse {
  clones?: Clone[];
  error?: string;
  from_cache: boolean;
}

const ANSWER_LETTERS = ["A", "B", "C", "D"];

export default function CloneGenPage() {
  const [rawSeed, setRawSeed] = useState("");
  const [manualCorrect, setManualCorrect] = useState<string>("");
  const [nClones, setNClones] = useState(3);
  const [loading, setLoading] = useState(false);
  const [clones, setClones] = useState<Clone[] | null>(null);
  const [fromCache, setFromCache] = useState(false);

  const handleGenerate = async () => {
    if (!rawSeed.trim()) {
      toast.error("Paste a seed question first.");
      return;
    }
    setLoading(true);
    setClones(null);
    try {
      const data = await apiPost<CloneResponse>("/clone", {
        raw_seed: rawSeed,
        manual_correct: manualCorrect || null,
        n_clones: nClones,
      });
      if (data.error) {
        toast.error(data.error);
        return;
      }
      setClones(data.clones ?? []);
      setFromCache(data.from_cache);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Generation failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <header>
        <h1 className="text-lg font-semibold">CloneGen</h1>
        <p className="text-sm text-muted-foreground">
          Paste a seed MCQ to generate conceptually equivalent clones, grounded in your ingested
          course material.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Seed question</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder={
              'A specialised membranous structure in a prokaryotic cell which helps in cell ' +
              'wall formation, DNA replication is: [NEET 2025] (A) Cristae (B) Endoplasmic ' +
              'Reticulum (C) Mesosome (D) Chromatophores'
            }
            value={rawSeed}
            onChange={(e) => setRawSeed(e.target.value)}
            rows={4}
          />

          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1">
              <Label htmlFor="manual-correct">Correct answer (if not marked inline)</Label>
              <select
                id="manual-correct"
                value={manualCorrect}
                onChange={(e) => setManualCorrect(e.target.value)}
                className="h-9 rounded-lg border border-input bg-background px-3 text-sm"
              >
                <option value="">—</option>
                {ANSWER_LETTERS.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <Label htmlFor="n-clones">Number of clones (1–10)</Label>
              <input
                id="n-clones"
                type="number"
                min={1}
                max={10}
                value={nClones}
                onChange={(e) => setNClones(Math.max(1, Math.min(10, Number(e.target.value))))}
                className="h-9 w-20 rounded-lg border border-input bg-background px-3 text-sm"
              />
            </div>

            <Button onClick={handleGenerate} disabled={loading} className="ml-auto">
              {loading ? "Generating…" : "Generate clones"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {clones && clones.length > 0 && (
        <div className="space-y-4">
          <p className="text-xs text-muted-foreground">
            {fromCache ? "Served from cache — no API call made." : `${clones.length} clone(s) generated in 1 Groq call.`}
          </p>
          {clones.map((clone, idx) => (
            <Card key={idx}>
              <CardHeader>
                <CardTitle className="text-sm font-medium">Clone {idx + 1}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm">{clone.question}</p>
                <div className="space-y-1.5">
                  {Object.entries(clone.options).map(([letter, text]) => (
                    <div
                      key={letter}
                      className={
                        "rounded-md border px-3 py-2 text-sm " +
                        (letter === clone.correct
                          ? "border-green-500/50 bg-green-500/10"
                          : "border-border")
                      }
                    >
                      <span className="font-medium">{letter}.</span> {text}
                      {letter === clone.correct && (
                        <span className="ml-2 text-xs text-green-600">CORRECT</span>
                      )}
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground border-l-2 border-primary/40 pl-3">
                  {clone.explanation}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
