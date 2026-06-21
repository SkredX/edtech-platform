import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

export default function AdminDashboardPage() {
  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <header>
        <h1 className="text-lg font-semibold">Teacher Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Manage course material and generate practice questions for your students.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Document Ingestion</CardTitle>
            <CardDescription>
              Upload course material (PDF). Chunked, embedded, clustered, and summarized.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild className="w-full">
              <Link href="/admin/ingest">Upload a PDF</Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">CloneGen</CardTitle>
            <CardDescription>
              Paste a seed MCQ to generate conceptually equivalent clones, grounded in your
              ingested course material.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="outline" className="w-full">
              <Link href="/admin/clonegen">Generate questions</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
