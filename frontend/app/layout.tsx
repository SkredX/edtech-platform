import type { Metadata } from "next";
import { Toaster } from "sonner";
import { ApiKeySettings } from "@/components/ui/api-key-settings";
import "./globals.css";

export const metadata: Metadata = {
  title: "SkredX-EdTech — AI Study Assistant",
  description: "Curriculum-aligned RAG chatbot and adaptive question bank for NEET/JEE prep.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-foreground antialiased">
        {children}
        <ApiKeySettings />
        <Toaster richColors position="top-right" />
      </body>
    </html>
  );
}
