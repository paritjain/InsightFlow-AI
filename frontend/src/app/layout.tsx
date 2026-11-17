import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "InsightFlow AI — Automated Data Analytics Platform",
  description: "Upload Excel, CSV, and PDF files and transform raw data into deterministic analytics, business story insights, and actionable decision plans.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-[#090d16] text-slate-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}
