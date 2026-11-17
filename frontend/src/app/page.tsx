'use client';

import React from 'react';
import Link from 'next/link';
import { 
  Sparkles, 
  ArrowRight, 
  FileSpreadsheet, 
  ShieldCheck, 
  Cpu, 
  LineChart, 
  FileText, 
  CheckCircle2, 
  Zap,
  Play,
  Layers,
  Database,
  BrainCircuit,
  PieChart
} from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col selection:bg-blue-600 selection:text-white">
      {/* Navigation Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-[#090d16]/80 border-b border-slate-800/80 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">
                InsightFlow <span className="text-blue-500">AI</span>
              </span>
              <span className="ml-2 text-[10px] uppercase font-semibold tracking-wider px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Enterprise
              </span>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <a href="#workflow" className="hover:text-white transition-colors">Workflow</a>
            <a href="#architecture" className="hover:text-white transition-colors">Deterministic AI</a>
          </nav>

          <div className="flex items-center gap-4">
            <Link
              href="/dashboard"
              className="px-5 py-2.5 rounded-lg font-medium text-sm text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 shadow-lg shadow-blue-600/25 transition-all flex items-center gap-2"
            >
              Open Dashboard
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="flex-1">
        <section className="relative pt-24 pb-20 px-6 max-w-6xl mx-auto text-center overflow-hidden">
          {/* Background Glow */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-blue-600/10 rounded-full blur-[140px] pointer-events-none" />

          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold uppercase tracking-wider mb-8">
            <Sparkles className="w-3.5 h-3.5" />
            Next-Gen Automated Data Analytics Platform
          </div>

          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-slate-100 max-w-4xl mx-auto leading-[1.15]">
            Turn messy data into <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400">decisions.</span>
          </h1>

          <p className="mt-6 text-lg md:text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed">
            Upload your Excel, CSV, or PDF files and let AI clean, profile, calculate, explain, and transform raw numbers into an actionable executive story.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/workspace/proj-demo-q3-sales"
              className="w-full sm:w-auto px-8 py-4 rounded-xl font-semibold text-white bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:opacity-95 shadow-xl shadow-blue-600/30 transition-all flex items-center justify-center gap-3 text-base"
            >
              <Play className="w-5 h-5 fill-current" />
              Launch Demo Workspace
            </Link>
            <Link
              href="/dashboard"
              className="w-full sm:w-auto px-8 py-4 rounded-xl font-semibold text-slate-300 bg-slate-900 border border-slate-800 hover:border-slate-700 hover:text-white transition-all flex items-center justify-center gap-2 text-base"
            >
              Upload Your Data
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>

          <div className="mt-12 flex items-center justify-center gap-8 text-xs font-medium text-slate-500">
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              100% Deterministic Python Calculations
            </span>
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Human-in-the-loop Data Cleaning
            </span>
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Multi-Sheet Excel Export
            </span>
          </div>
        </section>

        {/* Visual Workflow Steps */}
        <section id="workflow" className="py-16 px-6 max-w-7xl mx-auto border-t border-slate-800/60">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-3xl font-bold">Primary User Journey</h2>
            <p className="text-slate-400 text-sm mt-2">End-to-End Automated Analytics & Decision Pipeline</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {[
              { step: "01", name: "Upload", desc: "Excel, CSV, or PDF files", icon: FileSpreadsheet, color: "from-blue-500/20 to-blue-600/20 text-blue-400" },
              { step: "02", name: "Clean", desc: "Interactive issue rules", icon: ShieldCheck, color: "from-emerald-500/20 to-emerald-600/20 text-emerald-400" },
              { step: "03", name: "Analyze", desc: "Deterministic Pandas EDA", icon: LineChart, color: "from-purple-500/20 to-purple-600/20 text-purple-400" },
              { step: "04", name: "Understand", desc: "Evidence-backed story", icon: BrainCircuit, color: "from-amber-500/20 to-amber-600/20 text-amber-400" },
              { step: "05", name: "Act", desc: "Action plan & master export", icon: Zap, color: "from-rose-500/20 to-rose-600/20 text-rose-400" },
            ].map((item, i) => (
              <div key={i} className="p-6 rounded-2xl glass-card border border-slate-800/80 hover:border-slate-700 transition-all flex flex-col justify-between">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-mono font-bold text-slate-500">{item.step}</span>
                  <div className={`p-2.5 rounded-xl bg-gradient-to-br ${item.color}`}>
                    <item.icon className="w-5 h-5" />
                  </div>
                </div>
                <div>
                  <h3 className="font-bold text-lg text-slate-200">{item.name}</h3>
                  <p className="text-xs text-slate-400 mt-1">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Feature Highlights Grid */}
        <section id="features" className="py-20 px-6 max-w-7xl mx-auto border-t border-slate-800/60">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="p-8 rounded-2xl glass-card border border-slate-800/80">
              <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-6">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-2">Data Quality & Audit Trail</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Automatic data health score (0-100), duplicate detection, whitespace trimming, casing standardization, and full step-by-step undoable audit trail.
              </p>
            </div>

            <div className="p-8 rounded-2xl glass-card border border-slate-800/80">
              <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-6">
                <Layers className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-2">Multi-File ETL & Data Model</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Infers foreign key relationships across files (`Sales.Customer_ID → Customers.Customer_ID`), generates logical schema graphs, and builds safe master datasets.
              </p>
            </div>

            <div className="p-8 rounded-2xl glass-card border border-slate-800/80">
              <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 mb-6">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-2">Zero-Hallucination Analytics</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Aggregations, growth rates, and statistical tests are computed deterministically via Python Pandas. LLMs are used strictly for semantic storytelling and recommendations.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-8 px-6 text-center text-xs text-slate-500">
        <p>© 2026 InsightFlow AI. Built with Next.js, FastAPI & Deterministic Analytics Engine.</p>
      </footer>
    </div>
  );
}
