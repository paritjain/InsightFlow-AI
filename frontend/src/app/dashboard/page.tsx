'use client';

import React, { useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { 
  Sparkles, 
  Plus, 
  FileSpreadsheet, 
  FolderKanban, 
  Activity, 
  ArrowUpRight, 
  Upload, 
  CheckCircle2, 
  Clock, 
  BarChart3,
  Search,
  X,
  FileCode,
  Loader2,
  AlertCircle
} from 'lucide-react';
import { fetchProjects, triggerDemoProject, uploadProjectFiles } from '@/lib/api';

export default function DashboardPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [projects, setProjects] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // Upload modal states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    loadProjects();
  }, []);

  async function loadProjects() {
    try {
      setLoading(true);
      const data = await fetchProjects();
      setProjects(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateDemo() {
    try {
      await triggerDemoProject();
      await loadProjects();
    } catch (err) {
      console.error(err);
    }
  }

  function openModal() {
    setErrorMsg(null);
    setIsModalOpen(true);
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      const filesArray = Array.from(e.target.files);
      setSelectedFiles(prev => [...prev, ...filesArray]);
      setErrorMsg(null);
      setIsModalOpen(true);
    }
    e.target.value = '';
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const filesArray = Array.from(e.dataTransfer.files).filter(f => 
        f.name.endsWith('.xlsx') || f.name.endsWith('.xls') || f.name.endsWith('.csv') || f.name.endsWith('.pdf')
      );
      if (filesArray.length === 0) {
        setErrorMsg('Please select valid Excel (.xlsx, .xls), CSV (.csv), or PDF (.pdf) files.');
        return;
      }
      setSelectedFiles(prev => [...prev, ...filesArray]);
      setErrorMsg(null);
    }
  }

  function removeFile(e: React.MouseEvent, index: number) {
    e.stopPropagation();
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  }

  async function handleUploadSubmit() {
    if (selectedFiles.length === 0) {
      setErrorMsg('Please select at least one Excel, CSV, or PDF file.');
      return;
    }
    try {
      setUploading(true);
      setErrorMsg(null);
      const res = await uploadProjectFiles(selectedFiles);
      setIsModalOpen(false);
      setSelectedFiles([]);
      router.push(`/workspace/${res.project_id}`);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Error uploading files. Please try again.');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".xlsx,.xls,.csv,.pdf"
        onChange={handleFileSelect}
        className="fixed top-0 left-0 w-0 h-0 opacity-0 pointer-events-none z-[-1]"
      />
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-[#090d16]/90 px-8 py-4 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center shadow-md shadow-blue-500/20">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-lg tracking-tight">
                InsightFlow <span className="text-blue-500">AI</span>
              </span>
            </Link>
            <span className="text-slate-700">/</span>
            <span className="text-sm font-semibold text-slate-400">Dashboard</span>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={handleCreateDemo}
              className="px-4 py-2 rounded-lg font-medium text-xs text-white bg-slate-800 border border-slate-700 hover:border-slate-600 transition-all flex items-center gap-2"
            >
              <Plus className="w-3.5 h-3.5" />
              New Demo Project
            </button>
            <button
              onClick={openModal}
              className="px-4 py-2 rounded-lg font-semibold text-xs text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 shadow-md shadow-blue-600/20 transition-all flex items-center gap-2"
            >
              <Upload className="w-3.5 h-3.5" />
              Import Excel Files
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-8 py-8 space-y-8">
        {/* Welcome Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/60 pb-6">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Analytics Workspace</h1>
            <p className="text-sm text-slate-400 mt-1">Manage project datasets, profiling health, and automated executive reports.</p>
          </div>
          <button
            onClick={openModal}
            className="px-5 py-2.5 rounded-xl font-semibold text-sm text-white bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-500/20 transition-all flex items-center gap-2"
          >
            <Upload className="w-4 h-4" />
            Upload New Dataset
          </button>
        </div>

        {/* Quick Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          <div className="p-5 rounded-xl glass-card border border-slate-800/80 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Active Projects</p>
              <h3 className="text-2xl font-extrabold mt-1">{projects.length}</h3>
            </div>
            <div className="p-3 rounded-lg bg-blue-500/10 text-blue-400">
              <FolderKanban className="w-5 h-5" />
            </div>
          </div>

          <div className="p-5 rounded-xl glass-card border border-slate-800/80 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Avg Data Health Score</p>
              <h3 className="text-2xl font-extrabold text-emerald-400 mt-1">
                {projects.length > 0 ? `${projects[0].health_score}/100` : 'N/A'}
              </h3>
            </div>
            <div className="p-3 rounded-lg bg-emerald-500/10 text-emerald-400">
              <Activity className="w-5 h-5" />
            </div>
          </div>

          <div className="p-5 rounded-xl glass-card border border-slate-800/80 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total Rows Processed</p>
              <h3 className="text-2xl font-extrabold text-purple-400 mt-1">
                {projects.length > 0 ? projects[0].row_count.toLocaleString() : '0'}
              </h3>
            </div>
            <div className="p-3 rounded-lg bg-purple-500/10 text-purple-400">
              <BarChart3 className="w-5 h-5" />
            </div>
          </div>

          <div className="p-5 rounded-xl glass-card border border-slate-800/80 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Files Merged (ETL)</p>
              <h3 className="text-2xl font-extrabold text-indigo-400 mt-1">
                {projects.length > 0 ? projects[0].file_count : '0'}
              </h3>
            </div>
            <div className="p-3 rounded-lg bg-indigo-500/10 text-indigo-400">
              <FileSpreadsheet className="w-5 h-5" />
            </div>
          </div>
        </div>

        {/* Projects List */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-slate-200">Recent Projects</h2>
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search projects..."
                className="pl-9 pr-4 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          {loading ? (
            <div className="p-12 text-center text-slate-500 text-sm">Loading projects...</div>
          ) : projects.length === 0 ? (
            <div className="p-12 rounded-2xl border border-dashed border-slate-800 text-center space-y-4">
              <FileSpreadsheet className="w-12 h-12 text-blue-500/60 mx-auto" />
              <div>
                <p className="text-base font-semibold text-slate-300">No active dataset projects yet</p>
                <p className="text-xs text-slate-500 mt-1">Import your Excel (.xlsx, .xls) or CSV files to generate instant automated analytics.</p>
              </div>
              <div className="flex items-center justify-center gap-3">
                <button
                  onClick={openModal}
                  className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white shadow-md shadow-blue-600/20"
                >
                  Import Excel File
                </button>
                <button
                  onClick={handleCreateDemo}
                  className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300 border border-slate-700"
                >
                  Create Demo Project
                </button>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {projects.map((p) => (
                <Link
                  key={p.id}
                  href={`/workspace/${p.id}`}
                  className="group p-6 rounded-2xl glass-card border border-slate-800/80 hover:border-blue-500/40 transition-all flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="px-2.5 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        {p.status}
                      </span>
                      <span className="text-xs font-semibold text-slate-400 flex items-center gap-1">
                        Health Score: <strong className="text-emerald-400">{p.health_score}/100</strong>
                      </span>
                    </div>

                    <h3 className="font-bold text-lg text-slate-100 group-hover:text-blue-400 transition-colors flex items-center justify-between">
                      {p.name}
                      <ArrowUpRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </h3>
                    <p className="text-xs text-slate-400 mt-2 line-clamp-2 leading-relaxed">
                      {p.description}
                    </p>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-500">
                    <span className="flex items-center gap-1.5">
                      <FileSpreadsheet className="w-3.5 h-3.5 text-slate-400" />
                      {p.file_count} Files (Master ETL)
                    </span>
                    <span>{p.row_count.toLocaleString()} rows</span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </main>

      {/* Excel File Import Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
          <div className="bg-[#0f172a] border border-slate-800 rounded-2xl max-w-xl w-full p-6 shadow-2xl space-y-6 relative">
            
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  <FileSpreadsheet className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-bold text-lg text-slate-100">Import Excel / CSV Files</h3>
                  <p className="text-xs text-slate-400">Upload one or multiple spreadsheets for automated cleaning & EDA.</p>
                </div>
              </div>
              <button
                onClick={() => !uploading && setIsModalOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Error banner */}
            {errorMsg && (
              <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* File Drag and Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-blue-500 bg-blue-500/10 scale-[0.99]'
                  : 'border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/80'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".xlsx,.xls,.csv,.pdf"
                onChange={handleFileSelect}
                className="hidden"
              />

              <Upload className="w-10 h-10 text-blue-400 mx-auto mb-3" />
              <p className="text-sm font-semibold text-slate-200">
                Click to browse or drag & drop files here
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Supports Excel (.xlsx, .xls), CSV (.csv), and PDF (.pdf) documents
              </p>
            </div>

            {/* Selected files list */}
            {selectedFiles.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Selected Files ({selectedFiles.length})</p>
                <div className="max-h-40 overflow-y-auto space-y-2 pr-1">
                  {selectedFiles.map((file, idx) => (
                    <div key={idx} className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2.5">
                        <FileCode className="w-4 h-4 text-emerald-400" />
                        <span className="font-semibold text-slate-200 truncate max-w-xs">{file.name}</span>
                        <span className="text-slate-500">({(file.size / 1024).toFixed(1)} KB)</span>
                      </div>
                      <button
                        onClick={(e) => removeFile(e, idx)}
                        disabled={uploading}
                        className="text-slate-500 hover:text-rose-400 transition-colors p-1"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                disabled={uploading}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-white bg-slate-900 border border-slate-800 transition-all"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handleUploadSubmit}
                disabled={uploading || selectedFiles.length === 0}
                className="px-6 py-2.5 rounded-xl text-xs font-semibold text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 shadow-lg shadow-blue-500/20 disabled:opacity-50 transition-all flex items-center gap-2"
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-white" />
                    <span>Processing Pipeline...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Upload & Process</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

