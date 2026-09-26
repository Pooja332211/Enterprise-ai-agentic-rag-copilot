import React, { useEffect, useState, useRef } from 'react';
import { api } from '../services/api';
import { DocumentItem } from '../types';
import {
  UploadCloud,
  FileText,
  Trash2,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Clock,
  Layers,
  HardDrive,
} from 'lucide-react';
import { FileTypeIcon, EmptyState } from './VisualAssets';

export const DocumentManager: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'info' } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const res = await api.getDocuments();
      setDocuments(res.items || []);
    } catch (e: any) {
      setMessage({ text: e.message || 'Failed to load documents', type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage(null);
    try {
      const res = await api.uploadDocument(file);
      if (res.is_duplicate) {
        setMessage({
          text: `Duplicate detected: document already indexed (ID: ${res.document?.id?.slice(0, 8)}...)`,
          type: 'info',
        });
      } else {
        setMessage({
          text: `Successfully ingested '${file.name}' (${res.document?.chunk_count || 1} chunks indexed)`,
          type: 'success',
        });
      }
      fetchDocuments();
    } catch (e: any) {
      setMessage({ text: e.message || 'Upload failed', type: 'error' });
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const handleDelete = async (docId: string, filename: string) => {
    if (!window.confirm(`Delete document '${filename}' and all vector/BM25 chunks?`)) {
      return;
    }
    try {
      await api.deleteDocument(docId);
      setMessage({ text: `Deleted '${filename}' from database and indexes`, type: 'success' });
      fetchDocuments();
    } catch (e: any) {
      setMessage({ text: e.message || 'Delete failed', type: 'error' });
    }
  };

  return (
    <div className="space-y-6">
      {/* Title Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-emerald-100 text-emerald-700">
              <FileText className="w-5 h-5" />
            </div>
            Enterprise Knowledge Base & Document Corpus
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-3xl">
            Multi-format document ingestion (PDF, DOCX, TXT, MD, HTML) with structure-aware chunking and dual Qdrant/BM25 indexing.
          </p>
        </div>
        <button
          onClick={fetchDocuments}
          disabled={loading}
          className="flex items-center gap-2 px-3.5 py-1.5 bg-white hover:bg-slate-50 text-slate-700 rounded-xl text-xs font-semibold transition border border-slate-200 shadow-2xs disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Corpus
        </button>
      </div>

      {message && (
        <div
          className={`p-4 rounded-2xl text-xs font-medium flex items-center gap-2.5 shadow-2xs ${
            message.type === 'success'
              ? 'bg-emerald-50 border border-emerald-200 text-emerald-800'
              : message.type === 'error'
              ? 'bg-rose-50 border border-rose-200 text-rose-800'
              : 'bg-sky-50 border border-sky-200 text-sky-800'
          }`}
        >
          {message.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 flex-shrink-0 text-emerald-600" />
          ) : (
            <AlertCircle className="w-4 h-4 flex-shrink-0 text-sky-600" />
          )}
          {message.text}
        </div>
      )}

      {/* Fresh Upload Box */}
      <div className="p-8 bg-gradient-to-b from-sky-50/40 via-white to-white border-2 border-dashed border-sky-200/80 hover:border-sky-400/80 rounded-3xl flex flex-col items-center justify-center text-center transition-all shadow-xs group">
        <div className="w-14 h-14 rounded-2xl bg-sky-100 text-sky-600 flex items-center justify-center mb-3 group-hover:scale-105 transition-transform duration-200 shadow-2xs">
          <UploadCloud className={`w-7 h-7 ${uploading ? 'animate-bounce' : ''}`} />
        </div>
        <h3 className="font-bold text-slate-900 text-sm">Upload Enterprise Knowledge File</h3>
        <p className="text-xs text-slate-500 max-w-md mt-1 leading-relaxed">
          Upload PDF, DOCX, TXT, MD, or HTML documents (up to 50MB). The backend extracts hierarchical headers, generates 384-d dense vectors in Qdrant, and indexes Okapi BM25 tokens.
        </p>

        {/* Supported format badges */}
        <div className="flex items-center gap-1.5 mt-3">
          <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-rose-50 text-rose-700 border border-rose-200">
            PDF
          </span>
          <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-sky-50 text-sky-700 border border-sky-200">
            DOCX
          </span>
          <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-slate-100 text-slate-700 border border-slate-200">
            TXT
          </span>
          <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-purple-50 text-purple-700 border border-purple-200">
            MD
          </span>
          <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-amber-50 text-amber-700 border border-amber-200">
            HTML
          </span>
        </div>

        <label className="mt-5 px-5 py-2.5 bg-sky-600 hover:bg-sky-700 text-white rounded-xl text-xs font-semibold transition cursor-pointer shadow-md shadow-sky-600/20 flex items-center gap-2 hover:scale-[1.02] active:scale-[0.99]">
          {uploading ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              Parsing & Indexing Document...
            </>
          ) : (
            <>
              <UploadCloud className="w-4 h-4" />
              Select File to Ingest
            </>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt,.md,.html"
            onChange={handleFileUpload}
            disabled={uploading}
            className="hidden"
          />
        </label>
      </div>

      {/* Documents Table */}
      <div className="bg-white border border-slate-200/80 rounded-3xl overflow-hidden shadow-xs">
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-600" />
            <h3 className="font-bold text-slate-800 text-xs uppercase tracking-wider">
              Ingested Document Corpus ({documents.length})
            </h3>
          </div>
          <span className="text-xs text-slate-400 font-mono flex items-center gap-1.5">
            <HardDrive className="w-3 h-3 text-slate-400" />
            Storage: SQLite + Qdrant
          </span>
        </div>

        {documents.length === 0 ? (
          <EmptyState
            type="no-documents"
            onAction={() => fileInputRef.current?.click()}
            actionLabel="Upload First Document"
          />
        ) : (
          <div className="divide-y divide-slate-100">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="p-4 sm:px-6 flex items-center justify-between hover:bg-slate-50/80 transition-all duration-150"
              >
                <div className="flex items-center gap-3.5 min-w-0">
                  <FileTypeIcon extension={doc.file_extension} size="md" />
                  <div className="min-w-0">
                    <div className="font-semibold text-slate-900 text-sm truncate">
                      {doc.filename}
                    </div>
                    <div className="text-xs text-slate-500 flex flex-wrap items-center gap-2 mt-0.5">
                      <span className="uppercase font-mono text-[10px] font-bold text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                        {doc.file_extension}
                      </span>
                      <span>•</span>
                      <span className="font-mono text-[11px]">
                        {(doc.file_size_bytes / 1024).toFixed(1)} KB
                      </span>
                      <span>•</span>
                      <span className="font-medium text-slate-700">
                        {doc.chunk_count} Chunks
                      </span>
                      <span>•</span>
                      <span className="inline-flex items-center gap-1 text-emerald-700 font-medium capitalize text-[11px]">
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                        {doc.status}
                      </span>
                      {doc.created_at && (
                        <>
                          <span>•</span>
                          <span className="text-slate-400 text-[10px] font-mono flex items-center gap-1">
                            <Clock className="w-3 h-3 text-slate-400" />
                            {new Date(doc.created_at).toLocaleDateString()}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 flex-shrink-0 ml-4">
                  <span className="text-xs font-mono text-slate-400 hidden md:inline">
                    {doc.id.slice(0, 8)}...
                  </span>
                  <button
                    onClick={() => handleDelete(doc.id, doc.filename)}
                    aria-label={`Delete document ${doc.filename}`}
                    className="p-2 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-xl transition cursor-pointer"
                    title="Cascade delete document from database and indexes"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
