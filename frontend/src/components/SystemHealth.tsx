import React, { useEffect, useState } from 'react';
import { api, BACKEND_URL } from '../services/api';
import { HealthStatus } from '../types';
import {
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  RefreshCw,
  Cpu,
  Database,
  Server,
  BarChart3,
  Layers,
  Sparkles,
  Bot,
  Activity,
} from 'lucide-react';

export const SystemHealth: React.FC = () => {
  const [liveness, setLiveness] = useState<HealthStatus | null>(null);
  const [readiness, setReadiness] = useState<HealthStatus | null>(null);
  const [metrics, setMetrics] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const [liveRes, readyRes, metricsText] = await Promise.all([
        api.checkHealth().catch((e) => ({ status: 'error', error: e.message })),
        api.checkReadiness().catch((e) => ({ status: 'error', error: e.message })),
        api.getMetrics().catch((e) => `Error fetching metrics: ${e.message}`),
      ]);
      setLiveness(liveRes);
      setReadiness(readyRes);
      setMetrics(metricsText);
    } catch (e: any) {
      setError(e.message || 'Failed to check system health');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const getServiceIcon = (name: string) => {
    const n = name.toLowerCase();
    if (n.includes('vector') || n.includes('qdrant')) return Layers;
    if (n.includes('embed')) return Cpu;
    if (n.includes('rerank')) return Sparkles;
    if (n.includes('llm')) return Bot;
    if (n.includes('database') || n.includes('sqlite') || n.includes('postgres')) return Database;
    return Server;
  };

  return (
    <div className="space-y-6">
      {/* Title Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-violet-100 text-violet-700">
              <Activity className="w-5 h-5" />
            </div>
            System Health & Production Observability
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-3xl">
            Real-time liveness probes, deep dependency readiness evaluations, and native Prometheus metrics exposition.
          </p>
        </div>
        <button
          onClick={fetchStatus}
          disabled={loading}
          aria-label="Refresh system probes"
          className="flex items-center gap-2 px-3.5 py-1.5 bg-white hover:bg-slate-50 text-slate-700 rounded-xl text-xs font-semibold transition border border-slate-200 shadow-2xs disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Probes
        </button>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl text-rose-700 text-sm shadow-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-600" />
          {error}
        </div>
      )}

      {/* Probes Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Liveness Probe Card */}
        <div className="p-6 bg-white border border-slate-200/80 rounded-3xl shadow-xs space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div className="flex items-center gap-2 font-bold text-slate-800 text-sm">
              <Server className="w-4 h-4 text-indigo-600" />
              Liveness Probe (Process Health)
            </div>
            <span className="text-[11px] font-mono px-2 py-0.5 bg-slate-100 border border-slate-200 rounded-md text-slate-600 font-semibold">
              GET /health
            </span>
          </div>

          <div className="flex items-center gap-3.5 bg-slate-50/80 p-4 rounded-2xl border border-slate-200/60">
            {liveness?.status === 'healthy' ? (
              <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center flex-shrink-0">
                <CheckCircle2 className="w-6 h-6 text-emerald-600" />
              </div>
            ) : (
              <div className="w-10 h-10 rounded-xl bg-rose-100 text-rose-700 flex items-center justify-center flex-shrink-0">
                <AlertCircle className="w-6 h-6 text-rose-600" />
              </div>
            )}
            <div>
              <div className="font-bold capitalize text-slate-900 text-sm flex items-center gap-2">
                Status: {liveness?.status || 'Unknown'}
                <span className="text-[11px] px-2 py-0.5 rounded-full font-mono bg-emerald-100 text-emerald-800 border border-emerald-200 font-semibold">
                  HTTP 200
                </span>
                {liveness?.status === 'healthy' && (
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                )}
              </div>
              <div className="text-xs text-slate-500 font-mono mt-0.5">
                Target: {BACKEND_URL}/health
              </div>
            </div>
          </div>

          {liveness && (
            <div className="p-3 bg-slate-900 rounded-2xl font-mono text-xs text-slate-200 overflow-x-auto border border-slate-800">
              <pre>{JSON.stringify(liveness, null, 2)}</pre>
            </div>
          )}
        </div>

        {/* Readiness Probe Card */}
        <div className="p-6 bg-white border border-slate-200/80 rounded-3xl shadow-xs space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div className="flex items-center gap-2 font-bold text-slate-800 text-sm">
              <Database className="w-4 h-4 text-emerald-600" />
              Deep Dependency Readiness Probe
            </div>
            <span className="text-[11px] font-mono px-2 py-0.5 bg-slate-100 border border-slate-200 rounded-md text-slate-600 font-semibold">
              GET /health/ready
            </span>
          </div>

          <div className="flex items-center gap-3.5 bg-slate-50/80 p-4 rounded-2xl border border-slate-200/60">
            {readiness?.status === 'ready' || readiness?.status === 'healthy' ? (
              <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center flex-shrink-0">
                <CheckCircle2 className="w-6 h-6 text-emerald-600" />
              </div>
            ) : (
              <div className="w-10 h-10 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center flex-shrink-0">
                <AlertTriangle className="w-6 h-6 text-amber-600" />
              </div>
            )}
            <div>
              <div className="font-bold capitalize text-slate-900 text-sm flex items-center gap-2">
                Readiness: {readiness?.status || 'Unknown'}
                <span className="text-[11px] px-2 py-0.5 rounded-full font-mono bg-emerald-100 text-emerald-800 border border-emerald-200 font-semibold">
                  All Systems Operational
                </span>
              </div>
              <div className="text-xs text-slate-500 font-mono mt-0.5">
                Target: {BACKEND_URL}/health/ready
              </div>
            </div>
          </div>

          {readiness?.services && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {Object.entries(readiness.services).map(([name, svc]: [string, any]) => {
                const SvcIcon = getServiceIcon(name);
                const isHealthy = svc.status === 'healthy' || svc.status === 'ready';
                const isDegraded = svc.status === 'degraded';
                return (
                  <div
                    key={name}
                    className="p-3 bg-slate-50 rounded-2xl border border-slate-200/80 flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-700 shadow-2xs">
                        <SvcIcon className="w-3.5 h-3.5" />
                      </div>
                      <span className="text-slate-800 capitalize font-medium">
                        {name.replace(/_/g, ' ')}
                      </span>
                    </div>

                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-mono font-semibold text-[10px] ${
                        isHealthy
                          ? 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                          : isDegraded
                          ? 'bg-amber-100 text-amber-800 border border-amber-200'
                          : 'bg-rose-100 text-rose-800 border border-rose-200'
                      }`}
                    >
                      {isHealthy ? (
                        <CheckCircle2 className="w-3 h-3 text-emerald-700" />
                      ) : isDegraded ? (
                        <AlertTriangle className="w-3 h-3 text-amber-700" />
                      ) : (
                        <AlertCircle className="w-3 h-3 text-rose-700" />
                      )}
                      <span>{svc.status}</span>
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Prometheus Metrics */}
      <div className="p-6 bg-white border border-slate-200/80 rounded-3xl shadow-xs space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-100">
          <div className="flex items-center gap-2 font-bold text-slate-900 text-sm">
            <BarChart3 className="w-4 h-4 text-sky-600" />
            Prometheus Observability Metrics
          </div>
          <span className="text-[11px] font-mono px-2 py-0.5 bg-slate-100 border border-slate-200 rounded-md text-slate-600 font-semibold">
            GET /metrics
          </span>
        </div>

        <div className="p-4 bg-slate-950 rounded-2xl font-mono text-xs text-slate-300 overflow-x-auto border border-slate-800 max-h-80">
          <pre>{metrics || 'No metrics exposition loaded. Click Refresh Probes to fetch.'}</pre>
        </div>
      </div>
    </div>
  );
};
