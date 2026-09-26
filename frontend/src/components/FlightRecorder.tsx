import React, { useState } from 'react';
import {
  Terminal,
  Copy,
  Check,
  ChevronRight,
  Activity,
} from 'lucide-react';
import { ExecutionTraceItem } from '../types';

interface FlightRecorderProps {
  executionTrace: ExecutionTraceItem[];
  timings?: Record<string, number>;
  totalDurationMs?: number;
  status?: string;
}

export const FlightRecorder: React.FC<FlightRecorderProps> = ({
  executionTrace,
  timings,
  totalDurationMs = 0,
  status = 'completed',
}) => {
  const [copied, setCopied] = useState(false);
  const [expandedSpanIdx, setExpandedSpanIdx] = useState<number | null>(null);

  if (!executionTrace || executionTrace.length === 0) return null;

  const handleCopy = () => {
    const logText = executionTrace
      .map((item, i) => {
        const time = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : `Step ${i + 1}`;
        const node = item.node || item.step || item.stage || 'Node';
        const dur = (item.duration_ms ?? item.latency_ms ?? 0).toFixed(2);
        return `[${time}] ${node} (${dur} ms) -> ${item.output_summary || 'Done'}`;
      })
      .join('\n');

    navigator.clipboard.writeText(logText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-slate-950 text-slate-200 rounded-3xl border border-slate-800 shadow-xl overflow-hidden font-mono text-xs">
      {/* Flight Recorder Header */}
      <div className="p-4 sm:px-6 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Terminal className="w-4 h-4" />
          </div>
          <div>
            <div className="font-bold text-slate-100 text-xs flex items-center gap-2">
              LangGraph Research Flight Recorder
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            </div>
            <div className="text-[10px] text-slate-400 font-sans">
              Deterministic state-by-state execution telemetry & trace log
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {timings && Object.keys(timings).length > 0 && (
            <div className="hidden xl:flex items-center gap-2 text-[10px] text-slate-400">
              {Object.entries(timings).slice(0, 2).map(([k, v]) => (
                <span key={k} className="bg-slate-800/80 px-1.5 py-0.5 rounded border border-slate-700/60 font-mono">
                  {k.replace(/_ms$/, '')}: {typeof v === 'number' ? v.toFixed(0) : v}ms
                </span>
              ))}
            </div>
          )}
          <div className="text-[11px] text-slate-400">
            Total Duration:{' '}
            <span className="font-bold text-emerald-400">
              {totalDurationMs > 0 ? `${totalDurationMs.toFixed(1)} ms` : '—'}
            </span>
          </div>

          <button
            type="button"
            onClick={handleCopy}
            aria-label="Copy trace logs"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition flex items-center gap-1 text-[11px] cursor-pointer"
            title="Copy Flight Recorder Trace"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>

      {/* Flight Recorder Log Stream */}
      <div className="p-4 sm:p-5 space-y-2 max-h-96 overflow-y-auto font-mono text-[11px]">
        {executionTrace.map((item, idx) => {
          const formattedTime = item.timestamp
            ? new Date(item.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              })
            : `+${idx * 10}ms`;

          const nodeName = item.node || item.step || item.stage || `Stage ${idx + 1}`;
          const duration = item.duration_ms ?? item.latency_ms ?? 0;
          const isExpanded = expandedSpanIdx === idx;

          return (
            <div
              key={idx}
              onClick={() => setExpandedSpanIdx(isExpanded ? null : idx)}
              className="p-2.5 rounded-xl bg-slate-900/60 hover:bg-slate-900 border border-slate-800/80 transition cursor-pointer space-y-1.5"
            >
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2.5">
                  <span className="text-slate-500 font-mono text-[10px] w-16">{formattedTime}</span>
                  <span className="px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-bold text-[10px]">
                    N{idx + 1}
                  </span>
                  <span className="font-semibold text-slate-200">{nodeName}</span>
                </div>

                <div className="flex items-center gap-3 text-slate-400 font-mono text-[10px]">
                  <span className="text-emerald-400 font-semibold">{duration.toFixed(2)} ms</span>
                  <ChevronRight
                    className={`w-3.5 h-3.5 text-slate-500 transition-transform ${
                      isExpanded ? 'rotate-90' : ''
                    }`}
                  />
                </div>
              </div>

              {item.output_summary && (
                <div className="text-slate-400 pl-8 text-[10.5px] truncate max-w-2xl font-sans">
                  ↳ {item.output_summary}
                </div>
              )}

              {isExpanded && (
                <div className="mt-2 pt-2 border-t border-slate-800 text-[10px] space-y-1 text-slate-300 pl-8 bg-slate-950/60 p-2 rounded-lg">
                  {item.input_summary && (
                    <div>
                      <span className="text-slate-500 uppercase font-bold">Input:</span> {item.input_summary}
                    </div>
                  )}
                  {item.output_summary && (
                    <div>
                      <span className="text-indigo-400 uppercase font-bold">Output:</span> {item.output_summary}
                    </div>
                  )}
                  {item.iteration !== undefined && (
                    <div>
                      <span className="text-slate-500 uppercase font-bold">Iteration:</span> {item.iteration}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer Status Bar */}
      <div className="px-6 py-2.5 bg-slate-900 border-t border-slate-800 text-[10px] text-slate-400 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-emerald-400" />
          <span>Status: <span className="text-emerald-400 uppercase font-bold">{status}</span></span>
        </div>
        <div className="text-slate-500">
          Telemetry Source: LangGraph StateGraph trace recorder
        </div>
      </div>
    </div>
  );
};
