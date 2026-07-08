import { ChevronRight, Radar, ArrowRight } from 'lucide-react';
import { PHASES } from '../types';
import { Page } from '../App';

interface Props {
  onNavigate: (page: Page) => void;
}

const colorMap: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  sky:     { bg: 'bg-sky-500/10',     text: 'text-sky-400',     border: 'border-sky-500/20',     dot: 'bg-sky-400' },
  teal:    { bg: 'bg-teal-500/10',    text: 'text-teal-400',    border: 'border-teal-500/20',    dot: 'bg-teal-400' },
  cyan:    { bg: 'bg-cyan-500/10',    text: 'text-cyan-400',    border: 'border-cyan-500/20',    dot: 'bg-cyan-400' },
  blue:    { bg: 'bg-blue-500/10',    text: 'text-blue-400',    border: 'border-blue-500/20',    dot: 'bg-blue-400' },
  amber:   { bg: 'bg-amber-500/10',   text: 'text-amber-400',   border: 'border-amber-500/20',   dot: 'bg-amber-400' },
  orange:  { bg: 'bg-orange-500/10',  text: 'text-orange-400',  border: 'border-orange-500/20',  dot: 'bg-orange-400' },
  emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/20', dot: 'bg-emerald-400' },
};

export default function PipelinePage({ onNavigate }: Props) {
  return (
    <div className="p-8">
      <div className="mb-8">
        <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
          <Radar size={14} className="text-sky-400" />
          <span>Pipeline</span>
        </div>
        <h1 className="text-3xl font-bold text-white">Detection Pipeline</h1>
        <p className="text-slate-400 mt-1">
          9-phase multimodal deep learning pipeline for all-weather maritime ship detection.
          Click any phase to view code, outputs, and details.
        </p>
      </div>

      {/* Flow diagram */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 mb-8">
        <h2 className="text-slate-300 font-semibold text-sm mb-4">Pipeline Flow</h2>
        <div className="flex items-center gap-1 overflow-x-auto pb-2">
          {PHASES.map((p, i) => {
            const c = colorMap[p.color] ?? colorMap.sky;
            return (
              <div key={p.phase} className="flex items-center gap-1 flex-shrink-0">
                <button
                  onClick={() => onNavigate(`phase-${p.phase}` as Page)}
                  className={`flex flex-col items-center gap-1 px-3 py-2 rounded-lg border ${c.border} ${c.bg} hover:opacity-80 transition-all group`}
                >
                  <span className={`text-xs font-bold ${c.text}`}>P{p.phase}</span>
                  <span className="text-slate-500 text-xs whitespace-nowrap max-w-16 truncate text-center group-hover:text-slate-300 transition-colors">
                    {p.subtitle.split(' ')[0]}
                  </span>
                </button>
                {i < PHASES.length - 1 && (
                  <ArrowRight size={12} className="text-slate-700 flex-shrink-0" />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Phase cards */}
      <div className="space-y-3">
        {PHASES.map((p) => {
          const c = colorMap[p.color] ?? colorMap.sky;
          return (
            <button
              key={p.phase}
              onClick={() => onNavigate(`phase-${p.phase}` as Page)}
              className="w-full bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-2xl p-5 text-left transition-all group hover:bg-slate-800/50"
            >
              <div className="flex items-start gap-4">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${c.bg} border ${c.border}`}>
                  <span className={`text-lg font-bold ${c.text}`}>{p.phase}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-xs font-semibold uppercase tracking-wider ${c.text}`}>{p.title}</span>
                    <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
                  </div>
                  <h3 className="text-white font-semibold text-base group-hover:text-sky-300 transition-colors">
                    {p.subtitle}
                  </h3>
                  <p className="text-slate-400 text-sm mt-1 leading-relaxed line-clamp-2">{p.description}</p>
                  <div className="flex flex-wrap gap-4 mt-3">
                    <div>
                      <p className="text-slate-600 text-xs uppercase tracking-wider">Input</p>
                      <p className="text-slate-400 text-xs mt-0.5">{p.input}</p>
                    </div>
                    <div>
                      <p className="text-slate-600 text-xs uppercase tracking-wider">Output</p>
                      <p className="text-slate-400 text-xs mt-0.5">{p.output}</p>
                    </div>
                  </div>
                </div>
                <ChevronRight size={16} className="text-slate-600 group-hover:text-sky-400 transition-colors flex-shrink-0 mt-1" />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
