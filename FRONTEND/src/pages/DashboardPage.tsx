import { useEffect, useState } from 'react';
import {
  Radar, ScanSearch, BarChart2, Ship, TrendingUp, Clock,
  Activity, Waves, ChevronRight, Network, AlertTriangle
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../hooks/useAuth';
import { DetectionHistory, PHASES } from '../types';
import { Page } from '../App';

interface Props {
  onNavigate: (page: Page) => void;
}

export default function DashboardPage({ onNavigate }: Props) {
  const { user } = useAuth();
  const [history, setHistory] = useState<DetectionHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [profile, setProfile] = useState<{ full_name: string; organization: string } | null>(null);

  useEffect(() => {
    const load = async () => {
      if (!user) return;
      const [histRes, profRes] = await Promise.all([
        supabase.from('detection_history').select('*').eq('user_id', user.id).order('created_at', { ascending: false }).limit(5),
        supabase.from('profiles').select('full_name, organization').eq('id', user.id).maybeSingle(),
      ]);
      setHistory(histRes.data ?? []);
      setProfile(profRes.data);
      setLoading(false);
    };
    load();
  }, [user]);

  const totalDetections = history.length;
  const totalShips = history.reduce((sum, h) => sum + (h.ship_count ?? 0), 0);
  const avgConf = history.length > 0
    ? (history.reduce((sum, h) => sum + (h.confidence_avg ?? 0), 0) / history.length * 100).toFixed(1)
    : '0.0';

  const firstName = profile?.full_name?.split(' ')[0] || user?.email?.split('@')[0] || 'Analyst';

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-emerald-400 text-xs font-medium uppercase tracking-widest">System Active</span>
        </div>
        <h1 className="text-3xl font-bold text-white">
          Welcome back, {firstName}
        </h1>
        <p className="text-slate-400 mt-1">
          Multimodal Maritime Surveillance — SAR &amp; Optical Fusion Platform
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[
          {
            icon: <ScanSearch size={20} />,
            label: 'Total Scans',
            value: loading ? '—' : totalDetections.toString(),
            sub: 'Image + video',
            color: 'sky',
          },
          {
            icon: <Ship size={20} />,
            label: 'Ships Detected',
            value: loading ? '—' : totalShips.toString(),
            sub: 'All time',
            color: 'teal',
          },
          {
            icon: <BarChart2 size={20} />,
            label: 'Avg Confidence',
            value: loading ? '—' : `${avgConf}%`,
            sub: 'Per detection',
            color: 'amber',
          },
          {
            icon: <Activity size={20} />,
            label: 'Pipeline Phases',
            value: '9',
            sub: 'Phase 0–8',
            color: 'emerald',
          },
        ].map((s) => (
          <div key={s.label} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 hover:border-slate-700 transition-all">
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 ${
              s.color === 'sky' ? 'bg-sky-500/15 text-sky-400' :
              s.color === 'teal' ? 'bg-teal-500/15 text-teal-400' :
              s.color === 'amber' ? 'bg-amber-500/15 text-amber-400' :
              'bg-emerald-500/15 text-emerald-400'
            }`}>
              {s.icon}
            </div>
            <p className="text-2xl font-bold text-white">{s.value}</p>
            <p className="text-slate-300 text-sm font-medium mt-0.5">{s.label}</p>
            <p className="text-slate-600 text-xs mt-0.5">{s.sub}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick actions */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <h2 className="text-slate-200 font-semibold mb-4 flex items-center gap-2">
            <Waves size={16} className="text-sky-400" />
            Quick Actions
          </h2>
          <div className="space-y-2">
            {[
              { icon: <ScanSearch size={16} />, label: 'Run Ship Detection', sub: 'Upload SAR/optical image', page: 'detection' as Page, color: 'sky' },
              { icon: <Radar size={16} />, label: 'View Pipeline', sub: 'All 9 phases', page: 'pipeline' as Page, color: 'teal' },
              { icon: <Network size={16} />, label: 'System Architecture', sub: 'Model structure', page: 'architecture' as Page, color: 'amber' },
            ].map((a) => (
              <button
                key={a.label}
                onClick={() => onNavigate(a.page)}
                className="w-full flex items-center gap-3 p-3 rounded-xl bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 hover:border-slate-600 transition-all group"
              >
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                  a.color === 'sky' ? 'bg-sky-500/15 text-sky-400' :
                  a.color === 'teal' ? 'bg-teal-500/15 text-teal-400' :
                  'bg-amber-500/15 text-amber-400'
                }`}>
                  {a.icon}
                </div>
                <div className="flex-1 text-left">
                  <p className="text-slate-200 text-sm font-medium">{a.label}</p>
                  <p className="text-slate-500 text-xs">{a.sub}</p>
                </div>
                <ChevronRight size={14} className="text-slate-600 group-hover:text-slate-400 transition-colors" />
              </button>
            ))}
          </div>
        </div>

        {/* Recent history */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-slate-200 font-semibold flex items-center gap-2">
              <Clock size={16} className="text-sky-400" />
              Recent Detections
            </h2>
            <button onClick={() => onNavigate('account')} className="text-sky-400 text-xs hover:text-sky-300 transition-colors">
              View all
            </button>
          </div>

          {loading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-14 bg-slate-800/50 rounded-xl animate-pulse" />
              ))}
            </div>
          ) : history.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <AlertTriangle size={32} className="text-slate-700 mb-3" />
              <p className="text-slate-500 text-sm">No detections yet</p>
              <p className="text-slate-600 text-xs mt-1">Run ship detection to see results here</p>
              <button
                onClick={() => onNavigate('detection')}
                className="mt-4 bg-sky-500/20 text-sky-400 border border-sky-500/30 px-4 py-2 rounded-lg text-xs hover:bg-sky-500/30 transition-all"
              >
                Start Detection
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {history.map((h) => (
                <div key={h.id} className="flex items-center gap-4 p-3 bg-slate-800/40 rounded-xl border border-slate-700/30 hover:border-slate-600/50 transition-all">
                  <div className="w-8 h-8 rounded-lg bg-sky-500/15 flex items-center justify-center flex-shrink-0">
                    {h.input_type === 'video'
                      ? <TrendingUp size={14} className="text-sky-400" />
                      : <ScanSearch size={14} className="text-sky-400" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-slate-200 text-sm font-medium truncate">{h.filename}</p>
                    <p className="text-slate-500 text-xs">{new Date(h.created_at).toLocaleDateString()}</p>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <p className="text-slate-200 text-sm font-bold">{h.ship_count} ships</p>
                    <p className="text-slate-500 text-xs">{(h.confidence_avg * 100).toFixed(0)}% conf</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Pipeline overview */}
      <div className="mt-6 bg-slate-900 border border-slate-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-slate-200 font-semibold flex items-center gap-2">
            <Radar size={16} className="text-sky-400" />
            Pipeline Overview — 9 Phases
          </h2>
          <button onClick={() => onNavigate('pipeline')} className="text-sky-400 text-xs hover:text-sky-300 transition-colors flex items-center gap-1">
            View details <ChevronRight size={12} />
          </button>
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-9 gap-2">
          {PHASES.map((p) => (
            <button
              key={p.phase}
              onClick={() => onNavigate(`phase-${p.phase}` as Page)}
              className="flex flex-col items-center gap-2 p-3 bg-slate-800/50 hover:bg-slate-800 rounded-xl border border-slate-700/30 hover:border-slate-600 transition-all group"
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${
                p.color === 'sky' ? 'bg-sky-500/20 text-sky-400' :
                p.color === 'teal' ? 'bg-teal-500/20 text-teal-400' :
                p.color === 'cyan' ? 'bg-cyan-500/20 text-cyan-400' :
                p.color === 'blue' ? 'bg-blue-500/20 text-blue-400' :
                p.color === 'amber' ? 'bg-amber-500/20 text-amber-400' :
                p.color === 'orange' ? 'bg-orange-500/20 text-orange-400' :
                'bg-emerald-500/20 text-emerald-400'
              }`}>
                P{p.phase}
              </div>
              <p className="text-slate-500 text-xs text-center leading-tight group-hover:text-slate-300 transition-colors line-clamp-2">{p.subtitle.split(' ').slice(0, 2).join(' ')}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
