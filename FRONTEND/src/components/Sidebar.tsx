import { ReactNode } from 'react';
import {
  LayoutDashboard, Radar, ScanSearch, User, Network,
  Waves, ChevronRight, LogOut, Activity
} from 'lucide-react';
import { Page } from '../App';
import { useAuth } from '../hooks/useAuth';
import { PHASES } from '../types';

interface Props {
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

const phaseColors: Record<number, string> = {
  0: 'sky', 1: 'teal', 2: 'cyan', 3: 'blue',
  4: 'sky', 5: 'teal', 6: 'amber', 7: 'orange', 8: 'emerald',
};

const dotClass: Record<string, string> = {
  sky: 'bg-sky-400',
  teal: 'bg-teal-400',
  cyan: 'bg-cyan-400',
  blue: 'bg-blue-400',
  amber: 'bg-amber-400',
  orange: 'bg-orange-400',
  emerald: 'bg-emerald-400',
};

function navItem(page: Page, icon: ReactNode, label: string, currentPage: Page, onNavigate: (page: Page) => void, badge?: string) {
  const active = currentPage === page;
  return (
    <button
      key={page}
      onClick={() => onNavigate(page)}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group ${
        active
          ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30'
          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
      }`}
    >
      <span className={`flex-shrink-0 ${active ? 'text-sky-400' : 'text-slate-500 group-hover:text-slate-300'}`}>
        {icon}
      </span>
      <span className="flex-1 text-left">{label}</span>
      {badge && (
        <span className="text-xs bg-sky-500/20 text-sky-400 px-1.5 py-0.5 rounded-full">{badge}</span>
      )}
      {active && <ChevronRight size={14} className="text-sky-400" />}
    </button>
  );
}

export default function Sidebar({ currentPage, onNavigate }: Props) {
  const { signOut, user } = useAuth();

  return (
    <aside className="w-64 flex-shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col h-full">
      {/* Logo */}
      <div className="p-5 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-sky-500 to-teal-500 flex items-center justify-center shadow-lg shadow-sky-500/20">
            <Waves size={18} className="text-white" />
          </div>
          <div>
            <p className="text-white font-bold text-sm leading-none">MarineVision</p>
            <p className="text-slate-500 text-xs mt-0.5">SAR + Optical Fusion</p>
          </div>
        </div>
      </div>

      {/* Status indicator */}
      <div className="px-4 py-3 border-b border-slate-800">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Activity size={12} className="text-emerald-400" />
          <span className="text-emerald-400">System Online</span>
          <span className="ml-auto text-slate-600">Panel 5</span>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-1">
        <p className="text-slate-600 text-xs font-semibold uppercase tracking-wider px-3 pb-2 pt-1">Navigation</p>
        {navItem('dashboard', <LayoutDashboard size={16} />, 'Dashboard', currentPage, onNavigate)}
        {navItem('detection', <ScanSearch size={16} />, 'Ship Detection', currentPage, onNavigate)}
        {navItem('architecture', <Network size={16} />, 'Architecture', currentPage, onNavigate)}
        {navItem('pipeline', <Radar size={16} />, 'Pipeline', currentPage, onNavigate, '9')}

        {/* Phase list */}
        <div className="pt-3">
          <p className="text-slate-600 text-xs font-semibold uppercase tracking-wider px-3 pb-2">Pipeline Phases</p>
          <div className="space-y-0.5">
            {PHASES.map((p) => {
              const page = `phase-${p.phase}` as Page;
              const active = currentPage === page;
              const dot = dotClass[phaseColors[p.phase]] ?? 'bg-sky-400';
              return (
                <button
                  key={p.phase}
                  onClick={() => onNavigate(page)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs transition-all duration-200 ${
                    active
                      ? 'bg-slate-800 text-slate-200'
                      : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/40'
                  }`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dot}`} />
                  <span className="font-medium">P{p.phase}</span>
                  <span className="text-slate-600 truncate flex-1 text-left">{p.subtitle.split(' ').slice(0, 2).join(' ')}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Bottom */}
      <div className="p-3 border-t border-slate-800 space-y-1">
        {navItem('account', <User size={16} />, 'My Account', currentPage, onNavigate)}
        <button
          onClick={signOut}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-all duration-200"
        >
          <LogOut size={16} />
          <span>Sign Out</span>
        </button>
        <div className="px-3 pt-2">
          <p className="text-slate-600 text-xs truncate">{user?.email}</p>
        </div>
      </div>
    </aside>
  );
}
