import { useEffect, useState } from 'react';
import {
  User, Mail, Building2, Shield, Clock, Ship,
  BarChart2, Trash2, AlertCircle, CheckCircle, Edit3, Save
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../hooks/useAuth';
import { Profile, DetectionHistory } from '../types';

export default function AccountPage() {
  const { user } = useAuth();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [history, setHistory] = useState<DetectionHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState({ full_name: '', organization: '', role: '' });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    const load = async () => {
      if (!user) return;
      const [profRes, histRes] = await Promise.all([
        supabase.from('profiles').select('*').eq('id', user.id).maybeSingle(),
        supabase.from('detection_history').select('*').eq('user_id', user.id).order('created_at', { ascending: false }),
      ]);
      const prof = profRes.data as Profile | null;
      setProfile(prof);
      setHistory(histRes.data ?? []);
      setForm({
        full_name: prof?.full_name ?? '',
        organization: prof?.organization ?? '',
        role: prof?.role ?? 'researcher',
      });
      setLoading(false);
    };
    load();
  }, [user]);

  const saveProfile = async () => {
    if (!user) return;
    setSaving(true);
    setError('');
    const { error: err } = await supabase.from('profiles').upsert({
      id: user.id,
      ...form,
      updated_at: new Date().toISOString(),
    });
    if (err) setError(err.message);
    else {
      setSuccess('Profile updated');
      setEditing(false);
      setProfile((prev) => prev ? { ...prev, ...form } : null);
    }
    setSaving(false);
  };

  const deleteDetection = async (id: string) => {
    const { error: err } = await supabase.from('detection_history').delete().eq('id', id).eq('user_id', user!.id);
    if (!err) setHistory((prev) => prev.filter((h) => h.id !== id));
  };

  const totalShips = history.reduce((s, h) => s + (h.ship_count ?? 0), 0);
  const avgConf = history.length > 0
    ? (history.reduce((s, h) => s + (h.confidence_avg ?? 0), 0) / history.length * 100).toFixed(1)
    : '0.0';

  if (loading) {
    return (
      <div className="p-8">
        <div className="space-y-4">
          {[1, 2, 3].map((i) => <div key={i} className="h-24 bg-slate-900 border border-slate-800 rounded-2xl animate-pulse" />)}
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
          <User size={14} className="text-sky-400" />
          <span>Account</span>
        </div>
        <h1 className="text-3xl font-bold text-white">My Account</h1>
        <p className="text-slate-400 mt-1">Manage your profile and view detection history</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Profile card */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
            {/* Avatar area */}
            <div className="h-24 bg-gradient-to-br from-sky-900/60 to-teal-900/60 relative">
              <div className="absolute -bottom-8 left-6">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-sky-500 to-teal-500 flex items-center justify-center shadow-xl shadow-sky-500/20 border-4 border-slate-900">
                  <span className="text-white font-bold text-xl">
                    {(profile?.full_name || user?.email || 'A').charAt(0).toUpperCase()}
                  </span>
                </div>
              </div>
            </div>

            <div className="pt-12 px-6 pb-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-white font-semibold text-lg">{profile?.full_name || 'Researcher'}</p>
                  <p className="text-slate-400 text-sm">{profile?.role || 'researcher'}</p>
                </div>
                <button
                  onClick={() => setEditing(!editing)}
                  className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-all"
                >
                  <Edit3 size={14} />
                </button>
              </div>

              {error && (
                <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 mb-3">
                  <AlertCircle size={12} className="text-red-400" />
                  <p className="text-red-400 text-xs">{error}</p>
                </div>
              )}
              {success && (
                <div className="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2 mb-3">
                  <CheckCircle size={12} className="text-emerald-400" />
                  <p className="text-emerald-400 text-xs">{success}</p>
                </div>
              )}

              {editing ? (
                <div className="space-y-3">
                  {[
                    { label: 'Full Name', key: 'full_name', placeholder: 'Dr. Jane Smith' },
                    { label: 'Organization', key: 'organization', placeholder: 'Research Institute' },
                    { label: 'Role', key: 'role', placeholder: 'researcher' },
                  ].map((f) => (
                    <div key={f.key}>
                      <label className="text-slate-500 text-xs mb-1 block">{f.label}</label>
                      <input
                        value={form[f.key as keyof typeof form]}
                        onChange={(e) => setForm((prev) => ({ ...prev, [f.key]: e.target.value }))}
                        placeholder={f.placeholder}
                        className="w-full bg-slate-800 border border-slate-700 text-slate-200 placeholder-slate-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500 transition-all"
                      />
                    </div>
                  ))}
                  <button
                    onClick={saveProfile}
                    disabled={saving}
                    className="w-full flex items-center justify-center gap-2 bg-sky-500/20 text-sky-400 border border-sky-500/30 rounded-lg py-2 text-sm hover:bg-sky-500/30 transition-all disabled:opacity-50"
                  >
                    <Save size={14} />
                    {saving ? 'Saving...' : 'Save Changes'}
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  {[
                    { icon: <Mail size={13} />, label: 'Email', value: user?.email ?? '' },
                    { icon: <Building2 size={13} />, label: 'Organization', value: profile?.organization || 'Not set' },
                    { icon: <Shield size={13} />, label: 'Role', value: profile?.role || 'researcher' },
                    { icon: <Clock size={13} />, label: 'Joined', value: profile?.created_at ? new Date(profile.created_at).toLocaleDateString() : '—' },
                  ].map((f) => (
                    <div key={f.label} className="flex items-center gap-3">
                      <span className="text-slate-600 flex-shrink-0">{f.icon}</span>
                      <div>
                        <p className="text-slate-600 text-xs">{f.label}</p>
                        <p className="text-slate-300 text-sm">{f.value}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Stats */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-slate-400 text-sm font-medium mb-4">Statistics</p>
            <div className="space-y-4">
              {[
                { icon: <BarChart2 size={14} />, label: 'Total Scans', value: history.length.toString(), color: 'text-sky-400' },
                { icon: <Ship size={14} />, label: 'Ships Found', value: totalShips.toString(), color: 'text-teal-400' },
                { icon: <BarChart2 size={14} />, label: 'Avg Confidence', value: `${avgConf}%`, color: 'text-amber-400' },
              ].map((s) => (
                <div key={s.label} className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-slate-500">
                    {s.icon}
                    <span className="text-sm">{s.label}</span>
                  </div>
                  <span className={`font-bold text-sm ${s.color}`}>{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Detection history */}
        <div className="lg:col-span-2">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800">
              <h2 className="text-slate-200 font-semibold flex items-center gap-2">
                <Clock size={16} className="text-sky-400" />
                Detection History
                <span className="ml-auto text-slate-600 text-sm font-normal">{history.length} records</span>
              </h2>
            </div>

            {history.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16">
                <BarChart2 size={32} className="text-slate-700 mb-3" />
                <p className="text-slate-500 text-sm">No detection history</p>
                <p className="text-slate-600 text-xs mt-1">Run ship detection to build history</p>
              </div>
            ) : (
              <div className="divide-y divide-slate-800">
                {history.map((h) => (
                  <div key={h.id} className="px-6 py-4 flex items-center gap-4 hover:bg-slate-800/30 transition-all group">
                    <div className="w-10 h-10 rounded-xl bg-sky-500/15 flex items-center justify-center flex-shrink-0">
                      <Ship size={16} className="text-sky-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-slate-200 text-sm font-medium truncate">{h.filename}</p>
                      <div className="flex items-center gap-3 mt-0.5">
                        <span className="text-slate-500 text-xs">{new Date(h.created_at).toLocaleString()}</span>
                        <span className="text-slate-700 text-xs">·</span>
                        <span className="text-slate-500 text-xs capitalize">{h.input_type}</span>
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      <p className="text-white font-bold text-sm">{h.ship_count} ships</p>
                      <p className="text-slate-500 text-xs">{(h.confidence_avg * 100).toFixed(0)}% confidence</p>
                    </div>
                    <button
                      onClick={() => deleteDetection(h.id)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-red-400 hover:bg-red-500/10 transition-all flex-shrink-0"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
