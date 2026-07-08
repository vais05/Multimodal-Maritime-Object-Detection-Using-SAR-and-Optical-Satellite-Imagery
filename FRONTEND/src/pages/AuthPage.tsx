import { useState } from 'react';
import { Waves, Eye, EyeOff, Lock, Mail, User, AlertCircle } from 'lucide-react';
import { supabase } from '../lib/supabase';

export default function AuthPage() {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [organization, setOrganization] = useState('');
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (mode === 'login') {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
      } else {
        const { error } = await supabase.auth.signUp({
          email, password,
          options: { data: { full_name: fullName, organization } },
        });
        if (error) throw error;
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex overflow-hidden">
      {/* Left panel — hero */}
      <div className="hidden lg:flex flex-1 relative overflow-hidden bg-gradient-to-br from-slate-900 via-sky-950 to-slate-950">
        <div className="absolute inset-0">
          <div className="absolute top-20 left-20 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-20 right-20 w-80 h-80 bg-teal-500/10 rounded-full blur-3xl" />
          {/* Grid overlay */}
          <div className="absolute inset-0" style={{
            backgroundImage: 'linear-gradient(rgba(148,163,184,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.03) 1px, transparent 1px)',
            backgroundSize: '50px 50px'
          }} />
        </div>
        <div className="relative z-10 flex flex-col justify-center px-16">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-sky-500 to-teal-500 flex items-center justify-center shadow-2xl shadow-sky-500/30 mb-8">
            <Waves size={28} className="text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-4 leading-tight">
            Maritime<br />Surveillance<br />
            <span className="text-sky-400">Intelligence</span>
          </h1>
          <p className="text-slate-400 text-lg leading-relaxed max-w-sm">
            Multimodal SAR + optical deep learning platform for all-weather ship detection and maritime monitoring.
          </p>
          <div className="mt-12 grid grid-cols-2 gap-4">
            {[
              { label: 'Accuracy', value: 'mAP@0.5' },
              { label: 'Modalities', value: 'SAR + Optical' },
              { label: 'Backbone', value: 'ResNet50 + CSP' },
              { label: 'Fusion', value: 'CrossModal Attn' },
            ].map((s) => (
              <div key={s.label} className="bg-slate-800/40 backdrop-blur border border-slate-700/40 rounded-xl p-4">
                <p className="text-sky-400 font-bold text-sm">{s.value}</p>
                <p className="text-slate-500 text-xs mt-1">{s.label}</p>
              </div>
            ))}
          </div>
          <div className="mt-10 text-slate-600 text-sm">
            Team 22UG0005 &middot; Panel 5 &middot; Guide: Mr. Nagesh Koundinya Subbanna
          </div>
        </div>
      </div>

      {/* Right panel — form */}
      <div className="flex-1 lg:max-w-md flex items-center justify-center p-8">
        <div className="w-full max-w-sm">
          <div className="lg:hidden flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-teal-500 flex items-center justify-center">
              <Waves size={20} className="text-white" />
            </div>
            <div>
              <p className="text-white font-bold">MarineVision</p>
              <p className="text-slate-500 text-xs">SAR + Optical Fusion</p>
            </div>
          </div>

          <h2 className="text-2xl font-bold text-white mb-1">
            {mode === 'login' ? 'Welcome back' : 'Create account'}
          </h2>
          <p className="text-slate-400 text-sm mb-8">
            {mode === 'login'
              ? 'Sign in to access the surveillance platform'
              : 'Join the maritime intelligence platform'}
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === 'register' && (
              <>
                <div>
                  <label className="text-slate-400 text-xs font-medium block mb-1.5">Full Name</label>
                  <div className="relative">
                    <User size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                    <input
                      type="text"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      placeholder="Dr. Jane Smith"
                      className="w-full bg-slate-800/60 border border-slate-700 text-slate-200 placeholder-slate-600 rounded-lg pl-9 pr-4 py-2.5 text-sm focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30 transition-all"
                    />
                  </div>
                </div>
                <div>
                  <label className="text-slate-400 text-xs font-medium block mb-1.5">Organization</label>
                  <div className="relative">
                    <User size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                    <input
                      type="text"
                      value={organization}
                      onChange={(e) => setOrganization(e.target.value)}
                      placeholder="Research Institute"
                      className="w-full bg-slate-800/60 border border-slate-700 text-slate-200 placeholder-slate-600 rounded-lg pl-9 pr-4 py-2.5 text-sm focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30 transition-all"
                    />
                  </div>
                </div>
              </>
            )}

            <div>
              <label className="text-slate-400 text-xs font-medium block mb-1.5">Email</label>
              <div className="relative">
                <Mail size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="analyst@maritime.gov"
                  required
                  className="w-full bg-slate-800/60 border border-slate-700 text-slate-200 placeholder-slate-600 rounded-lg pl-9 pr-4 py-2.5 text-sm focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30 transition-all"
                />
              </div>
            </div>

            <div>
              <label className="text-slate-400 text-xs font-medium block mb-1.5">Password</label>
              <div className="relative">
                <Lock size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input
                  type={showPass ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  className="w-full bg-slate-800/60 border border-slate-700 text-slate-200 placeholder-slate-600 rounded-lg pl-9 pr-10 py-2.5 text-sm focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30 transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPass(!showPass)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                >
                  {showPass ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                <AlertCircle size={14} className="text-red-400 flex-shrink-0" />
                <p className="text-red-400 text-xs">{error}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-sky-600 to-teal-600 hover:from-sky-500 hover:to-teal-500 text-white font-semibold py-2.5 rounded-lg text-sm transition-all duration-200 shadow-lg shadow-sky-500/20 disabled:opacity-50 disabled:cursor-not-allowed mt-2"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {mode === 'login' ? 'Signing in...' : 'Creating account...'}
                </span>
              ) : (
                mode === 'login' ? 'Sign In' : 'Create Account'
              )}
            </button>
          </form>

          <p className="text-center text-slate-500 text-sm mt-6">
            {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
            <button
              onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(''); }}
              className="text-sky-400 hover:text-sky-300 font-medium transition-colors"
            >
              {mode === 'login' ? 'Register' : 'Sign In'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
