import { useEffect, useState, useRef } from 'react';
import {
  ArrowLeft, Upload, Trash2, Code2, FileText, ChevronDown, ChevronRight,
  AlertCircle, CheckCircle, Play, BookOpen
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../hooks/useAuth';
import { PHASES, NotebookFile, NotebookCell, CellOutput } from '../types';

interface Props {
  phase: number;
  onBack: () => void;
}

const colorMap: Record<string, { bg: string; text: string; border: string }> = {
  sky:     { bg: 'bg-sky-500/10',     text: 'text-sky-400',     border: 'border-sky-500/30' },
  teal:    { bg: 'bg-teal-500/10',    text: 'text-teal-400',    border: 'border-teal-500/30' },
  cyan:    { bg: 'bg-cyan-500/10',    text: 'text-cyan-400',    border: 'border-cyan-500/30' },
  blue:    { bg: 'bg-blue-500/10',    text: 'text-blue-400',    border: 'border-blue-500/30' },
  amber:   { bg: 'bg-amber-500/10',   text: 'text-amber-400',   border: 'border-amber-500/30' },
  orange:  { bg: 'bg-orange-500/10',  text: 'text-orange-400',  border: 'border-orange-500/30' },
  emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30' },
};

function renderSource(src: string | string[]): string {
  return Array.isArray(src) ? src.join('') : src;
}

function CellOutputView({ output }: { output: CellOutput }) {
  if (output.output_type === 'stream') {
    const text = renderSource(output.text ?? '');
    return (
      <pre className="text-xs text-slate-300 whitespace-pre-wrap font-mono leading-relaxed">
        {text}
      </pre>
    );
  }
  if (output.output_type === 'error') {
    return (
      <div className="text-red-400 text-xs font-mono">
        <p className="font-bold">{output.ename}: {output.evalue}</p>
        {(output.traceback ?? []).map((line, i) => (
          <p key={i} className="mt-0.5 opacity-70">{line.replace(/\x1b\[[0-9;]*m/g, '')}</p>
        ))}
      </div>
    );
  }
  if (output.output_type === 'display_data' || output.output_type === 'execute_result') {
    const data = output.data ?? {};
    if (data['image/png']) {
      return <img src={`data:image/png;base64,${data['image/png']}`} alt="output" className="max-w-full rounded-lg mt-1" />;
    }
    if (data['text/plain']) {
      const text = renderSource(data['text/plain'] as string | string[]);
      return <pre className="text-xs text-slate-300 whitespace-pre-wrap font-mono">{text}</pre>;
    }
    if (data['text/html']) {
      const html = renderSource(data['text/html'] as string | string[]);
      return <div className="text-xs text-slate-300" dangerouslySetInnerHTML={{ __html: html }} />;
    }
  }
  return null;
}

function NotebookCellView({ cell, index }: { cell: NotebookCell; index: number }) {
  const [collapsed, setCollapsed] = useState(false);
  const src = renderSource(cell.source);
  const hasOutput = (cell.outputs ?? []).length > 0;

  return (
    <div className={`border rounded-xl overflow-hidden mb-3 ${
      cell.cell_type === 'code'
        ? 'border-slate-700/50 bg-slate-900'
        : 'border-slate-800/30 bg-slate-900/50'
    }`}>
      {/* Cell header */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800/50">
        {cell.cell_type === 'code'
          ? <Code2 size={12} className="text-sky-400" />
          : <FileText size={12} className="text-slate-500" />}
        <span className="text-slate-600 text-xs">
          {cell.cell_type === 'code' ? `In [${cell.execution_count ?? ' '}]` : 'Markdown'}
        </span>
        <span className="text-slate-700 text-xs ml-auto">Cell {index + 1}</span>
        <button onClick={() => setCollapsed(!collapsed)} className="text-slate-600 hover:text-slate-400 transition-colors">
          {collapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
        </button>
      </div>

      {!collapsed && (
        <>
          {/* Source */}
          {src.trim() && (
            <div className="px-4 py-3">
              {cell.cell_type === 'code' ? (
                <pre className="text-xs text-slate-200 font-mono whitespace-pre-wrap leading-relaxed overflow-x-auto">
                  <code>{src}</code>
                </pre>
              ) : (
                <div className="text-sm text-slate-300 prose prose-invert prose-sm max-w-none">
                  {src.split('\n').map((line, i) => {
                    if (line.startsWith('# ')) return <h1 key={i} className="text-white font-bold text-lg">{line.slice(2)}</h1>;
                    if (line.startsWith('## ')) return <h2 key={i} className="text-slate-200 font-semibold text-base">{line.slice(3)}</h2>;
                    if (line.startsWith('### ')) return <h3 key={i} className="text-slate-300 font-semibold text-sm">{line.slice(4)}</h3>;
                    if (line.startsWith('- ') || line.startsWith('* ')) return <li key={i} className="text-slate-400 text-sm ml-4">{line.slice(2)}</li>;
                    if (line.trim() === '') return <br key={i} />;
                    return <p key={i} className="text-slate-400 text-sm">{line}</p>;
                  })}
                </div>
              )}
            </div>
          )}

          {/* Outputs */}
          {hasOutput && cell.cell_type === 'code' && (
            <div className="border-t border-slate-800/50 bg-slate-950/50 px-4 py-3 space-y-2">
              <p className="text-slate-600 text-xs mb-2">Output:</p>
              {(cell.outputs ?? []).map((out, i) => (
                <CellOutputView key={i} output={out} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function PhasePage({ phase, onBack }: Props) {
  const { user } = useAuth();
  const [notebook, setNotebook] = useState<NotebookFile | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const phaseInfo = PHASES[phase];
  const c = colorMap[phaseInfo?.color ?? 'sky'] ?? colorMap.sky;

  useEffect(() => {
    loadNotebook();
  }, [phase, user]);

  const loadNotebook = async () => {
    if (!user) return;
    setLoading(true);
    const { data } = await supabase
      .from('notebooks')
      .select('*')
      .eq('user_id', user.id)
      .eq('phase', phase)
      .order('uploaded_at', { ascending: false })
      .limit(1)
      .maybeSingle();
    setNotebook(data);
    setLoading(false);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !user) return;
    if (!file.name.endsWith('.ipynb')) {
      setError('Please upload a .ipynb file');
      return;
    }
    setUploading(true);
    setError('');
    setSuccess('');
    try {
      const text = await file.text();
      const content = JSON.parse(text);
      const { error: err } = await supabase.from('notebooks').insert({
        user_id: user.id,
        phase,
        filename: file.name,
        content,
        file_size: file.size,
      });
      if (err) throw err;
      setSuccess('Notebook uploaded successfully');
      await loadNotebook();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const handleDelete = async () => {
    if (!notebook || !user) return;
    if (!confirm('Delete this notebook? This cannot be undone.')) return;
    setDeleting(true);
    setError('');
    const { error: err } = await supabase.from('notebooks').delete().eq('id', notebook.id).eq('user_id', user.id);
    if (err) {
      setError(err.message);
    } else {
      setNotebook(null);
      setSuccess('Notebook deleted');
    }
    setDeleting(false);
  };

  if (!phaseInfo) return <div className="p-8 text-slate-400">Unknown phase</div>;

  const cells = notebook?.content?.cells ?? [];

  return (
    <div className="p-8">
      {/* Back */}
      <button onClick={onBack} className="flex items-center gap-2 text-slate-400 hover:text-slate-200 text-sm mb-6 transition-colors group">
        <ArrowLeft size={14} className="group-hover:-translate-x-0.5 transition-transform" />
        Back to Pipeline
      </button>

      {/* Header */}
      <div className="flex items-start gap-4 mb-8">
        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0 ${c.bg} border ${c.border}`}>
          <span className={`text-2xl font-bold ${c.text}`}>{phase}</span>
        </div>
        <div className="flex-1">
          <p className={`text-xs font-semibold uppercase tracking-widest ${c.text} mb-1`}>{phaseInfo.title}</p>
          <h1 className="text-3xl font-bold text-white">{phaseInfo.subtitle}</h1>
          <p className="text-slate-400 mt-2 leading-relaxed max-w-3xl">{phaseInfo.description}</p>
        </div>
      </div>

      {/* Input / Output */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
        <div className={`${c.bg} border ${c.border} rounded-xl p-4`}>
          <p className={`text-xs font-semibold uppercase tracking-wider ${c.text} mb-2`}>Input</p>
          <p className="text-slate-300 text-sm leading-relaxed">{phaseInfo.input}</p>
        </div>
        <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-emerald-400 mb-2">Output</p>
          <p className="text-slate-300 text-sm leading-relaxed">{phaseInfo.output}</p>
        </div>
      </div>

      {/* Notebook section */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <BookOpen size={16} className="text-sky-400" />
            <span className="text-slate-200 font-semibold text-sm">
              {notebook ? notebook.filename : 'Jupyter Notebook'}
            </span>
            {notebook && (
              <span className="text-slate-600 text-xs">
                {cells.length} cells · {(notebook.file_size / 1024).toFixed(0)}KB
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {notebook && (
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg border border-red-500/20 transition-all disabled:opacity-50"
              >
                <Trash2 size={12} />
                {deleting ? 'Deleting...' : 'Delete'}
              </button>
            )}
            <input ref={fileRef} type="file" accept=".ipynb" className="hidden" onChange={handleUpload} />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-sky-500/20 text-sky-400 hover:bg-sky-500/30 rounded-lg border border-sky-500/30 transition-all disabled:opacity-50"
            >
              <Upload size={12} />
              {uploading ? 'Uploading...' : 'Upload .ipynb'}
            </button>
          </div>
        </div>

        {/* Messages */}
        {error && (
          <div className="flex items-center gap-2 bg-red-500/10 border-b border-red-500/20 px-5 py-3">
            <AlertCircle size={14} className="text-red-400" />
            <p className="text-red-400 text-xs">{error}</p>
          </div>
        )}
        {success && (
          <div className="flex items-center gap-2 bg-emerald-500/10 border-b border-emerald-500/20 px-5 py-3">
            <CheckCircle size={14} className="text-emerald-400" />
            <p className="text-emerald-400 text-xs">{success}</p>
          </div>
        )}

        {/* Content */}
        <div className="p-5">
          {loading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => <div key={i} className="h-20 bg-slate-800/50 rounded-xl animate-pulse" />)}
            </div>
          ) : !notebook ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center mb-4">
                <Play size={24} className="text-slate-600" />
              </div>
              <p className="text-slate-300 font-medium mb-1">No notebook uploaded</p>
              <p className="text-slate-500 text-sm mb-6 max-w-sm">
                Upload the corresponding Jupyter notebook (.ipynb) file for this phase to view its code and outputs.
              </p>
              <p className="text-slate-600 text-xs bg-slate-800 rounded-lg px-4 py-2">
                Expected: PHASE{phase}*.ipynb
              </p>
              <button
                onClick={() => fileRef.current?.click()}
                className="mt-4 flex items-center gap-2 bg-sky-500/20 text-sky-400 border border-sky-500/30 px-5 py-2.5 rounded-xl text-sm hover:bg-sky-500/30 transition-all"
              >
                <Upload size={14} />
                Upload Notebook
              </button>
            </div>
          ) : cells.length === 0 ? (
            <p className="text-slate-500 text-sm text-center py-8">No cells found in notebook</p>
          ) : (
            <div>
              {cells.map((cell, i) => (
                <NotebookCellView key={i} cell={cell} index={i} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
