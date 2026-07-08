import { useState, useRef, useCallback } from 'react';
import {
  Upload, ScanSearch, AlertCircle, CheckCircle, X,
  Ship, Crosshair, Clock, Activity, Info, Image, Video
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../hooks/useAuth';
import { Detection } from '../types';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000';

interface DetectionResponse {
  detections: Detection[];
  ship_count: number;
  confidence_avg: number;
  processing_time_ms: number;
  annotated_image?: string;
  frames?: { frame_index: number; timestamp: number; detections: Detection[]; image?: string }[];
  error?: string;
}

export default function DetectionPage() {
  const { user } = useAuth();
  const [inputType, setInputType] = useState<'image' | 'video'>('image');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleFile = (f: File) => {
    const isVideo = f.type.startsWith('video/');
    setInputType(isVideo ? 'video' : 'image');
    setFile(f);
    setResult(null);
    setError('');
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const runDetection = async () => {
    if (!file || !user) return;
    setRunning(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('input_type', inputType);

    try {
      const res = await fetch(`${BACKEND_URL}/detect`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data: DetectionResponse = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);

      // Save to history
      await supabase.from('detection_history').insert({
        user_id: user.id,
        input_type: inputType,
        filename: file.name,
        ship_count: data.ship_count,
        confidence_avg: data.confidence_avg,
        processing_time_ms: data.processing_time_ms,
        result_json: data,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Detection failed';
      // Demo mode — generate mock result
      const mock: DetectionResponse = {
        detections: [
          { bbox: [120, 80, 200, 140], confidence: 0.91, class: 'ship' },
          { bbox: [300, 200, 380, 260], confidence: 0.85, class: 'ship' },
          { bbox: [450, 120, 510, 170], confidence: 0.78, class: 'ship' },
        ],
        ship_count: 3,
        confidence_avg: 0.847,
        processing_time_ms: 342,
      };
      setResult(mock);
      setError(`Backend not reachable (${msg}). Showing demo results.`);
    } finally {
      setRunning(false);
    }
  };

  const confColor = (c: number) => c >= 0.85 ? 'text-emerald-400' : c >= 0.7 ? 'text-amber-400' : 'text-red-400';

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
          <ScanSearch size={14} className="text-sky-400" />
          <span>Detection</span>
        </div>
        <h1 className="text-3xl font-bold text-white">Ship Detection</h1>
        <p className="text-slate-400 mt-1">
          Upload SAR or optical satellite imagery. The model will detect and localise ships with bounding boxes.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload panel */}
        <div className="space-y-4">
          {/* Mode toggle */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-1 flex gap-1">
            {(['image', 'video'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setInputType(t)}
                className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all ${
                  inputType === t
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {t === 'image' ? <Image size={14} /> : <Video size={14} />}
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
            className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
              dragOver
                ? 'border-sky-500 bg-sky-500/10'
                : 'border-slate-700 hover:border-slate-600 bg-slate-900 hover:bg-slate-800/50'
            }`}
          >
            <input
              ref={fileRef}
              type="file"
              className="hidden"
              accept={inputType === 'image' ? 'image/*' : 'video/*'}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
            />
            {preview && inputType === 'image' ? (
              <div className="relative">
                <img src={preview} alt="preview" className="max-h-48 mx-auto rounded-xl object-contain" />
                <button
                  onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setResult(null); }}
                  className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center hover:bg-red-400 transition-colors"
                >
                  <X size={12} className="text-white" />
                </button>
              </div>
            ) : (
              <>
                <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center mx-auto mb-4">
                  <Upload size={24} className="text-slate-500" />
                </div>
                <p className="text-slate-300 font-medium mb-1">
                  Drop your {inputType} here
                </p>
                <p className="text-slate-500 text-sm">
                  {inputType === 'image' ? 'PNG, JPG, TIFF — SAR or optical' : 'MP4, AVI, MOV'}
                </p>
              </>
            )}
            {file && (
              <p className="text-slate-400 text-xs mt-3 truncate max-w-xs mx-auto">{file.name}</p>
            )}
          </div>

          {/* Run button */}
          <button
            onClick={runDetection}
            disabled={!file || running}
            className="w-full bg-gradient-to-r from-sky-600 to-teal-600 hover:from-sky-500 hover:to-teal-500 text-white font-semibold py-3 rounded-xl text-sm transition-all shadow-lg shadow-sky-500/20 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {running ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Running inference...
              </>
            ) : (
              <>
                <ScanSearch size={16} />
                Run Detection
              </>
            )}
          </button>

          {/* Info box */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Info size={14} className="text-sky-400" />
              <span className="text-slate-300 text-sm font-medium">Model Info</span>
            </div>
            <div className="space-y-2 text-xs">
              {[
                ['Model', 'best_model.pth (SAR+Optical fusion)'],
                ['Backbone', 'ResNet50 (SAR) + CSPDarknet (Optical)'],
                ['Fusion', 'CrossModalAttention + FPN'],
                ['Classes', 'Ship (1 class)'],
                ['Input size', '320×320 → 640×640'],
                ['NMS threshold', '0.45'],
                ['Confidence threshold', '0.50'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between gap-4">
                  <span className="text-slate-600">{k}</span>
                  <span className="text-slate-400 text-right">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Results panel */}
        <div className="space-y-4">
          {error && (
            <div className="flex items-start gap-2 bg-amber-500/10 border border-amber-500/20 rounded-xl px-4 py-3">
              <AlertCircle size={14} className="text-amber-400 flex-shrink-0 mt-0.5" />
              <p className="text-amber-400 text-xs">{error}</p>
            </div>
          )}

          {result && (
            <>
              {/* Stats */}
              <div className="grid grid-cols-3 gap-3">
                {[
                  { icon: <Ship size={16} />, label: 'Ships Found', value: result.ship_count.toString(), color: 'sky' },
                  { icon: <Crosshair size={16} />, label: 'Avg Confidence', value: `${(result.confidence_avg * 100).toFixed(1)}%`, color: 'emerald' },
                  { icon: <Clock size={16} />, label: 'Processing', value: `${result.processing_time_ms}ms`, color: 'amber' },
                ].map((s) => (
                  <div key={s.label} className={`bg-slate-900 border border-slate-800 rounded-xl p-4 text-center`}>
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center mx-auto mb-2 ${
                      s.color === 'sky' ? 'bg-sky-500/15 text-sky-400' :
                      s.color === 'emerald' ? 'bg-emerald-500/15 text-emerald-400' :
                      'bg-amber-500/15 text-amber-400'
                    }`}>
                      {s.icon}
                    </div>
                    <p className="text-white font-bold text-lg">{s.value}</p>
                    <p className="text-slate-500 text-xs">{s.label}</p>
                  </div>
                ))}
              </div>

              {/* Annotated image */}
              {result.annotated_image && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2">
                    <Activity size={14} className="text-sky-400" />
                    <span className="text-slate-300 text-sm font-medium">Detection Output</span>
                    <span className="ml-auto text-emerald-400 text-xs flex items-center gap-1">
                      <CheckCircle size={10} /> Complete
                    </span>
                  </div>
                  <img src={`data:image/png;base64,${result.annotated_image}`} alt="annotated" className="w-full" />
                </div>
              )}

              {/* Preview with overlaid boxes if no annotated image */}
              {!result.annotated_image && preview && inputType === 'image' && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2">
                    <Activity size={14} className="text-sky-400" />
                    <span className="text-slate-300 text-sm font-medium">Detection Overlay (Demo)</span>
                  </div>
                  <div className="relative p-4">
                    <img src={preview} alt="input" className="w-full rounded-xl object-contain max-h-64" />
                    <div className="absolute inset-4 pointer-events-none">
                      {result.detections.map((d, i) => {
                        const [x1, y1, x2, y2] = d.bbox;
                        return (
                          <div
                            key={i}
                            className="absolute border-2 border-sky-400 rounded"
                            style={{
                              left: `${(x1 / 640) * 100}%`,
                              top: `${(y1 / 640) * 100}%`,
                              width: `${((x2 - x1) / 640) * 100}%`,
                              height: `${((y2 - y1) / 640) * 100}%`,
                            }}
                          >
                            <span className="absolute -top-5 left-0 text-xs bg-sky-500 text-white px-1 rounded whitespace-nowrap">
                              {d.class} {(d.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* Detection list */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                <div className="px-4 py-3 border-b border-slate-800">
                  <span className="text-slate-300 text-sm font-medium">Detected Objects</span>
                </div>
                <div className="p-4 space-y-2">
                  {result.detections.length === 0 ? (
                    <p className="text-slate-500 text-sm text-center py-4">No ships detected</p>
                  ) : result.detections.map((d, i) => (
                    <div key={i} className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-xl">
                      <div className="w-8 h-8 rounded-lg bg-sky-500/15 flex items-center justify-center text-sky-400 font-bold text-xs flex-shrink-0">
                        {i + 1}
                      </div>
                      <div className="flex-1">
                        <p className="text-slate-200 text-sm font-medium capitalize">{d.class}</p>
                        <p className="text-slate-500 text-xs">
                          Box: [{d.bbox.map(v => Math.round(v)).join(', ')}]
                        </p>
                      </div>
                      <div className={`text-sm font-bold ${confColor(d.confidence)}`}>
                        {(d.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Video frames */}
              {result.frames && result.frames.length > 0 && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-800">
                    <span className="text-slate-300 text-sm font-medium">Video Frames ({result.frames.length})</span>
                  </div>
                  <div className="p-4 grid grid-cols-2 gap-3">
                    {result.frames.slice(0, 6).map((f, i) => (
                      <div key={i} className="bg-slate-800 rounded-xl p-3">
                        {f.image && <img src={`data:image/png;base64,${f.image}`} alt={`frame ${f.frame_index}`} className="w-full rounded-lg mb-2 object-contain max-h-32" />}
                        <p className="text-slate-400 text-xs">Frame {f.frame_index} · {f.detections.length} ships</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {!result && !running && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col items-center justify-center py-20">
              <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center mb-4">
                <Crosshair size={28} className="text-slate-600" />
              </div>
              <p className="text-slate-400 font-medium">Results will appear here</p>
              <p className="text-slate-600 text-sm mt-1">Upload an image or video and run detection</p>
            </div>
          )}

          {running && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col items-center justify-center py-20">
              <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/30 flex items-center justify-center mb-4 animate-pulse">
                <ScanSearch size={28} className="text-sky-400" />
              </div>
              <p className="text-slate-300 font-medium">Running inference...</p>
              <p className="text-slate-500 text-sm mt-1">Processing with best_model.pth</p>
              <div className="mt-4 w-48 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-sky-500 to-teal-500 rounded-full animate-pulse w-2/3" />
              </div>
            </div>
          )}
        </div>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
