import { Network, Cpu, GitMerge, Crosshair, BarChart2, Layers, ArrowDown, ArrowRight } from 'lucide-react';

interface NodeProps {
  label: string;
  sub?: string;
  color: string;
  icon?: React.ReactNode;
}

function ArchNode({ label, sub, color, icon }: NodeProps) {
  const colors: Record<string, string> = {
    sky: 'bg-sky-500/15 border-sky-500/30 text-sky-300',
    teal: 'bg-teal-500/15 border-teal-500/30 text-teal-300',
    amber: 'bg-amber-500/15 border-amber-500/30 text-amber-300',
    emerald: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300',
    orange: 'bg-orange-500/15 border-orange-500/30 text-orange-300',
    blue: 'bg-blue-500/15 border-blue-500/30 text-blue-300',
    slate: 'bg-slate-700/50 border-slate-600/50 text-slate-300',
  };
  return (
    <div className={`border rounded-xl px-4 py-3 text-center ${colors[color] ?? colors.slate}`}>
      {icon && <div className="flex justify-center mb-1.5 opacity-80">{icon}</div>}
      <p className="font-semibold text-sm">{label}</p>
      {sub && <p className="text-xs opacity-60 mt-0.5">{sub}</p>}
    </div>
  );
}

function Arrow({ dir = 'down' }: { dir?: 'down' | 'right' }) {
  return dir === 'down'
    ? <div className="flex justify-center py-1"><ArrowDown size={14} className="text-slate-600" /></div>
    : <div className="flex items-center px-1"><ArrowRight size={14} className="text-slate-600" /></div>;
}

export default function ArchitecturePage() {
  return (
    <div className="p-8">
      <div className="mb-8">
        <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
          <Network size={14} className="text-sky-400" />
          <span>Architecture</span>
        </div>
        <h1 className="text-3xl font-bold text-white">System Architecture</h1>
        <p className="text-slate-400 mt-1">
          Multimodal SAR + Optical maritime ship detection system — deep learning pipeline overview.
        </p>
      </div>

      {/* Architecture summary table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden mb-8">
        <div className="px-6 py-4 border-b border-slate-800">
          <h2 className="text-slate-200 font-semibold flex items-center gap-2">
            <Cpu size={16} className="text-sky-400" />
            Architecture Summary
          </h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[
              { label: 'SAR Backbone', value: 'ResNet50 (pretrained ImageNet)', color: 'sky' },
              { label: 'Optical Backbone', value: 'CSPDarknet (pretrained)', color: 'teal' },
              { label: 'Fusion Module', value: 'CrossModalAttention', color: 'amber' },
              { label: 'Neck', value: 'FPN (Feature Pyramid Network)', color: 'blue' },
              { label: 'Detection Head', value: 'Multi-scale YOLO at P3/P4/P5', color: 'orange' },
              { label: 'Image Size', value: '640×640 (320×320 in training)', color: 'slate' },
              { label: 'Feature Channels', value: '256 (FEAT_CH)', color: 'slate' },
              { label: 'Anchors per Scale', value: '3 (NUM_ANCHORS)', color: 'slate' },
              { label: 'Classes', value: '1 (ship)', color: 'emerald' },
              { label: 'Optimizer', value: 'AdamW (wd=0.01)', color: 'slate' },
              { label: 'Scheduler', value: 'CosineAnnealingLR (50 epochs)', color: 'slate' },
              { label: 'Loss', value: 'L_cls + L_bbox (CIoU) + L_obj', color: 'slate' },
            ].map((s) => (
              <div key={s.label} className="bg-slate-800/50 border border-slate-700/30 rounded-xl p-4">
                <p className={`text-xs font-semibold mb-1 ${
                  s.color === 'sky' ? 'text-sky-400' :
                  s.color === 'teal' ? 'text-teal-400' :
                  s.color === 'amber' ? 'text-amber-400' :
                  s.color === 'blue' ? 'text-blue-400' :
                  s.color === 'orange' ? 'text-orange-400' :
                  s.color === 'emerald' ? 'text-emerald-400' :
                  'text-slate-500'
                }`}>{s.label}</p>
                <p className="text-slate-200 text-sm">{s.value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Visual architecture diagram */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 mb-8">
        <h2 className="text-slate-200 font-semibold mb-6 flex items-center gap-2">
          <Network size={16} className="text-sky-400" />
          Pipeline Architecture Diagram
        </h2>

        {/* Input row */}
        <div className="flex gap-4 mb-2 justify-center">
          <div className="flex-1 max-w-xs">
            <ArchNode label="SAR Input" sub="[B, 1, H, W]" color="sky" icon={<Layers size={16} />} />
          </div>
          <div className="flex-1 max-w-xs">
            <ArchNode label="Optical Input" sub="[B, 3, H, W]" color="teal" icon={<Layers size={16} />} />
          </div>
        </div>

        <div className="flex gap-4 mb-2 justify-center">
          <Arrow /><Arrow />
        </div>

        {/* Backbone row */}
        <div className="flex gap-4 mb-2 justify-center">
          <div className="flex-1 max-w-xs">
            <ArchNode label="ResNet50" sub="1-ch modified" color="sky" icon={<Cpu size={16} />} />
          </div>
          <div className="flex-1 max-w-xs">
            <ArchNode label="CSPDarknet" sub="3-ch optical" color="teal" icon={<Cpu size={16} />} />
          </div>
        </div>

        <div className="flex gap-4 mb-2 justify-center">
          <Arrow /><Arrow />
        </div>

        {/* Feature maps */}
        <div className="flex gap-4 mb-2 justify-center">
          <div className="flex-1 max-w-xs">
            <ArchNode label="SAR Features" sub="Fs_P3 / P4 / P5" color="blue" />
          </div>
          <div className="flex-1 max-w-xs">
            <ArchNode label="Optical Features" sub="Fo_P3 / P4 / P5" color="blue" />
          </div>
        </div>

        {/* Merge arrow */}
        <div className="flex justify-center my-3">
          <div className="flex items-center gap-2">
            <Arrow dir="right" />
            <ArchNode label="CrossModalAttention" sub="Channel-wise fusion at P3/P4/P5" color="amber" icon={<GitMerge size={14} />} />
            <Arrow dir="right" />
          </div>
        </div>

        <Arrow />

        {/* FPN */}
        <div className="max-w-xs mx-auto mb-2">
          <ArchNode label="FPN Neck" sub="Top-down multi-scale fusion, 256ch" color="orange" />
        </div>

        <Arrow />

        {/* Detection heads */}
        <div className="flex gap-3 justify-center mb-2">
          {[
            { label: 'P3 Head', sub: 'Stride 8 — Small', color: 'emerald' },
            { label: 'P4 Head', sub: 'Stride 16 — Med', color: 'emerald' },
            { label: 'P5 Head', sub: 'Stride 32 — Large', color: 'emerald' },
          ].map((h) => (
            <div key={h.label} className="flex-1 max-w-40">
              <ArchNode label={h.label} sub={h.sub} color={h.color} icon={<Crosshair size={14} />} />
            </div>
          ))}
        </div>

        <Arrow />

        {/* Output */}
        <div className="max-w-xs mx-auto">
          <ArchNode label="NMS + Decode" sub="bbox, objectness, class" color="slate" />
        </div>

        <Arrow />

        <div className="max-w-xs mx-auto">
          <ArchNode label="Ship Detections" sub="x, y, w, h, confidence" color="emerald" icon={<BarChart2 size={14} />} />
        </div>
      </div>

      {/* Hardware table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden mb-8">
        <div className="px-6 py-4 border-b border-slate-800">
          <h2 className="text-slate-200 font-semibold">Hardware Configuration</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800">
                {['Component', 'Specification'].map((h) => (
                  <th key={h} className="text-left px-6 py-3 text-slate-500 text-xs font-semibold uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {[
                ['Device', 'Lenovo IdeaPad Slim 5 14IAH8'],
                ['Processor', '12th Gen Intel Core i5-12450H (8 cores, 2.00 GHz)'],
                ['RAM', '16.0 GB (15.7 GB usable)'],
                ['GPU', 'Intel UHD Graphics (128 MB — integrated, no CUDA)'],
                ['Storage', '954 GB (715 GB used)'],
                ['OS', 'Windows 11 Home 64-bit'],
                ['Training Mode', 'CPU only (no GPU acceleration)'],
              ].map(([k, v]) => (
                <tr key={k} className="hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-3 text-slate-400 font-medium">{k}</td>
                  <td className="px-6 py-3 text-slate-300">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Training params comparison */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-800">
          <h2 className="text-slate-200 font-semibold">Training Parameters — Full vs Actual</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800">
                {['Parameter', 'Full Scale', 'Actual (CPU)', 'Reason'].map((h) => (
                  <th key={h} className="text-left px-6 py-3 text-slate-500 text-xs font-semibold uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {[
                ['IMAGE_SIZE', '640×640', '320×320', 'Reduces memory by ~75%'],
                ['BATCH_SIZE', '8', '4', 'Prevents RAM overflow'],
                ['Dataset', '192K pairs', '10K stratified', 'CPU training constraint'],
                ['NUM_EPOCHS', '50', '20', '~45-90 min/epoch on CPU'],
                ['NUM_WORKERS', '4+', '2', 'Windows multiprocessing limit'],
                ['Mixed Precision', 'Enabled', 'Disabled', 'No effect on CPU'],
                ['Backbone Freeze', 'After epoch 10', 'All epochs', 'Reduce compute'],
                ['Grad Accumulation', '—', '2 steps', 'Simulates batch size 8'],
              ].map(([p, full, actual, reason]) => (
                <tr key={p} className="hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-3 text-sky-400 font-mono text-xs">{p}</td>
                  <td className="px-6 py-3 text-slate-300">{full}</td>
                  <td className="px-6 py-3 text-amber-400">{actual}</td>
                  <td className="px-6 py-3 text-slate-500 text-xs">{reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
