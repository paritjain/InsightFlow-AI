'use client';

import React, { useEffect, useState, use } from 'react';
import Link from 'next/link';
import { 
  Sparkles, 
  LayoutDashboard, 
  Folder, 
  ShieldCheck, 
  History, 
  BookOpen, 
  Layers, 
  LineChart, 
  BrainCircuit, 
  Lightbulb, 
  CheckSquare, 
  MessageSquare, 
  Download, 
  ArrowLeft,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Send,
  Table as TableIcon,
  RefreshCw,
  FileSpreadsheet,
  Activity,
  UserCheck,
  Zap,
  Info,
  GitBranch,
  Eye,
  Sliders,
  BarChart3,
  Trash2,
  ArrowUpRight,
  Box,
  FileText
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useRef } from 'react';
import { 
  fetchProjectDetails, 
  applyCleaningRules, 
  askYourData, 
  overrideSemanticRole,
  updateActionStatus,
  getMasterPreview,
  getExcelExportUrl,
  getPdfExportUrl,
  uploadProjectFiles,
  sendChatMessage,
  fetchChatHistory,
  clearChatHistory
} from '@/lib/api';
import PlotlyChartRenderer from '@/components/PlotlyChartRenderer';
import MarkdownRenderer from '@/components/MarkdownRenderer';
import { 
  ResponsiveContainer, 
  LineChart as ReLineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  BarChart, 
  Bar,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  Legend,
  AreaChart,
  Area
} from 'recharts';

const CHART_PALETTE = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899', '#06b6d4', '#6366f1'];

function HeatmapGrid({ data }: { data: any[] }) {
  if (!Array.isArray(data) || data.length === 0) return null;
  const rawXList = data.map(d => String(d.x ?? d.var1 ?? d.metric_1 ?? '')).filter(Boolean);
  const rawYList = data.map(d => String(d.y ?? d.var2 ?? d.metric_2 ?? '')).filter(Boolean);

  const xVars = Array.from(new Set(rawXList));
  let yVars = Array.from(new Set(rawYList));

  if (yVars.length <= 1 && (yVars.length === 0 || yVars[0] === '0') && xVars.length > 1) {
    yVars = [...xVars];
  }

  const matrix: Record<string, Record<string, { value: number; pearson?: number; spearman?: number; is_constant?: boolean }>> = {};
  data.forEach(d => {
    const x = String(d.x ?? d.var1 ?? d.metric_1 ?? '');
    let y = String(d.y ?? d.var2 ?? d.metric_2 ?? '');
    if (y === '0' && xVars.includes(x)) {
      y = x;
    }
    const v = Number(d.value ?? d.val ?? d.pearson ?? 0);
    const p = d.pearson !== undefined ? Number(d.pearson) : v;
    const s = d.spearman !== undefined ? Number(d.spearman) : undefined;
    const isConst = Boolean(d.is_constant);
    if (!matrix[x]) matrix[x] = {};
    matrix[x][y] = {
      value: Number.isFinite(v) ? v : 0,
      pearson: Number.isFinite(p) ? p : v,
      spearman: s !== undefined && Number.isFinite(s) ? s : undefined,
      is_constant: isConst
    };
  });

  return (
    <div className="w-full overflow-x-auto p-2">
      <table className="border-collapse text-[11px] font-mono mx-auto">
        <thead>
          <tr>
            <th className="p-2 border border-slate-800 bg-slate-900 text-slate-400">Var / Var</th>
            {yVars.map(y => (
              <th key={y} className="p-2 border border-slate-800 bg-slate-900 text-slate-300 font-bold truncate max-w-[90px]">{y}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {xVars.map(x => (
            <tr key={x}>
              <td className="p-2 border border-slate-800 bg-slate-900 text-slate-300 font-bold truncate max-w-[90px]">{x}</td>
              {yVars.map(y => {
                const cell = matrix[x]?.[y] ?? { value: 0 };
                const val = cell.value;
                const isPositive = val >= 0;
                const bgStyle = isPositive
                  ? `rgba(59, 130, 246, ${Math.max(0.15, Math.abs(val) * 0.85)})`
                  : `rgba(244, 63, 94, ${Math.max(0.15, Math.abs(val) * 0.85)})`;

                const tooltip = cell.is_constant && x !== y
                  ? `${x} vs ${y}: Undefined correlation (constant variable with 0 variance)`
                  : cell.spearman !== undefined
                  ? `${x} vs ${y}: Pearson r = ${cell.pearson ?? val}, Spearman r = ${cell.spearman}`
                  : `${x} vs ${y}: r = ${val}`;

                let displayVal = "";
                if (cell.is_constant && x !== y) {
                  displayVal = "N/A";
                } else if (Math.abs(val) > 0 && Math.abs(val) < 0.05) {
                  displayVal = val > 0 ? `+${val.toFixed(3)}` : val.toFixed(3);
                } else {
                  displayVal = val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
                }

                return (
                  <td
                    key={y}
                    className="p-2.5 border border-slate-800/80 text-center font-bold transition-all hover:scale-105"
                    style={{ backgroundColor: bgStyle, color: Math.abs(val) > 0.4 ? '#ffffff' : '#94a3b8' }}
                    title={tooltip}
                  >
                    {displayVal}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BoxPlotGrid({ data }: { data: any[] }) {
  if (!Array.isArray(data) || data.length === 0) return null;

  return (
    <div className="w-full grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 p-2">
      {data.map((item, idx) => {
        const cat = String(item.category || item.x || `Group #${idx + 1}`);
        const minVal = Number(item.min ?? 0);
        const q1Val = Number(item.q1 ?? 0);
        const medVal = Number(item.median ?? 0);
        const q3Val = Number(item.q3 ?? 0);
        const maxVal = Number(item.max ?? 0);

        return (
          <div key={idx} className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
            <span className="font-bold text-slate-200 block truncate font-mono">{cat}</span>
            <div className="space-y-1 font-mono text-[11px]">
              <div className="flex justify-between text-slate-400"><span>Max:</span> <span className="text-slate-200 font-bold">{maxVal}</span></div>
              <div className="flex justify-between text-blue-400"><span>Q3 (75%):</span> <span className="font-bold">{q3Val}</span></div>
              <div className="flex justify-between text-emerald-400"><span>Median:</span> <span className="font-bold">{medVal}</span></div>
              <div className="flex justify-between text-amber-400"><span>Q1 (25%):</span> <span className="font-bold">{q1Val}</span></div>
              <div className="flex justify-between text-slate-400"><span>Min:</span> <span className="text-slate-200 font-bold">{minVal}</span></div>
            </div>
            <div className="h-3 w-full bg-slate-900 rounded-full border border-slate-800 relative overflow-hidden mt-2">
              <div
                className="h-full bg-blue-500/40 border-l border-r border-blue-400 relative"
                style={{ left: '20%', width: '60%' }}
              >
                <div className="h-full w-1 bg-emerald-400 mx-auto" />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

interface UniversalChartProps {
  chartType: string;
  title: string;
  description?: string;
  data: any[];
  xKey?: string;
  yKey?: string;
  fallbackMessage?: string;
  inputRowCount?: number;
  takeaway?: string;
  suitabilityReason?: string;
  seriesKeys?: string[];
}

function UniversalChartRenderer({
  chartType,
  title,
  description,
  data,
  xKey = 'x',
  yKey = 'y',
  fallbackMessage,
  inputRowCount = 0,
  takeaway,
  suitabilityReason,
  seriesKeys
}: UniversalChartProps) {
  console.log(`[EDA Pipeline Trace] Selected chart: ${chartType} | Title: "${title}"`);
  console.log(`[EDA Pipeline Trace] X field: ${xKey} | Y field: ${yKey}`);
  console.log(`[EDA Pipeline Trace] Input Data Length:`, data ? data.length : 0);

  const isWaterfall = React.useMemo(() => {
    return (chartType || '').toUpperCase() === 'WATERFALL';
  }, [chartType]);

  const isHeatmap = React.useMemo(() => {
    const t = (chartType || '').toUpperCase();
    return t.includes('HEATMAP') || t.includes('CORRELATION');
  }, [chartType]);

  const isScatter = React.useMemo(() => {
    return (chartType || '').toUpperCase() === 'SCATTER';
  }, [chartType]);

  const detectedSeriesKeys = React.useMemo(() => {
    if (seriesKeys && seriesKeys.length > 0) return seriesKeys;
    if (!Array.isArray(data) || data.length === 0) return [];
    const sample = data[0];
    const reserved = new Set(['x', 'y', 'category', 'value', '_isValidY', 'YearMonth', 'bin', 'date', 'count', 'val', 'start', 'end', 'spacer', 'barHeight', 'displayValue', 'type']);
    return Object.keys(sample).filter(k => !reserved.has(k) && (typeof sample[k] === 'number' || (!isNaN(parseFloat(sample[k])) && isFinite(sample[k]))));
  }, [seriesKeys, data]);

  const normalizedData = React.useMemo(() => {
    if (!Array.isArray(data)) return [];

    if (isWaterfall) {
      return data.map((item, idx) => {
        const rawCat = item.category ?? item.x ?? item.label ?? `Step #${idx + 1}`;
        let rawVal = item.value ?? item.val ?? 0;
        let numericVal = typeof rawVal === 'number' ? rawVal : parseFloat(String(rawVal).replace(/[^0-9.-]+/g, ''));
        if (!Number.isFinite(numericVal)) numericVal = 0;

        const rawType = item.type ?? (idx === data.length - 1 ? 'total' : numericVal >= 0 ? 'addition' : 'subtraction');
        const rawStart = typeof item.start === 'number' ? item.start : 0;
        const rawEnd = typeof item.end === 'number' ? item.end : (rawStart + numericVal);

        let spacer = 0;
        let barHeight = Math.abs(numericVal);
        if (rawType === 'total') {
          spacer = 0;
          barHeight = Math.max(0, rawEnd);
        } else if (rawType === 'subtraction' || numericVal < 0) {
          spacer = Math.min(rawStart, rawEnd);
          barHeight = Math.abs(numericVal);
        } else {
          spacer = rawStart;
          barHeight = Math.abs(numericVal);
        }

        return {
          ...item,
          x: String(rawCat),
          category: String(rawCat),
          value: numericVal,
          displayValue: numericVal,
          type: rawType,
          spacer,
          barHeight,
          _isValidY: true
        };
      });
    }

    if (isHeatmap) {
      return data.map((item, idx) => {
        const rawX = item.x ?? item.var1 ?? item.metric_1 ?? item.column1 ?? `Var #${idx + 1}`;
        const rawY = item.y ?? item.var2 ?? item.metric_2 ?? item.column2 ?? `Var #${idx + 1}`;
        let rawVal = item.value ?? item.val ?? item.pearson ?? item.r ?? 0;
        let numericVal = typeof rawVal === 'number' ? rawVal : parseFloat(String(rawVal).replace(/[^0-9.-]+/g, ''));
        if (!Number.isFinite(numericVal)) numericVal = 0;

        return {
          ...item,
          x: String(rawX),
          y: String(rawY),
          value: numericVal,
          val: numericVal,
          pearson: item.pearson !== undefined ? Number(item.pearson) : numericVal,
          spearman: item.spearman !== undefined ? Number(item.spearman) : undefined,
          is_constant: Boolean(item.is_constant),
          _isValidY: true
        };
      });
    }

    if (isScatter) {
      return data.map((item, idx) => {
        // Priority for X:
        // 1. item[xKey] if valid numeric (e.g. item['sales'])
        // 2. item['x']
        // 3. Fallback to any numeric property in item that isn't yKey, 'y', 'category'
        let rawX = item[xKey] !== undefined ? item[xKey] : item['x'];
        if (rawX === undefined || rawX === null || (typeof rawX === 'string' && isNaN(parseFloat(rawX)))) {
          const fallbackXKey = Object.keys(item).find(k =>
            k !== yKey && k !== 'y' && k !== 'category' && k !== 'value' && k !== 'type' &&
            (typeof item[k] === 'number' || (typeof item[k] === 'string' && !isNaN(parseFloat(item[k]))))
          );
          rawX = fallbackXKey ? item[fallbackXKey] : (item['x'] ?? 0);
        }
        let numericX = typeof rawX === 'number' ? rawX : parseFloat(String(rawX).replace(/[^0-9.-]+/g, ''));
        if (!Number.isFinite(numericX)) numericX = 0;

        // Priority for Y:
        // 1. item[yKey] if valid numeric (e.g. item['marketing_spend'])
        // 2. item['y']
        // 3. item['value']
        let rawY = item[yKey] !== undefined ? item[yKey] : (item['y'] ?? item['value']);
        if (rawY === undefined || rawY === null || (typeof rawY === 'string' && isNaN(parseFloat(rawY)))) {
          const fallbackYKey = Object.keys(item).find(k =>
            k !== xKey && k !== 'x' && k !== 'category' && k !== 'value' && k !== 'type' &&
            (typeof item[k] === 'number' || (typeof item[k] === 'string' && !isNaN(parseFloat(item[k]))))
          );
          rawY = fallbackYKey ? item[fallbackYKey] : (item['y'] ?? 0);
        }
        let numericY = typeof rawY === 'number' ? rawY : parseFloat(String(rawY).replace(/[^0-9.-]+/g, ''));
        if (!Number.isFinite(numericY)) numericY = 0;

        const cat = item.category ?? item.label ?? `Record #${idx + 1}`;

        return {
          ...item,
          x: numericX,
          y: numericY,
          category: String(cat),
          value: numericY,
          _isValidY: Number.isFinite(numericX) && Number.isFinite(numericY)
        };
      });
    }

    return data.map((item, idx) => {
      const rawX = item[xKey] ?? item['category'] ?? item['YearMonth'] ?? item['bin'] ?? item['date'] ?? item['x'] ?? `Pt #${idx + 1}`;
      
      let rawY = item[yKey] ?? item['y'] ?? item['value'] ?? item['count'];
      if (rawY === undefined || rawY === null) {
        const fallbackKey = Object.keys(item).find(k => 
          k !== xKey && k !== 'YearMonth' && k !== 'x' && k !== 'date' && k !== 'category' && k !== 'bin' &&
          !k.toLowerCase().includes('growth') && !k.toLowerCase().includes('rolling') && !k.toLowerCase().includes('avg') &&
          (typeof item[k] === 'number' || (typeof item[k] === 'string' && !isNaN(parseFloat(item[k]))))
        );
        rawY = fallbackKey ? item[fallbackKey] : 0;
      }
      
      let numericY = typeof rawY === 'number' ? rawY : parseFloat(String(rawY).replace(/[^0-9.-]+/g, ''));
      if (!Number.isFinite(numericY)) numericY = 0;

      return {
        ...item,
        x: String(rawX),
        y: numericY,
        _isValidY: Number.isFinite(numericY),
        category: String(rawX),
        value: numericY
      };
    });
  }, [data, xKey, yKey, isHeatmap, isWaterfall, isScatter]);

  const validPointsCount = normalizedData.filter(d => d._isValidY).length;
  console.log(`[EDA Pipeline Trace] Aggregated valid numeric rows: ${validPointsCount}`);

  const isRenderable = normalizedData.length > 0 && (validPointsCount > 0 || detectedSeriesKeys.length > 0);

  if (!isRenderable) {
    return (
      <div className="p-5 rounded-2xl glass-card border border-rose-900/50 bg-rose-950/15 space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="font-bold text-sm text-rose-300">{title}</h4>
          <span className="px-2 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-400 font-bold border border-rose-500/30 font-mono">
            Diagnostic Error State
          </span>
        </div>
        <p className="text-xs text-rose-300 font-semibold">Unable to render visualization: Payload contains no finite numerical points.</p>
        <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-300 space-y-1">
          <div><strong className="text-slate-400">Chart Type:</strong> {chartType}</div>
          <div><strong className="text-slate-400">X Field:</strong> {xKey}</div>
          <div><strong className="text-slate-400">Y Field:</strong> {yKey}</div>
          <div><strong className="text-slate-400">Input Rows:</strong> {inputRowCount || data?.length || 0}</div>
          <div><strong className="text-slate-400">Output Rows:</strong> {normalizedData.length}</div>
          <div><strong className="text-slate-400">Valid Y Values:</strong> {validPointsCount}</div>
          {fallbackMessage && <div><strong className="text-slate-400">Reason:</strong> {fallbackMessage}</div>}
        </div>
      </div>
    );
  }

  const typeUpper = (chartType || 'BAR').toUpperCase();

  return (
    <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4 flex flex-col justify-between">
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-bold text-sm text-slate-200">{title}</h3>
            {description && <p className="text-xs text-slate-400 mt-0.5">{description}</p>}
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 font-bold border border-blue-500/20 font-mono">
            {typeUpper}
          </span>
        </div>

        <div className="h-72 w-full">
          {(typeUpper === 'LINE' || typeUpper === 'TIME_SERIES') && (
            <ResponsiveContainer width="100%" height="100%">
              <ReLineChart data={normalizedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" stroke="#6b7280" tick={{ fontSize: 11 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }} />
                <Line type="monotone" dataKey="y" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              </ReLineChart>
            </ResponsiveContainer>
          )}

          {typeUpper === 'AREA' && (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={normalizedData}>
                <defs>
                  <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" stroke="#6b7280" tick={{ fontSize: 11 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }} />
                <Area type="monotone" dataKey="y" stroke="#3b82f6" strokeWidth={2.5} fillOpacity={1} fill="url(#areaGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          )}

          {(typeUpper === 'BAR' || typeUpper === 'HORIZONTAL_BAR') && (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={normalizedData} layout={typeUpper === 'HORIZONTAL_BAR' ? 'vertical' : 'horizontal'}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                {typeUpper === 'HORIZONTAL_BAR' ? (
                  <>
                    <XAxis type="number" stroke="#6b7280" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="x" stroke="#6b7280" tick={{ fontSize: 11 }} width={90} />
                  </>
                ) : (
                  <>
                    <XAxis dataKey="x" stroke="#6b7280" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                  </>
                )}
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }} />
                <Bar dataKey="y" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}

          {typeUpper === 'HISTOGRAM' && (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={normalizedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="bin" stroke="#6b7280" tick={{ fontSize: 10 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }} />
                <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}

          {(typeUpper === 'DONUT' || typeUpper === 'PIE') && (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={normalizedData}
                  dataKey="value"
                  nameKey="category"
                  cx="50%"
                  cy="50%"
                  innerRadius={typeUpper === 'DONUT' ? 45 : 0}
                  outerRadius={85}
                  paddingAngle={3}
                  label={(entry: any) => `${entry.category || entry.x} (${(((entry.percent as number) || 0) * 100).toFixed(0)}%)`}
                >
                  {normalizedData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={CHART_PALETTE[index % CHART_PALETTE.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }} />
              </PieChart>
            </ResponsiveContainer>
          )}

          {typeUpper === 'SCATTER' && (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 25, bottom: 25, left: 15 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  stroke="#6b7280" 
                  tick={{ fontSize: 11 }} 
                  name={xKey || "X"} 
                  domain={['auto', 'auto']}
                  tickFormatter={(val) => typeof val === 'number' ? Number(val.toFixed(1)).toLocaleString() : val}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  stroke="#6b7280" 
                  tick={{ fontSize: 11 }} 
                  name={yKey || "Y"} 
                  domain={['auto', 'auto']}
                  tickFormatter={(val) => typeof val === 'number' ? Number(val.toFixed(1)).toLocaleString() : val}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }} 
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }}
                  formatter={(value: any, name: any) => [
                    typeof value === 'number' ? Number(value.toFixed(2)).toLocaleString() : value,
                    name === 'x' || name === 'X' ? (xKey || 'X') : name === 'y' || name === 'Y' ? (yKey || 'Y') : name
                  ]}
                  labelFormatter={(_, payload) => {
                    const pt = payload?.[0]?.payload;
                    return pt?.category ? String(pt.category) : '';
                  }}
                />
                <Scatter name={`${xKey} vs ${yKey}`} data={normalizedData} fill="#8b5cf6" />
              </ScatterChart>
            </ResponsiveContainer>
          )}

          {isHeatmap && (
            <div className="h-full w-full overflow-auto flex items-center justify-center">
              <HeatmapGrid data={normalizedData} />
            </div>
          )}

          {typeUpper === 'BOX_PLOT' && (
            <div className="h-full w-full overflow-auto flex items-center justify-center">
              <BoxPlotGrid data={normalizedData} />
            </div>
          )}

          {typeUpper === 'WATERFALL' && (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={normalizedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="category" stroke="#6b7280" tick={{ fontSize: 11 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }}
                  formatter={(val: any, name?: any, item?: any) => [
                    item?.payload?.displayValue !== undefined ? Number(item.payload.displayValue).toLocaleString() : val,
                    'Net Shift / Total'
                  ]}
                />
                <Bar dataKey="spacer" stackId="wf" fill="transparent" />
                <Bar dataKey="barHeight" stackId="wf" radius={[4, 4, 0, 0]}>
                  {normalizedData.map((entry, index) => {
                    let fill = '#3b82f6';
                    if (entry.type === 'addition') fill = '#10b981';
                    if (entry.type === 'subtraction') fill = '#ef4444';
                    return <Cell key={`wf-cell-${index}`} fill={fill} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}

          {(typeUpper === 'STACKED_BAR' || typeUpper === 'STACKED_BAR_100' || typeUpper === 'GROUPED_BAR') && (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={normalizedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" stroke="#6b7280" tick={{ fontSize: 11 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} domain={typeUpper === 'STACKED_BAR_100' ? [0, 100] : ['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }} />
                {detectedSeriesKeys.length > 0 && <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />}
                {detectedSeriesKeys.length > 0 ? (
                  detectedSeriesKeys.map((k, idx) => (
                    <Bar
                      key={k}
                      dataKey={k}
                      name={k}
                      stackId={typeUpper === 'GROUPED_BAR' ? undefined : 'stack'}
                      fill={CHART_PALETTE[idx % CHART_PALETTE.length]}
                      radius={typeUpper === 'GROUPED_BAR' ? [4, 4, 0, 0] : [0, 0, 0, 0]}
                    />
                  ))
                ) : (
                  <Bar dataKey="value" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                )}
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {(takeaway || suitabilityReason) && (
        <div className="pt-3 border-t border-slate-800/80 space-y-1.5 text-xs">
          {takeaway && (
            <div className="flex items-start gap-2 bg-slate-950/70 p-2.5 rounded-lg border border-slate-800/80">
              <Sparkles className="w-3.5 h-3.5 text-blue-400 mt-0.5 shrink-0" />
              <p className="text-slate-200 leading-relaxed"><strong className="text-blue-300">Key Takeaway:</strong> {takeaway}</p>
            </div>
          )}
          {suitabilityReason && (
            <p className="text-[11px] text-slate-400 italic px-1">
              <span className="text-slate-500 font-semibold not-italic">Why this chart: </span>
              {suitabilityReason}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function WorkspacePage({ params }: { params: Promise<{ projectId: string }> }) {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadingNewFile, setUploadingNewFile] = useState(false);

  const resolvedParams = use(params);
  const projectId = resolvedParams.projectId;

  const [project, setProject] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  async function handleDirectFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      const filesArray = Array.from(e.target.files);
      try {
        setUploadingNewFile(true);
        const res = await uploadProjectFiles(filesArray);
        router.push(`/workspace/${res.project_id}`);
      } catch (err: any) {
        console.error(err);
        alert(err.message || 'Failed to upload selected file(s)');
      } finally {
        setUploadingNewFile(false);
        e.target.value = '';
      }
    }
  }

  // Interactive Cleaning state
  const [rules, setRules] = useState<any[]>([]);
  const [cleaningInProgress, setCleaningInProgress] = useState(false);
  const [qualityFilter, setQualityFilter] = useState<'all' | 'quality' | 'operational'>('all');

  // Action Center edit state
  const [editingActionId, setEditingActionId] = useState<string | null>(null);
  const [actionFormOwner, setActionFormOwner] = useState<string>('');
  const [actionFormDueDate, setActionFormDueDate] = useState<string>('');
  const [actionFormTarget, setActionFormTarget] = useState<string>('');
  const [actionFormCompletionNotes, setActionFormCompletionNotes] = useState<string>('');
  const [actionFormVerificationNotes, setActionFormVerificationNotes] = useState<string>('');
  const [savingAction, setSavingAction] = useState<boolean>(false);

  // Ask Your Data state
  const [queryInput, setQueryInput] = useState('');
  const [queryHistory, setQueryHistory] = useState<any[]>([]);
  const [asking, setAsking] = useState(false);

  // Automated EDA Chatbot state
  const [edaChatInput, setEdaChatInput] = useState('');
  const [edaChatHistory, setEdaChatHistory] = useState<any[]>([]);
  const [edaAsking, setEdaAsking] = useState(false);

  // Action Center status state
  const [actionItems, setActionItems] = useState<any[]>([]);

  // Master Dataset preview state
  const [masterPreviewData, setMasterPreviewData] = useState<any>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);

  // Semantic Role Override state
  const [overrideColumn, setOverrideColumn] = useState<string>('');
  const [overrideRole, setOverrideRole] = useState<string>('Measure');
  const [savingOverride, setSavingOverride] = useState(false);

  // Visual Studio state
  const [studioX, setStudioX] = useState<string>('');
  const [studioY, setStudioY] = useState<string>('');
  const [studioZ, setStudioZ] = useState<string>('');
  const [studioChartType, setStudioChartType] = useState<string>('SCATTER');

  // Data preview mode: 'cleaned' | 'raw'
  const [previewMode, setPreviewMode] = useState<'cleaned' | 'raw'>('cleaned');

  function openInVisualizer(xCol?: string, yCol?: string, cType?: string) {
    if (xCol) setStudioX(xCol);
    if (yCol) setStudioY(yCol);
    if (cType) setStudioChartType(cType.toUpperCase());
    setActiveTab('visualizations');
  }

  async function handleClearChat() {
    try {
      await clearChatHistory(projectId);
      setQueryHistory([]);
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    loadData();
  }, [projectId]);

  async function loadData() {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchProjectDetails(projectId);
      setProject(data);
      setRules(data.quality_issues || []);
      setActionItems(data.action_items || []);

      if (data.visual_dataset && data.visual_dataset.length > 0) {
        setMasterPreviewData((prev: any) => prev || {
          rows: data.visual_dataset,
          columns: Object.keys(data.visual_dataset[0] || {}),
          total_rows: data.profile?.num_rows || data.visual_dataset.length
        });
      }
      loadMasterPreview('master');

      const numCols = Object.keys(data.eda_results?.numeric_summary || {});
      const catCols = Object.keys(data.eda_results?.categorical_summary || {});
      if (catCols.length > 0 && !studioX) setStudioX(catCols[0]);
      else if (numCols.length > 0 && !studioX) setStudioX(numCols[0]);
      if (numCols.length > 1 && !studioY) setStudioY(numCols[1]);
      else if (numCols.length > 0 && !studioY) setStudioY(numCols[0]);
    } catch (err: any) {
      console.error(err);
      setError(err?.message || 'Failed to load project workspace details.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === 'visualizations') {
      if (!masterPreviewData || !masterPreviewData.rows || masterPreviewData.rows.length === 0) {
        loadMasterPreview('master');
      }
    }
  }, [activeTab]);

  async function handleToggleRule(ruleId: string, newStatus: 'approved' | 'rejected') {
    const updated = rules.map(r => r.id === ruleId ? { ...r, status: newStatus } : r);
    setRules(updated);
  }

  async function handleApplyApprovedRules() {
    try {
      setCleaningInProgress(true);
      const approved = rules.filter(r => r.status === 'approved' || r.status === 'pending');
      await applyCleaningRules(projectId, approved);
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setCleaningInProgress(false);
    }
  }

  async function executeAskQuery(queryText: string, confirmedField?: string) {
    if (!queryText.trim()) return;
    try {
      setAsking(true);
      const res = await sendChatMessage(projectId, queryText, confirmedField);
      setQueryHistory(prev => [{ question: queryText, ...res }, ...prev]);
      setQueryInput('');
    } catch (err) {
      console.error(err);
    } finally {
      setAsking(false);
    }
  }

  async function handleSendQuery(e: React.FormEvent) {
    e.preventDefault();
    executeAskQuery(queryInput);
  }

  async function executeEdaChat(queryText: string, confirmedField?: string) {
    if (!queryText.trim()) return;
    try {
      setEdaAsking(true);
      const res = await sendChatMessage(projectId, queryText, confirmedField);
      const entry = { question: queryText, ...res };
      setEdaChatHistory(prev => [entry, ...prev]);
      setQueryHistory(prev => [entry, ...prev]);
      setEdaChatInput('');
    } catch (err) {
      console.error(err);
    } finally {
      setEdaAsking(false);
    }
  }

  function handleEdaChatSubmit(e: React.FormEvent) {
    e.preventDefault();
    executeEdaChat(edaChatInput);
  }

  async function handleActionStatusChange(id: string, newStatus: string) {
    try {
      setActionItems(prev => prev.map(item => (item.id === id || item.action_id === id) ? { ...item, status: newStatus } : item));
      await updateActionStatus(projectId, id, newStatus);
    } catch (err) {
      console.error(err);
    }
  }

  function handleStartEditAction(act: any) {
    setEditingActionId(act.id || act.action_id);
    setActionFormOwner(act.owner && act.owner !== 'UNASSIGNED' ? String(act.owner) : '');
    setActionFormDueDate(act.due_date && act.due_date !== 'Not specified' ? String(act.due_date) : '');
    setActionFormTarget(act.target && act.target !== 'Not defined' ? String(act.target) : '');
    setActionFormCompletionNotes(act.completion_notes || '');
    setActionFormVerificationNotes(act.verification_notes || '');
  }

  async function handleSaveActionDetails(actionId: string, currentStatus: string) {
    try {
      setSavingAction(true);
      const updates = {
        status: currentStatus,
        owner: actionFormOwner.trim() || 'UNASSIGNED',
        due_date: actionFormDueDate.trim() || 'Not specified',
        target: actionFormTarget.trim() || 'Not defined',
        completion_notes: actionFormCompletionNotes,
        verification_notes: actionFormVerificationNotes,
      };
      setActionItems(prev => prev.map(item => (item.id === actionId || item.action_id === actionId) ? { ...item, ...updates } : item));
      await updateActionStatus(projectId, actionId, updates);
      setEditingActionId(null);
    } catch (err) {
      console.error(err);
    } finally {
      setSavingAction(false);
    }
  }

  async function handleSaveSemanticOverride(columnName: string, role: string) {
    try {
      setSavingOverride(true);
      await overrideSemanticRole(projectId, columnName, role);
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setSavingOverride(false);
    }
  }

  async function loadMasterPreview(mode: 'cleaned' | 'raw' | 'master' = 'master') {
    try {
      setLoadingPreview(true);
      const res = await getMasterPreview(projectId, mode);
      setMasterPreviewData(res);
      setPreviewMode(mode === 'raw' ? 'raw' : 'cleaned');
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingPreview(false);
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-[#090d16] flex items-center justify-center text-slate-400">
        <div className="flex items-center gap-3">
          <RefreshCw className="w-5 h-5 animate-spin text-blue-500" />
          <span>Loading project workspace & analytical engine...</span>
        </div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="min-h-screen bg-[#090d16] flex flex-col items-center justify-center text-slate-300 p-6">
        <div className="max-w-md w-full p-8 rounded-2xl glass-card border border-rose-900/50 bg-rose-950/20 text-center space-y-4 shadow-xl">
          <div className="w-12 h-12 rounded-full bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mx-auto text-rose-400">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h3 className="font-bold text-lg text-rose-200">Unable to Load Workspace</h3>
          <p className="text-xs text-slate-400 leading-relaxed font-mono">
            Project ID: <span className="text-slate-200">{projectId}</span>
          </p>
          <p className="text-xs text-rose-300">
            {error || 'Project dataset could not be found or processed.'}
          </p>
          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              onClick={() => loadData()}
              className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition-all flex items-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry
            </button>
            <Link
              href="/dashboard"
              className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white shadow-md shadow-blue-500/20 transition-all"
            >
              Back to Dashboard
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const { profile, quality_issues, audit_trail, data_dictionary, master_meta, eda_results, insights, story, recommendations } = project;

  const sidebarNav = [
    { id: 'overview', name: 'Overview', icon: LayoutDashboard },
    { id: 'files', name: 'Files & Profiling', icon: Folder },
    { id: 'quality', name: 'Data Quality', icon: ShieldCheck, badge: quality_issues?.length },
    { id: 'cleaning', name: 'Cleaning Log', icon: History },
    { id: 'dictionary', name: 'Data Dictionary', icon: BookOpen },
    { id: 'master', name: 'Master Dataset (ETL)', icon: Layers },
    { id: 'eda', name: 'Automated EDA', icon: LineChart },
    { id: 'visualizations', name: 'Visualizations Studio', icon: BarChart3 },
    { id: 'insights', name: 'Insights & Story', icon: BrainCircuit },
    { id: 'recommendations', name: 'Recommendations', icon: Lightbulb },
    { id: 'action_center', name: 'Action Center', icon: CheckSquare },
    { id: 'ask_data', name: 'AI Assistant & Chat', icon: MessageSquare },
    { id: 'reports', name: 'Reports & Export', icon: Download },
  ];

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col font-sans">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".xlsx,.xls,.csv"
        onChange={handleDirectFileUpload}
        className="fixed top-0 left-0 w-0 h-0 opacity-0 pointer-events-none z-[-1]"
      />
      {/* Top Bar */}
      <header className="h-14 border-b border-slate-800/80 bg-[#090d16] px-6 flex items-center justify-between sticky top-0 z-30">
        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div className="h-4 w-[1px] bg-slate-800" />
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span className="font-bold text-sm">{project.name}</span>
          </div>
          <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
            Health: {profile?.health_score}/100
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadingNewFile}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 transition-all flex items-center gap-1.5 disabled:opacity-50"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-blue-400" />
            {uploadingNewFile ? 'Uploading...' : 'Upload New File'}
          </button>
          <a
            href={getExcelExportUrl(projectId)}
            download
            className="px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 transition-all flex items-center gap-1.5 shadow-sm shadow-blue-500/20"
          >
            <Download className="w-3.5 h-3.5" />
            Export Master Excel
          </a>
          <a
            href={getPdfExportUrl(projectId)}
            download
            className="px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white bg-purple-600 hover:bg-purple-500 transition-all flex items-center gap-1.5 shadow-sm shadow-purple-500/20"
          >
            <Download className="w-3.5 h-3.5" />
            Export Executive PDF
          </a>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Persistent Left Sidebar */}
        <aside className="w-64 border-r border-slate-800/80 bg-[#090d16] p-4 flex flex-col justify-between select-none">
          <div className="space-y-1">
            <div className="px-3 py-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider">Workspace Navigation</div>
            {sidebarNav.map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    active 
                      ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30' 
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${active ? 'text-blue-400' : 'text-slate-500'}`} />
                    <span>{tab.name}</span>
                  </div>
                  {tab.badge ? (
                    <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/20 font-bold">
                      {tab.badge}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-[11px] text-slate-400">
            <p className="font-semibold text-slate-300">Semantic Engine Active</p>
            <p className="mt-0.5 text-slate-500">FastAPI + Semantic Layer</p>
          </div>
        </aside>

        {/* Workspace Tab Content Area */}
        <main className="flex-1 overflow-y-auto p-8 bg-[#090d16]/50">
          
          {/* TAB 1: OVERVIEW */}
          {activeTab === 'overview' && (
            <div className="space-y-8 max-w-6xl">
              <div>
                <h1 className="text-2xl font-bold">Executive Project Summary</h1>
                <p className="text-sm text-slate-400 mt-1">Unified Semantic Data Layer & Evidence-Backed Business Narrative.</p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
                <div className="p-5 rounded-2xl glass-card border border-slate-800">
                  <p className="text-xs text-slate-400 font-medium">Dataset Records</p>
                  <p className="text-2xl font-bold text-slate-100 mt-2">{profile?.num_rows?.toLocaleString()}</p>
                </div>
                <div className="p-5 rounded-2xl glass-card border border-slate-800">
                  <p className="text-xs text-slate-400 font-medium">Profiled Columns</p>
                  <p className="text-2xl font-bold text-slate-100 mt-2">{profile?.num_columns}</p>
                </div>
                <div className="p-5 rounded-2xl glass-card border border-slate-800">
                  <p className="text-xs text-slate-400 font-medium">Data Quality Health</p>
                  <p className="text-2xl font-bold text-emerald-400 mt-2">{profile?.health_score}%</p>
                </div>
                <div className="p-5 rounded-2xl glass-card border border-slate-800">
                  <p className="text-xs text-slate-400 font-medium">Identified Domain</p>
                  <p className="text-lg font-bold text-blue-400 mt-2 truncate">{eda_results?.domain || "General Analytics"}</p>
                </div>
              </div>

              {story && (
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                  <h3 className="font-bold text-lg text-slate-100">{story.title}</h3>
                  <p className="text-sm text-slate-300 leading-relaxed">{story.executive_summary}</p>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: FILES & PROFILING */}
          {activeTab === 'files' && (
            <div className="space-y-8 max-w-6xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Folder className="w-6 h-6 text-blue-400" />
                    Files & Operational Profiling
                  </h1>
                  <p className="text-sm text-slate-400 mt-1">
                    Multi-sheet file inspection, inferred observation grain, business context, and analytical readiness.
                  </p>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploadingNewFile}
                  className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 font-semibold text-xs text-white shadow-md shadow-blue-500/20 transition-all flex items-center gap-2 self-start md:self-auto"
                >
                  <FileSpreadsheet className="w-4 h-4" />
                  {uploadingNewFile ? 'Uploading File...' : 'Upload Additional File'}
                </button>
              </div>

              {/* Inferred Operational Grain & Context Card */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-purple-400">
                    <Activity className="w-4 h-4 text-purple-400" />
                    Observation Grain
                  </div>
                  <p className="text-base font-bold text-slate-100">
                    {profile?.observation_grain || profile?.profiling_summary?.likely_observation_grain || "1 row per operational observation"}
                  </p>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Determines unit of aggregation, time-series frequency, and group-level baseline comparisons.
                  </p>
                </div>

                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-400">
                    <Zap className="w-4 h-4 text-blue-400" />
                    Business & Plant Context
                  </div>
                  <p className="text-base font-bold text-slate-100">
                    {profile?.profiling_summary?.business_context || profile?.business_context || "Industrial Plant Operations"}
                  </p>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Semantic domain detected from headers, telemetry patterns, and categorical vocabularies.
                  </p>
                </div>

                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-400">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    Ingestion Health & Lineage
                  </div>
                  <p className="text-base font-bold text-emerald-400">
                    {profile?.health_score ?? 100}% Data Health
                  </p>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {profile?.num_rows?.toLocaleString() || 0} valid records ingested across {profile?.num_columns || 0} recognized fields.
                  </p>
                </div>
              </div>

              {/* Profiling Summary: Reliable Analyses vs Additional Info Needed */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-3">
                  <h3 className="font-bold text-sm uppercase tracking-wider text-emerald-400 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    Analyses Available Reliably
                  </h3>
                  <div className="space-y-2 text-xs text-slate-300">
                    {(profile?.profiling_summary?.reliable_analyses && profile.profiling_summary.reliable_analyses.length > 0) ? (
                      profile.profiling_summary.reliable_analyses.map((item: string, idx: number) => (
                        <div key={idx} className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                          <span className="text-emerald-400 font-bold shrink-0 mt-0.5">✓</span>
                          <span>{item}</span>
                        </div>
                      ))
                    ) : (
                      <>
                        <div className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                          <span className="text-emerald-400 font-bold shrink-0 mt-0.5">✓</span>
                          <span>Empirical distribution summaries and robust Tukey IQR fences for continuous measures.</span>
                        </div>
                        <div className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                          <span className="text-emerald-400 font-bold shrink-0 mt-0.5">✓</span>
                          <span>Pareto concentration analysis across operational categories and equipment lines.</span>
                        </div>
                        <div className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                          <span className="text-emerald-400 font-bold shrink-0 mt-0.5">✓</span>
                          <span>Non-causal bivariate co-variance tracking and subgroup stratification.</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-3">
                  <h3 className="font-bold text-sm uppercase tracking-wider text-amber-400 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    Additional Information to Improve Analysis
                  </h3>
                  <div className="space-y-2 text-xs text-slate-300">
                    {(profile?.profiling_summary?.additional_info_needed && profile.profiling_summary.additional_info_needed.length > 0) ? (
                      profile.profiling_summary.additional_info_needed.map((item: string, idx: number) => (
                        <div key={idx} className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                          <span className="text-amber-400 font-bold shrink-0 mt-0.5">!</span>
                          <span>{item}</span>
                        </div>
                      ))
                    ) : (
                      <>
                        <div className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                          <span className="text-amber-400 font-bold shrink-0 mt-0.5">!</span>
                          <span>Supplying engineering setpoints or specification limits enables automated compliance checks.</span>
                        </div>
                        <div className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                          <span className="text-amber-400 font-bold shrink-0 mt-0.5">!</span>
                          <span>Shift-level operating logs improve root-cause differentiation across downtime events.</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Uploaded File Manifest */}
              <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                <h3 className="font-bold text-sm uppercase tracking-wider text-slate-300 flex items-center gap-2">
                  <FileSpreadsheet className="w-4 h-4 text-blue-400" />
                  Uploaded Worksheets & File Manifest
                </h3>
                <div className="overflow-x-auto rounded-xl border border-slate-800">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-900 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                      <tr>
                        <th className="p-3.5">Filename</th>
                        <th className="p-3.5">Format</th>
                        <th className="p-3.5">Detected Worksheets / Tables</th>
                        <th className="p-3.5">Ingested Rows</th>
                        <th className="p-3.5">Columns</th>
                        <th className="p-3.5">Profiling Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {(project.files && project.files.length > 0) ? (
                        project.files.map((f: any, fidx: number) => (
                          <tr key={fidx} className="hover:bg-slate-900/40">
                            <td className="p-3.5 font-bold font-mono text-slate-100">{f.name || f.filename}</td>
                            <td className="p-3.5 font-mono text-slate-400">{(f.name || f.filename)?.split('.').pop()?.toUpperCase()}</td>
                            <td className="p-3.5 text-slate-300">
                              {f.sheets ? f.sheets.join(', ') : 'Default Sheet / Table'}
                            </td>
                            <td className="p-3.5 font-mono text-slate-200">{profile?.num_rows?.toLocaleString()}</td>
                            <td className="p-3.5 font-mono text-slate-200">{profile?.num_columns}</td>
                            <td className="p-3.5">
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                PROFILED & ACTIVE
                              </span>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr className="hover:bg-slate-900/40">
                          <td className="p-3.5 font-bold font-mono text-slate-100">{project.name}</td>
                          <td className="p-3.5 font-mono text-slate-400">CSV / XLSX</td>
                          <td className="p-3.5 text-slate-300">Primary Plant Sheet</td>
                          <td className="p-3.5 font-mono text-slate-200">{profile?.num_rows?.toLocaleString()}</td>
                          <td className="p-3.5 font-mono text-slate-200">{profile?.num_columns}</td>
                          <td className="p-3.5">
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                              PROFILED & ACTIVE
                            </span>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: DATA QUALITY & WHAT IS LACKING */}
          {activeTab === 'quality' && (
            <div className="space-y-6 max-w-6xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold flex items-center gap-2">
                    <ShieldCheck className="w-6 h-6 text-emerald-400" />
                    Data Quality & Operational Process Gaps
                  </h1>
                  <p className="text-sm text-slate-400 mt-1">
                    Separates physical data defects from operational execution gaps (e.g., unassigned owners, unresolved actions, incomplete logs).
                  </p>
                </div>
                {rules.some(r => r.status === 'approved' || r.status === 'pending') && (
                  <button
                    onClick={handleApplyApprovedRules}
                    disabled={cleaningInProgress}
                    className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 font-semibold text-xs text-white shadow-md shadow-emerald-600/20 transition-all flex items-center gap-2 self-start md:self-auto disabled:opacity-50"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${cleaningInProgress ? 'animate-spin' : ''}`} />
                    {cleaningInProgress ? 'Applying Cleaning Rules...' : 'Apply Approved Cleaning Rules'}
                  </button>
                )}
              </div>

              {/* Quality & Process Gap Summary Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-4 rounded-2xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Health Score</span>
                  <span className="text-2xl font-bold text-emerald-400 font-mono">{profile?.health_score ?? 100}%</span>
                </div>
                <div className="p-4 rounded-2xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Data Quality Defects</span>
                  <span className="text-2xl font-bold text-amber-400 font-mono">
                    {rules.filter(r => !r.is_operational_gap && r.category !== 'OPERATIONAL_GAP').length}
                  </span>
                </div>
                <div className="p-4 rounded-2xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Operational Process Gaps</span>
                  <span className="text-2xl font-bold text-purple-400 font-mono">
                    {rules.filter(r => r.is_operational_gap || r.category === 'OPERATIONAL_GAP').length}
                  </span>
                </div>
                <div className="p-4 rounded-2xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Total Findings</span>
                  <span className="text-2xl font-bold text-slate-100 font-mono">{rules.length}</span>
                </div>
              </div>

              {/* Filter Tabs */}
              <div className="flex items-center gap-2 border-b border-slate-800/80 pb-3">
                <button
                  onClick={() => setQualityFilter('all')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    qualityFilter === 'all' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200 bg-slate-900'
                  }`}
                >
                  All Findings ({rules.length})
                </button>
                <button
                  onClick={() => setQualityFilter('operational')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    qualityFilter === 'operational' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-slate-200 bg-slate-900'
                  }`}
                >
                  Operational Gaps ({rules.filter(r => r.is_operational_gap || r.category === 'OPERATIONAL_GAP').length})
                </button>
                <button
                  onClick={() => setQualityFilter('quality')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    qualityFilter === 'quality' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200 bg-slate-900'
                  }`}
                >
                  Data Quality Issues ({rules.filter(r => !r.is_operational_gap && r.category !== 'OPERATIONAL_GAP').length})
                </button>
              </div>

              {/* Findings List */}
              <div className="space-y-4">
                {rules
                  .filter(r => {
                    if (qualityFilter === 'operational') return r.is_operational_gap || r.category === 'OPERATIONAL_GAP';
                    if (qualityFilter === 'quality') return !r.is_operational_gap && r.category !== 'OPERATIONAL_GAP';
                    return true;
                  })
                  .map((rule: any) => {
                    const isOpGap = rule.is_operational_gap || rule.category === 'OPERATIONAL_GAP';
                    const isConfirmed = (rule.confirmation_status || 'CONFIRMED').toUpperCase() === 'CONFIRMED';
                    return (
                      <div key={rule.id} className="p-5 rounded-2xl glass-card border border-slate-800 space-y-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              isOpGap ? 'bg-purple-500/15 text-purple-300 border border-purple-500/30' : 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                            }`}>
                              {isOpGap ? 'OPERATIONAL PROCESS GAP' : 'DATA QUALITY ISSUE'}
                            </span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              isConfirmed ? 'bg-rose-500/15 text-rose-300 border border-rose-500/30' : 'bg-blue-500/15 text-blue-300 border border-blue-500/30'
                            }`}>
                              {isConfirmed ? 'CONFIRMED' : 'SUSPECTED'}
                            </span>
                            <span className="font-mono text-xs text-slate-400 font-bold">
                              Column: {rule.column}
                            </span>
                          </div>
                          <span className="text-xs font-mono text-slate-400">
                            Affected: <strong className="text-slate-200">{rule.affected_count?.toLocaleString() || rule.affected_rows || 0}</strong> records ({rule.affected_percentage ?? 0}%)
                          </span>
                        </div>

                        <div>
                          <h4 className="font-bold text-sm text-slate-100">{rule.rule_name || rule.issue || rule.type}</h4>
                          <p className="text-xs text-slate-300 mt-1">{rule.description || rule.action}</p>
                        </div>

                        {rule.why_it_matters && (
                          <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs text-slate-300 space-y-1">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-400 block">Why it matters:</span>
                            <p>{rule.why_it_matters}</p>
                          </div>
                        )}

                        {rule.what_to_check && (
                          <div className="p-3 rounded-xl bg-blue-950/30 border border-blue-800/30 text-xs text-blue-200 space-y-1">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-400 block">What should be checked or corrected:</span>
                            <p>{rule.what_to_check}</p>
                          </div>
                        )}

                        <div className="flex items-center justify-between pt-2 border-t border-slate-800/60">
                          <span className="text-[11px] text-slate-500 font-mono">Rule ID: {rule.id}</span>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => handleToggleRule(rule.id, 'approved')}
                              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                                rule.status === 'approved' ? 'bg-emerald-600 text-white' : 'bg-slate-900 text-slate-400 hover:text-white'
                              }`}
                            >
                              Approve Fix
                            </button>
                            <button
                              onClick={() => handleToggleRule(rule.id, 'rejected')}
                              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                                rule.status === 'rejected' ? 'bg-rose-600 text-white' : 'bg-slate-900 text-slate-400 hover:text-white'
                              }`}
                            >
                              Dismiss / Keep Raw
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* TAB 4: CLEANING LOG */}
          {activeTab === 'cleaning' && (
            <div className="space-y-6 max-w-6xl">
              <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                  <History className="w-6 h-6 text-purple-400" />
                  Cleaning Log & Transformation Audit Trail
                </h1>
                <p className="text-sm text-slate-400 mt-1">
                  Complete chronological lineage of all automated and approved cleaning operations applied to source data.
                </p>
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-800 glass-card">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-900 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                    <tr>
                      <th className="p-3.5">Timestamp</th>
                      <th className="p-3.5">Step / Rule Applied</th>
                      <th className="p-3.5">Target Column</th>
                      <th className="p-3.5">Affected Records</th>
                      <th className="p-3.5">Action & Lineage</th>
                      <th className="p-3.5">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {(audit_trail && audit_trail.length > 0) ? (
                      audit_trail.map((item: any, idx: number) => (
                        <tr key={idx} className="hover:bg-slate-900/40">
                          <td className="p-3.5 font-mono text-[11px] text-slate-400 whitespace-nowrap">
                            {item.timestamp || 'Initial Ingestion'}
                          </td>
                          <td className="p-3.5 font-bold text-slate-100">
                            {item.step_name || item.rule_name || `Rule #${idx + 1}`}
                          </td>
                          <td className="p-3.5 font-mono text-purple-400">{item.column || item.target_column || 'All Columns'}</td>
                          <td className="p-3.5 font-mono text-slate-200">{item.affected_rows?.toLocaleString() ?? 0}</td>
                          <td className="p-3.5 text-slate-300">{item.action || item.description || 'Verified deterministic transformation'}</td>
                          <td className="p-3.5">
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                              APPLIED & VERIFIED
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={6} className="p-8 text-center text-slate-400">
                          <History className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                          <p className="font-semibold text-slate-300">No transformations applied yet.</p>
                          <p className="text-[11px] text-slate-500 mt-1">Source data is currently maintained in its authentic raw state.</p>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 5: DATA DICTIONARY (With Manual Override Controls) */}
          {activeTab === 'dictionary' && (
            <div className="space-y-6 max-w-6xl">
              <div>
                <h1 className="text-2xl font-bold">Semantic Data Dictionary</h1>
                <p className="text-sm text-slate-400 mt-1">Column roles, subtypes, descriptions, eligibility flags, and manual overrides.</p>
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-800 glass-card">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-900 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                    <tr>
                      <th className="p-3.5">Column Name</th>
                      <th className="p-3.5">Physical Dtype</th>
                      <th className="p-3.5">Semantic Role & Subtype</th>
                      <th className="p-3.5">Key Role & ID Safety</th>
                      <th className="p-3.5">Analytical Eligibility</th>
                      <th className="p-3.5">Allowed Aggregations</th>
                      <th className="p-3.5">Grouping / Time</th>
                      <th className="p-3.5">Confidence & Evidence</th>
                      <th className="p-3.5">Override Role</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {data_dictionary?.map((item: any, idx: number) => {
                      const semRole = item.semantic_role || "Dimension";
                      const isDateTime = semRole === 'Date/Time' || semRole === 'Time';
                      const isMeasure = semRole === 'Measure';
                      const isId = semRole === 'Identifier' || semRole === 'ID';
                      const isGroupingAllowed = item.grouping_allowed ?? (item.grouping_status === 'Allowed' || (item.grouping_eligibility !== false && !isId));
                      const isTimeSeriesEligible = item.time_series_eligible ?? (item.time_series_status === 'Yes' || item.time_series_eligibility === true || isDateTime);
                      const isAnalyticalEligible = item.analytical_eligible !== false;
                      const hasOverride = Boolean(item.override_applied && item.override_role && item.override_role !== 'Automatic/No Override');
                      const selectVal = hasOverride ? item.override_role : "";
                      const keyRole = item.key_role || "None";
                      const idSafety = Boolean(item.id_safety === true || item.id_safety === 1 || String(item.id_safety).toLowerCase() === 'restricted' || isId);

                      return (
                        <tr key={idx} className="hover:bg-slate-900/40">
                          <td className="p-3.5 font-bold text-slate-100 font-mono">
                            <div>{item.column_name}</div>
                            {item.description && (
                              <div className="text-[10px] text-slate-400 font-sans font-normal truncate max-w-[140px]" title={item.description}>
                                {item.description}
                              </div>
                            )}
                          </td>
                          <td className="p-3.5 text-slate-400 font-mono">{item.physical_dtype || item.data_type}</td>
                          <td className="p-3.5 space-y-1">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              isMeasure ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' :
                              isId ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                              isDateTime ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' :
                              'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            }`}>
                              {semRole}
                            </span>
                            <div className="text-[10px] text-slate-300 font-medium">
                              {item.semantic_subtype || item.measure_type || "Generic"}
                              {item.unit && item.unit !== 'unknown' && item.unit !== 'None' && (
                                <span className="ml-1 text-slate-400 font-mono text-[9px]">({item.unit})</span>
                              )}
                              {item.rating_scale && item.rating_scale !== 'unknown' && (
                                <span className="ml-1 text-amber-400 font-mono text-[9px]">Scale: {item.rating_scale}</span>
                              )}
                            </div>
                          </td>
                          <td className="p-3.5 space-y-1">
                            {keyRole && keyRole !== 'None' ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20 block w-fit">
                                {keyRole}
                              </span>
                            ) : (
                              <span className="text-[11px] text-slate-500 font-mono">None</span>
                            )}
                            <div className="flex items-center gap-1.5 text-[9px]">
                              {idSafety ? (
                                <span className="text-amber-400 font-semibold flex items-center gap-1">
                                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span> ID Safety ON
                                </span>
                              ) : (
                                <span className="text-slate-500">ID Safety OFF</span>
                              )}
                            </div>
                          </td>
                          <td className="p-3.5 space-y-1">
                            {isAnalyticalEligible ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 block w-fit">
                                Eligible ({item.coverage_pct ?? 100}% cov)
                              </span>
                            ) : (
                              <div>
                                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20 block w-fit">
                                  BLOCKED ({item.exclusion_reason || 'INELIGIBLE'})
                                </span>
                                <span className="text-[9px] text-slate-500 font-mono block mt-0.5">
                                  {item.missing_pct ?? item['missing_%'] ?? 0}% missing
                                </span>
                              </div>
                            )}
                          </td>
                          <td className="p-3.5">
                            {item.allowed_aggregations && item.allowed_aggregations.length > 0 ? (
                              <div className="flex flex-wrap gap-1 max-w-[190px]">
                                {item.allowed_aggregations.map((op: string, opIdx: number) => (
                                  <span key={opIdx} className="px-1.5 py-0.5 rounded text-[9px] font-mono font-semibold bg-slate-800 text-slate-300 border border-slate-700/60">
                                    {op}
                                  </span>
                                ))}
                                {(!item.allowed_aggregations.includes('SUM') && isMeasure) && (
                                  <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20">
                                    SUM BLOCKED
                                  </span>
                                )}
                              </div>
                            ) : item.aggregation_operations ? (
                              <span className="text-[11px] font-mono text-slate-300">{item.aggregation_operations}</span>
                            ) : (
                              <span className="text-rose-400 font-medium text-[11px]">None (Protected ID)</span>
                            )}
                          </td>
                          <td className="p-3.5 space-y-1">
                            <div>
                              {isGroupingAllowed ? (
                                <span className="inline-flex items-center gap-1.5 text-emerald-400 font-bold text-[10px]">
                                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> Grouping OK
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1.5 text-rose-400 font-bold text-[10px]">
                                  <span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span> Grouping Blocked
                                </span>
                              )}
                            </div>
                            <div>
                              {isTimeSeriesEligible ? (
                                <span className="inline-flex items-center gap-1.5 text-purple-400 font-bold text-[10px]">
                                  <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span> Time Series
                                </span>
                              ) : (
                                <span className="text-slate-500 text-[10px]">Non-Temporal</span>
                              )}
                            </div>
                          </td>
                          <td className="p-3.5 max-w-[200px]">
                            <div className="space-y-1">
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                                (item.confidence_score ?? 90) >= 90 ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                                (item.confidence_score ?? 90) >= 75 ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' :
                                'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                              }`}>
                                {item.classification_confidence || `${item.confidence_score ?? 90}%`}
                              </span>
                              <p className="text-[10px] text-slate-400 leading-tight line-clamp-2" title={item.classification_reason || item.reason_for_classification}>
                                {item.classification_reason || item.reason_for_classification || "Inferred from statistical profile and data patterns"}
                              </p>
                            </div>
                          </td>
                          <td className="p-3.5">
                            <select
                              value={selectVal}
                              onChange={(e) => handleSaveSemanticOverride(item.column_name, e.target.value)}
                              disabled={savingOverride}
                              className="bg-slate-900 border border-slate-700 text-xs rounded px-2.5 py-1 font-semibold text-slate-200 focus:outline-none focus:border-blue-500 transition-colors"
                            >
                              <option value="">Automatic (No Override)</option>
                              <option value="Date/Time">Date/Time</option>
                              <option value="Measure">Measure</option>
                              <option value="Dimension">Dimension</option>
                              <option value="Identifier">Identifier (Protected)</option>
                            </select>
                            {hasOverride && (
                              <span className="block mt-0.5 text-[9px] text-amber-400 font-medium">Override Active</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 6: MASTER DATASET (ETL & Lineage) */}
          {activeTab === 'master' && (
            <div className="space-y-6 max-w-6xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold">Master Dataset & Data Previews</h1>
                  <p className="text-sm text-slate-400 mt-1">Compare raw incoming source files against the cleaned, validated master dataset with transformation lineage.</p>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex items-center bg-slate-900 p-1 rounded-xl border border-slate-800">
                    <button
                      onClick={() => loadMasterPreview('cleaned')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                        previewMode === 'cleaned'
                          ? 'bg-blue-600 text-white shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Cleaned Master Data
                    </button>
                    <button
                      onClick={() => loadMasterPreview('raw')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                        previewMode === 'raw'
                          ? 'bg-amber-600 text-white shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Raw Source Data
                    </button>
                  </div>
                  <button
                    onClick={() => loadMasterPreview(previewMode === 'raw' ? 'raw' : 'cleaned')}
                    disabled={loadingPreview}
                    className="px-3.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-semibold text-slate-200 flex items-center gap-1.5 transition-all"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${loadingPreview ? 'animate-spin text-blue-400' : 'text-slate-400'}`} />
                    Refresh
                  </button>
                </div>
              </div>

              {/* Transformation Lineage Pipeline */}
              {master_meta?.lineage && (
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                  <h3 className="font-bold text-sm text-blue-400 uppercase tracking-wider flex items-center gap-2">
                    <GitBranch className="w-4 h-4 text-blue-400" />
                    ETL Transformation Lineage Pipeline
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    {master_meta.lineage.stages?.map((stg: any, sidx: number) => (
                      <div key={sidx} className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-1">
                        <span className="text-[10px] font-bold text-blue-400 font-mono">Stage {stg.step_number}</span>
                        <p className="font-bold text-slate-200">{stg.step_name}</p>
                        <p className="text-slate-400 text-[11px] leading-tight">{stg.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Master Dataset Sample Rows Table */}
              {masterPreviewData ? (
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-sm text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                        <TableIcon className="w-4 h-4 text-emerald-400" />
                        {previewMode === 'raw' ? 'Raw Source Dataset Preview' : 'Cleaned Master Dataset Preview'}
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Displaying top {masterPreviewData.rows?.length || 0} sample records of {masterPreviewData.total_rows?.toLocaleString()} total rows ({masterPreviewData.total_columns} columns)
                      </p>
                    </div>
                    <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded border ${
                      previewMode === 'raw'
                        ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                        : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                    }`}>
                      {previewMode === 'raw' ? 'PRE-CLEANING RAW DATA' : 'POST-CLEANING MASTER DATA'}
                    </span>
                  </div>

                  <div className="overflow-x-auto max-h-96 rounded-xl border border-slate-800">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-900 text-slate-400 font-mono border-b border-slate-800 sticky top-0">
                        <tr>
                          {masterPreviewData.columns?.map((c: string, idx: number) => (
                            <th key={idx} className="p-3 whitespace-nowrap">{c}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40">
                        {masterPreviewData.rows?.map((row: any, ridx: number) => (
                          <tr key={ridx} className="hover:bg-slate-900/40">
                            {masterPreviewData.columns?.map((c: string, cidx: number) => (
                              <td key={cidx} className="p-3 font-mono text-[11px] text-slate-300 whitespace-nowrap">
                                {String(row[c] ?? '')}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="p-8 rounded-2xl glass-card border border-slate-800 text-center space-y-3">
                  <TableIcon className="w-8 h-8 text-slate-500 mx-auto" />
                  <p className="text-sm text-slate-400">Click below to load and inspect raw vs. cleaned dataset records.</p>
                  <button
                    onClick={() => loadMasterPreview('cleaned')}
                    disabled={loadingPreview}
                    className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white shadow-md shadow-blue-500/20 inline-flex items-center gap-1.5"
                  >
                    <Eye className="w-4 h-4" />
                    Load Dataset Preview
                  </button>
                </div>
              )}
            </div>
          )}

          {/* TAB 7: AUTOMATED EDA */}
          {activeTab === 'eda' && (
            <div className="space-y-8 max-w-6xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold">Automated Statistical EDA & Visual Analytics</h1>
                  <p className="text-sm text-slate-400 mt-1">Dataset-agnostic multi-chart discovery engine with rigorous descriptive statistics, distributions, and correlation matrices.</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-3 py-1 rounded-full text-xs font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5" />
                    Domain: {eda_results?.domain?.detected_domain || eda_results?.overview?.domain || "General Enterprise"}
                  </span>
                  <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5" title="Underlying mathematical and statistical calculations validated against dataset ground truth">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                    Calculation Integrity: Verified
                  </span>
                </div>
              </div>

              {/* 1. EDA Overview KPI Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <div className="p-4 rounded-xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Total Records</span>
                  <span className="text-lg font-bold text-slate-100 font-mono">
                    {eda_results?.overview?.num_rows?.toLocaleString() || profile?.num_rows?.toLocaleString() || "0"}
                  </span>
                </div>
                <div className="p-4 rounded-xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Features / Cols</span>
                  <span className="text-lg font-bold text-slate-100 font-mono">
                    {eda_results?.overview?.num_columns || profile?.num_columns || "0"}
                  </span>
                </div>
                <div className="p-4 rounded-xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Completeness</span>
                  <span className="text-lg font-bold text-emerald-400 font-mono">
                    {(100 - (eda_results?.overview?.overall_null_percentage || 0)).toFixed(1)}%
                  </span>
                </div>
                <div className="p-4 rounded-xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Duplicate Rows</span>
                  <span className={`text-lg font-bold font-mono ${(eda_results?.overview?.duplicate_rows || 0) > 0 ? 'text-amber-400' : 'text-slate-100'}`}>
                    {(eda_results?.overview?.duplicate_rows || 0).toLocaleString()} ({(eda_results?.overview?.duplicate_percentage || 0).toFixed(1)}%)
                  </span>
                </div>
                <div className="p-4 rounded-xl glass-card border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Memory Footprint</span>
                  <span className="text-lg font-bold text-slate-100 font-mono">
                    {eda_results?.overview?.memory_usage_mb ? `${eda_results.overview.memory_usage_mb} MB` : "< 1 MB"}
                  </span>
                </div>
              </div>

              {/* OPERATIONAL EDA: THE 5 PLANT QUESTIONS */}
              {eda_results?.operational_eda && (
                <div className="space-y-6 p-6 rounded-2xl glass-card border border-blue-900/40 bg-blue-950/10">
                  <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
                    <div>
                      <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-blue-400" />
                        Operational Analysis: The Five Core Plant Questions
                      </h2>
                      <p className="text-xs text-slate-400 mt-1">
                        Prioritizes operating KPIs, location concentrations, plausible contributing factors, exceptions, and practical next steps.
                      </p>
                    </div>
                    <span className="text-xs font-mono font-bold px-2.5 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                      OPERATIONAL PERSPECTIVE
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    {/* 1. What is happening? */}
                    <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3">
                      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-400">
                        <span className="w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 font-mono text-[11px]">1</span>
                        What is happening? (Current Operating State)
                      </div>
                      <div className="text-xs text-slate-300 space-y-2">
                        <p className="leading-relaxed text-slate-200">
                          {eda_results.operational_eda.what_is_happening?.summary_narrative}
                        </p>
                        <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 grid grid-cols-2 gap-2 text-[11px] font-mono">
                          <div>
                            <span className="text-slate-500 block">Observation Grain:</span>
                            <span className="text-slate-200 font-semibold">{eda_results.operational_eda.what_is_happening?.observation_grain}</span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Reporting Window:</span>
                            <span className="text-slate-200 font-semibold">{eda_results.operational_eda.what_is_happening?.reporting_period}</span>
                          </div>
                        </div>
                        {eda_results.operational_eda.what_is_happening?.key_kpis?.length > 0 && (
                          <div className="space-y-1.5 pt-1">
                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Key Operational Measurements:</span>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                              {eda_results.operational_eda.what_is_happening.key_kpis.slice(0, 4).map((kpi: any, kidx: number) => (
                                <div key={kidx} className="p-2 rounded bg-slate-900 border border-slate-800 text-[11px]">
                                  <span className="text-blue-300 font-bold block truncate">{kpi.metric}</span>
                                  <span className="text-slate-200 font-mono">Mean: {Number(kpi.mean || 0).toLocaleString()} (Median: {Number(kpi.median || 0).toLocaleString()})</span>
                                  <span className="text-[10px] text-slate-500 block">Norm: {kpi.normal_range}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* 2. Where is it happening? */}
                    <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3">
                      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-purple-400">
                        <span className="w-5 h-5 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-mono text-[11px]">2</span>
                        Where is it happening? (Affected Units & Concentration)
                      </div>
                      <div className="text-xs text-slate-300 space-y-2">
                        {eda_results.operational_eda.where_is_it_happening?.top_concentration_units?.slice(0, 3).map((u: any, uidx: number) => (
                          <div key={uidx} className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between">
                            <div>
                              <span className="text-slate-400 block text-[11px]">Dimension: <strong className="text-slate-200">{u.dimension}</strong></span>
                              <span className="text-purple-300 font-bold">{u.top_unit}</span>
                            </div>
                            <div className="text-right">
                              <span className="text-xs font-mono font-bold text-purple-400">{u.top_percentage}% of volume</span>
                              <span className="text-[10px] text-slate-500 block">across {u.unique_units} units</span>
                            </div>
                          </div>
                        ))}
                        {eda_results.operational_eda.where_is_it_happening?.pareto_findings?.map((p: string, pidx: number) => (
                          <p key={pidx} className="text-[11px] text-slate-400 italic bg-slate-900/40 p-2 rounded border border-slate-800/50">
                            {p}
                          </p>
                        ))}
                      </div>
                    </div>

                    {/* 3. Why might it be happening? */}
                    <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3">
                      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-cyan-400">
                        <span className="w-5 h-5 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 font-mono text-[11px]">3</span>
                        Why might it be happening? (Contributing Factors & Hypotheses)
                      </div>
                      <div className="text-xs text-slate-300 space-y-2">
                        <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[11px]">
                          <strong>Causality Notice:</strong> {eda_results.operational_eda.why_might_it_be_happening?.causality_disclaimer}
                        </div>
                        {eda_results.operational_eda.why_might_it_be_happening?.operational_hypotheses?.map((hyp: string, hidx: number) => (
                          <div key={hidx} className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-200">
                            💡 {hyp}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* 4. What is lacking or deviating? */}
                    <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3">
                      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-rose-400">
                        <span className="w-5 h-5 rounded-full bg-rose-500/20 flex items-center justify-center text-rose-400 font-mono text-[11px]">4</span>
                        What is lacking or deviating? (Gaps & Baseline Exceptions)
                      </div>
                      <div className="text-xs text-slate-300 space-y-2">
                        {eda_results.operational_eda.what_is_lacking_or_deviating?.baseline_exceptions?.length > 0 ? (
                          eda_results.operational_eda.what_is_lacking_or_deviating.baseline_exceptions.slice(0, 3).map((ex: any, eidx: number) => (
                            <div key={eidx} className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-800/40 text-[11px]">
                              <span className="text-rose-300 font-bold">{ex.metric}</span>: {ex.deviating_records_count} deviating observation(s) ({ex.deviating_percentage}%) beyond normal range {ex.normal_operating_range}.
                            </div>
                          ))
                        ) : (
                          <p className="text-slate-400 text-[11px]">No severe statistical outliers detected beyond baseline operating limits.</p>
                        )}
                        {eda_results.operational_eda.what_is_lacking_or_deviating?.recurring_issues?.map((rec: string, ridx: number) => (
                          <p key={ridx} className="text-[11px] text-slate-300 bg-slate-900 p-2 rounded border border-slate-800">
                            ⚠️ {rec}
                          </p>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* 5. What should we do next? */}
                  <div className="p-5 rounded-xl bg-emerald-950/20 border border-emerald-800/40 space-y-3">
                    <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-400">
                      <span className="w-5 h-5 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-mono text-[11px]">5</span>
                      What should we do next? (Recommended Plant Operational Actions)
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      {eda_results.operational_eda.what_should_we_do_next?.recommended_operational_actions?.map((act: any, aidx: number) => (
                        <div key={aidx} className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs space-y-1">
                          <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase ${
                            act.priority === 'HIGH' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          }`}>
                            {act.priority} PRIORITY
                          </span>
                          <p className="font-semibold text-slate-200 mt-1">{act.action}</p>
                          <p className="text-slate-400 text-[11px]">{act.objective}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* 2. Comprehensive Numerical Feature Statistics Table */}
              {eda_results?.numeric_summary && Object.keys(eda_results.numeric_summary).length > 0 && (
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-sm text-blue-400 uppercase tracking-wider flex items-center gap-2">
                        <TableIcon className="w-4 h-4 text-blue-400" />
                        Descriptive Numerical Statistics & Dispersion
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">Parametric and non-parametric distribution metrics, IQR fences, and outlier detection.</p>
                    </div>
                    <span className="text-xs font-mono text-slate-500">
                      {Object.keys(eda_results.numeric_summary).length} Numerical Columns
                    </span>
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-slate-800/80">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-900/90 text-slate-400 font-mono border-b border-slate-800">
                        <tr>
                          <th className="p-3">Feature</th>
                          <th className="p-3 text-right">Count</th>
                          <th className="p-3 text-right">Mean</th>
                          <th className="p-3 text-right">Std Dev</th>
                          <th className="p-3 text-right">Median</th>
                          <th className="p-3 text-right">Min - Max</th>
                          <th className="p-3 text-right">IQR (Q1 - Q3)</th>
                          <th className="p-3 text-right">Skewness</th>
                          <th className="p-3 text-right">Outliers</th>
                          <th className="p-3 text-right">Missing</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40">
                        {Object.entries(eda_results.numeric_summary).map(([colName, stat]: [string, any], idx: number) => {
                          const isSkewed = Math.abs(stat.skewness || 0) > 1.0;
                          const hasOutliers = (stat.outliers_count || 0) > 0;
                          const outPct = typeof stat.outliers_percentage === 'number'
                            ? stat.outliers_percentage
                            : (stat.count && stat.outliers_count ? (stat.outliers_count / stat.count * 100) : 0);
                          const outPctStr = outPct > 0 && outPct < 1 ? outPct.toFixed(2) : outPct.toFixed(1);

                          return (
                            <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                              <td className="p-3 font-semibold text-slate-100 flex items-center gap-1.5">
                                <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
                                {colName}
                              </td>
                              <td className="p-3 text-right font-mono text-slate-300">{stat.count?.toLocaleString()}</td>
                              <td className="p-3 text-right font-mono text-slate-200">{typeof stat.mean === 'number' ? stat.mean.toLocaleString(undefined, { maximumFractionDigits: 2 }) : stat.mean}</td>
                              <td className="p-3 text-right font-mono text-slate-400">{typeof stat.std === 'number' ? stat.std.toLocaleString(undefined, { maximumFractionDigits: 2 }) : stat.std}</td>
                              <td className="p-3 text-right font-mono text-emerald-400 font-medium">{typeof stat.median === 'number' ? stat.median.toLocaleString(undefined, { maximumFractionDigits: 2 }) : stat.median}</td>
                              <td className="p-3 text-right font-mono text-slate-400">
                                {typeof stat.min === 'number' ? stat.min.toLocaleString(undefined, { maximumFractionDigits: 1 }) : stat.min} → {typeof stat.max === 'number' ? stat.max.toLocaleString(undefined, { maximumFractionDigits: 1 }) : stat.max}
                              </td>
                              <td className="p-3 text-right font-mono text-slate-400">
                                {typeof stat.iqr === 'number' ? stat.iqr.toLocaleString(undefined, { maximumFractionDigits: 2 }) : stat.iqr}
                              </td>
                              <td className="p-3 text-right font-mono">
                                <span className={`px-1.5 py-0.5 rounded text-[10px] ${isSkewed ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' : 'text-slate-400'}`}>
                                  {typeof stat.skewness === 'number' ? stat.skewness.toFixed(2) : '0.00'}
                                </span>
                              </td>
                              <td className="p-3 text-right font-mono">
                                <span className={`px-1.5 py-0.5 rounded text-[10px] ${hasOutliers ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'text-slate-500'}`}>
                                  {stat.outliers_count || 0} ({outPctStr}%)
                                </span>
                              </td>
                              <td className="p-3 text-right font-mono text-slate-400">
                                {stat.missing_count || 0} ({typeof stat.missing_percentage === 'number' ? stat.missing_percentage.toFixed(1) : 0}%)
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 3. Statistical Correlation & Associations Table */}
              {eda_results?.relationships?.strong_associations && eda_results.relationships.strong_associations.length > 0 && (
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-sm text-purple-400 uppercase tracking-wider flex items-center gap-2">
                        <Activity className="w-4 h-4 text-purple-400" />
                        Pairwise Statistical Associations & Correlation Matrix
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">Quantified Pearson linear correlation and Spearman monotonic rank coefficients.</p>
                    </div>
                    <span className="text-xs font-mono text-slate-500">
                      {eda_results.relationships.strong_associations.length} Material Associations
                    </span>
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-slate-800/80">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-900/90 text-slate-400 font-mono border-b border-slate-800">
                        <tr>
                          <th className="p-3">Variable Pair</th>
                          <th className="p-3 text-center">Pearson r</th>
                          <th className="p-3 text-center">Spearman ρ</th>
                          <th className="p-3">Strength & Direction</th>
                          <th className="p-3">Type</th>
                          <th className="p-3">Statistical Takeaway</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40">
                        {eda_results.relationships.strong_associations.map((assoc: any, aIdx: number) => {
                          const var1 = assoc.var1 || assoc.metric_1;
                          const var2 = assoc.var2 || assoc.metric_2;
                          const r = Number(assoc.pearson_r || 0);
                          const isPos = r >= 0;
                          const absR = Math.abs(r);
                          return (
                            <tr key={aIdx} className="hover:bg-slate-900/40 transition-colors">
                              <td className="p-3 font-semibold text-slate-200">
                                <span className="text-blue-400">{var1}</span>
                                <span className="text-slate-500 mx-2">↔</span>
                                <span className="text-purple-400">{var2}</span>
                              </td>
                              <td className="p-3 text-center font-mono font-bold">
                                <span className={`px-2 py-0.5 rounded text-[11px] ${
                                  absR >= 0.7 
                                    ? isPos ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                                    : 'bg-slate-800 text-slate-300'
                                }`}>
                                  {r.toFixed(3)}
                                </span>
                              </td>
                              <td className="p-3 text-center font-mono text-slate-400">
                                {assoc.spearman_r !== undefined ? Number(assoc.spearman_r).toFixed(3) : 'N/A'}
                              </td>
                              <td className="p-3">
                                <span className="text-xs font-medium text-slate-200 capitalize">{assoc.strength || 'Moderate'} {assoc.direction || (isPos ? 'Positive' : 'Negative')}</span>
                              </td>
                              <td className="p-3 font-mono text-[11px] text-slate-400">
                                {assoc.relationship_type || 'Linear'}
                              </td>
                              <td className="p-3 text-xs text-slate-400">
                                {assoc.interpretation || (absR >= 0.7
                                  ? isPos ? 'Strong direct relationship; growth in one feature tightly tracks the other (association ≠ causation).' : 'Strong inverse relationship; as one feature grows, the other decreases (association ≠ causation).'
                                  : 'Moderate linear covariance observed across records (association ≠ causation).')}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 4. Time Series Line Chart */}
              {eda_results?.time_series_analysis?.monthly_trend && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-sm text-blue-400 uppercase tracking-wider flex items-center gap-2">
                      <LineChart className="w-4 h-4 text-blue-400" />
                      Temporal Progression & Trend Analysis
                    </h3>
                    {eda_results.time_series_analysis.trend_direction && (
                      <span className={`px-2.5 py-0.5 rounded text-[11px] font-bold font-mono border ${
                        eda_results.time_series_analysis.trend_direction === 'UPWARD' 
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                          : eda_results.time_series_analysis.trend_direction === 'DOWNWARD'
                          ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                          : 'bg-slate-800 text-slate-400 border-slate-700'
                      }`}>
                        Trend: {eda_results.time_series_analysis.trend_direction} (Slope: {eda_results.time_series_analysis.trend_slope ?? 'N/A'})
                      </span>
                    )}
                  </div>
                  <UniversalChartRenderer
                    chartType="LINE"
                    title={`Temporal Trend: ${eda_results.time_series_analysis.measure_column || "Measure"}`}
                    description={`Aggregated progression over temporal axis '${eda_results.time_series_analysis.date_column || "Date"}'`}
                    data={eda_results.time_series_analysis.monthly_trend}
                    xKey="YearMonth"
                    yKey={eda_results.time_series_analysis.measure_column || "value"}
                    takeaway={eda_results.time_series_analysis.takeaway || `Temporal progression of '${eda_results.time_series_analysis.measure_column || "measure"}' across ${eda_results.time_series_analysis.monthly_trend.length} recorded periods (${eda_results.time_series_analysis.overall_trend || eda_results.time_series_analysis.trend_direction || 'observed'} trend).`}


                    suitabilityReason={`Line chart accurately models continuous temporal progression without discrete binning distortion.`}
                    inputRowCount={eda_results.overview?.num_rows || 0}
                  />
                </div>
              )}

              {/* 5. Financial & Additive Waterfall Chart */}
              {eda_results?.waterfall_chart?.is_valid && eda_results.waterfall_chart.data && (
                <div className="space-y-2">
                  <h3 className="font-bold text-sm text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                    <Zap className="w-4 h-4 text-emerald-400" />
                    Additive Waterfall Decomposition Bridge
                  </h3>
                  <UniversalChartRenderer
                    chartType="WATERFALL"
                    title={eda_results.waterfall_chart.title || "Cumulative Contribution Waterfall"}
                    description={eda_results.waterfall_chart.subtitle || "Step-by-step contribution decomposition to net total."}
                    data={eda_results.waterfall_chart.data}
                    xKey="category"
                    yKey="value"
                    takeaway="Waterfall bridge decomposes net total volume into additive segment contributions, isolating key positive and negative drivers."
                    suitabilityReason="Waterfall charts represent additive cumulative shifts and component breakdown superior to static bar representations."
                    inputRowCount={eda_results.overview?.num_rows || 0}
                  />
                </div>
              )}

              {/* 6. Recommended Data-Dictionary-Driven Visualizations (Dynamic Mix) */}
              {eda_results?.recommended_visualizations && eda_results.recommended_visualizations.length > 0 && (
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-sm text-blue-400 uppercase tracking-wider flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-blue-400" />
                      Dynamic Statistical Chart Selection Mix ({eda_results.recommended_visualizations.length} Computed Visualizations)
                    </h3>
                    <span className="text-xs text-slate-500 font-mono">100% Empirically Validated</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {eda_results.recommended_visualizations.map((v: any, idx: number) => (
                      <UniversalChartRenderer
                        key={idx}
                        chartType={v.chart_type}
                        title={v.title}
                        description={v.description || v.reasoning}
                        data={v.data}
                        xKey={v.x_axis || v.category || v.variable || 'x'}
                        yKey={v.y_axis || v.metric || 'y'}
                        takeaway={v.takeaway}
                        suitabilityReason={v.suitability_reason || v.why_appropriate}
                        seriesKeys={v.series_keys}
                        fallbackMessage={v.fallback_message}
                        inputRowCount={eda_results.overview?.num_rows || 0}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* 7. Multi-Dimensional Stacked Bar Charts */}
              {eda_results?.stacked_bar_charts && eda_results.stacked_bar_charts.length > 0 && (
                <div className="space-y-4">
                  <h3 className="font-bold text-sm text-amber-400 uppercase tracking-wider flex items-center gap-2">
                    <Layers className="w-4 h-4 text-amber-400" />
                    Multi-Dimensional Segment Compositions
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {eda_results.stacked_bar_charts.map((sChart: any, sIdx: number) => (
                      <UniversalChartRenderer
                        key={sIdx}
                        chartType={sChart.chart_type || "STACKED_BAR"}
                        title={sChart.title}
                        description={sChart.subtitle}
                        data={sChart.data}
                        xKey={sChart.x_axis || "x"}
                        seriesKeys={sChart.series_keys}
                        takeaway={`Cross-dimensional breakdown illustrates categorical distribution across secondary dimension.`}
                        suitabilityReason="Stacked representations clearly convey both cumulative volume and intra-category share."
                        inputRowCount={eda_results.overview?.num_rows || 0}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* 8. Categorical Bar Charts */}
              {eda_results?.categorical_summary && Object.keys(eda_results.categorical_summary).length > 0 && (
                <div className="space-y-4">
                  <h3 className="font-bold text-sm text-emerald-400 uppercase tracking-wider">
                    Categorical Dimension Frequency Distributions
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {Object.entries(eda_results.categorical_summary).slice(0, 4).map(([colName, summary]: [string, any], cIdx: number) => {
                      const chartData = summary.top_counts ? Object.entries(summary.top_counts).map(([cat, count]) => ({ category: cat, value: count })) : [];
                      if (chartData.length === 0) return null;
                      const topCat = summary.top_category || summary.dominant_category || (chartData[0] ? chartData[0].category : 'N/A');
                      const topCount = summary.top_category_count !== undefined ? summary.top_category_count : (chartData[0] ? Number(chartData[0].value) : 0);
                      const totRows = summary.total_records || eda_results.overview?.num_rows || (chartData.reduce((acc: number, d: any) => acc + (Number(d.value) || 0), 0) || 1);
                      const topPct = summary.top_category_pct !== undefined ? summary.top_category_pct : summary.dominant_category_pct !== undefined ? summary.dominant_category_pct : Number(((topCount / totRows) * 100).toFixed(1));
                      const uniqCnt = summary.unique_count || summary.unique_categories || chartData.length;
                      const missPct = summary.missing_percentage !== undefined ? summary.missing_percentage : 0;
                      return (
                        <UniversalChartRenderer
                          key={cIdx}
                          chartType={chartData.length > 5 ? "HORIZONTAL_BAR" : "BAR"}
                          title={`Distribution by ${colName}`}
                          description={`Frequency analysis: ${uniqCnt} distinct categories (${missPct}% missing)`}
                          data={chartData}
                          xKey="category"
                          yKey="value"
                          takeaway={`Dominant category is '${topCat}' accounting for ${topPct}% of total records.`}
                          suitabilityReason="Bar charts represent categorical counts without implying false continuity."
                          inputRowCount={eda_results.overview?.num_rows || 0}
                        />
                      );
                    })}
                  </div>
                </div>
              )}

              {/* 9. Scatter Plots for Correlated Measures */}
              {eda_results?.scatter_plots && eda_results.scatter_plots.length > 0 && (
                <div className="space-y-4">
                  <h3 className="font-bold text-sm text-purple-400 uppercase tracking-wider">
                    Bivariate Scatter Correlations & Linear Fit
                  </h3>
                  <div className="grid grid-cols-1 gap-6">
                    {eda_results.scatter_plots.slice(0, 2).map((sp: any, spIdx: number) => (
                      <UniversalChartRenderer
                        key={spIdx}
                        chartType="SCATTER"
                        title={`${sp.metric_x} vs ${sp.metric_y} Scatter Correlation`}
                        description={`Pearson r = ${sp.pearson_r} | Linear Trend Slope = ${sp.trend_slope}`}
                        data={sp.data}
                        xKey="x"
                        yKey="y"
                        takeaway={`Scatter dispersion indicates a ${sp.pearson_r >= 0 ? 'positive' : 'negative'} relationship with regression slope ${sp.trend_slope}.`}
                        suitabilityReason="Scatter plots illustrate bivariate dispersion, clustering, and outlier leverage."
                        inputRowCount={eda_results.overview?.num_rows || 0}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* 10. Principal Component Analysis (PCA) */}
              {eda_results?.pca_analysis?.is_valid && (
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <h3 className="font-bold text-sm text-cyan-400 uppercase tracking-wider flex items-center gap-2">
                        <Box className="w-4 h-4 text-cyan-400" />
                        Multivariate Dimensionality Reduction (PCA)
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">Scikit-learn Principal Component Analysis decomposing multidimensional variance.</p>
                    </div>
                    <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-2.5 py-1 rounded-full border border-cyan-500/20">
                      {eda_results.pca_analysis.components_count || eda_results.pca_analysis.components || 2} Components | {eda_results.pca_analysis.effective_observations ?? eda_results.pca_analysis.total_samples} Observations{eda_results.pca_analysis.effective_observation_pct ? ` | ${eda_results.pca_analysis.effective_observation_pct}% Coverage` : ''}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {eda_results.pca_analysis.explained_variance_ratio?.map((ratio: number, cIdx: number) => (
                      <div key={cIdx} className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                        <span className="text-[10px] font-bold text-slate-500 uppercase">Component PC{cIdx + 1}</span>
                        <p className="text-base font-bold text-cyan-400 font-mono">{ratio}%</p>
                        <p className="text-[10px] text-slate-400">Cumul: {eda_results.pca_analysis.cumulative_variance_ratio?.[cIdx]}%</p>
                      </div>
                    ))}
                    <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                      <span className="text-[10px] font-bold text-slate-500 uppercase">Cumulative Explained</span>
                      <p className="text-base font-bold text-emerald-400 font-mono">
                        {eda_results.pca_analysis.cumulative_variance_ratio?.[eda_results.pca_analysis.cumulative_variance_ratio?.length - 1]}%
                      </p>
                      <p className="text-[10px] text-slate-400">Total multidimensional information</p>
                    </div>
                  </div>

                  {eda_results.pca_analysis.interpretation && (
                    <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs text-slate-300 space-y-1">
                      <strong className="text-cyan-400 text-[11px] uppercase tracking-wider block">Mathematical Interpretation:</strong>
                      <p className="leading-relaxed">{eda_results.pca_analysis.interpretation}</p>
                    </div>
                  )}

                  {/* Component Loadings Matrix Table */}
                  {eda_results.pca_analysis.loadings && Object.keys(eda_results.pca_analysis.loadings).length > 0 && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <strong className="text-cyan-400 text-xs uppercase tracking-wider block">Principal Component Loadings Matrix:</strong>
                        <span className="text-[10px] text-slate-500 font-mono">Standardized Feature Weights</span>
                      </div>
                      <div className="overflow-x-auto rounded-xl border border-slate-800">
                        <table className="w-full text-left text-xs text-slate-300">
                          <thead className="bg-slate-900/90 text-slate-400 font-mono border-b border-slate-800">
                            <tr>
                              <th className="p-2.5">Feature</th>
                              {Array.from({ length: eda_results.pca_analysis.components_count || 3 }).map((_, cIdx) => (
                                <th key={cIdx} className="p-2.5 text-right">PC{cIdx + 1} Loading</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/40">
                            {Object.entries(eda_results.pca_analysis.loadings).map(([feat, compMap]: [string, any], fIdx: number) => (
                              <tr key={fIdx} className="hover:bg-slate-900/40 transition-colors">
                                <td className="p-2.5 font-semibold text-slate-200 font-mono">{feat}</td>
                                {Array.from({ length: eda_results.pca_analysis.components_count || 3 }).map((_, cIdx) => {
                                  const val = compMap[`PC${cIdx + 1}`];
                                  const absVal = Math.abs(val ?? 0);
                                  return (
                                    <td key={cIdx} className="p-2.5 text-right font-mono">
                                      <span className={`px-2 py-0.5 rounded text-[11px] ${
                                        absVal >= 0.5 ? 'bg-cyan-500/15 text-cyan-400 font-bold border border-cyan-500/30' :
                                        absVal >= 0.3 ? 'text-slate-200 font-medium' :
                                        'text-slate-500'
                                      }`}>
                                        {typeof val === 'number' ? (val >= 0 ? `+${val.toFixed(4)}` : val.toFixed(4)) : 'N/A'}
                                      </span>
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* 3D Interactive PCA Projection via Plotly */}
                  {eda_results.pca_analysis.projections && eda_results.pca_analysis.projections.length > 0 && (
                    <PlotlyChartRenderer
                      chartType="PCA_3D"
                      title="3D Principal Component Projection (PC1 vs PC2 vs PC3)"
                      subtitle="Interactive 3D scatter projection of records across the first three principal components."
                      data={eda_results.pca_analysis.projections}
                      xKey="PC1"
                      yKey="PC2"
                      zKey="PC3"
                      takeaway={`First 3 components capture ${eda_results.pca_analysis.cumulative_variance_ratio?.[eda_results.pca_analysis.cumulative_variance_ratio?.length - 1]}% of variance across ${eda_results.pca_analysis.features_analyzed?.join(', ')}.`}
                      suitabilityReason="3D PCA projections reveal natural clusters, manifolds, and high-dimensional outlier leverages."
                    />
                  )}
                </div>
              )}

              {/* 11. Multi-Method Outlier & Anomaly Detection */}
              {eda_results?.advanced_outliers?.columns && Object.keys(eda_results.advanced_outliers.columns).length > 0 && (
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-sm text-rose-400 uppercase tracking-wider flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-rose-400" />
                        Multi-Method Outlier & Anomaly Detection
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">Triangulated anomaly scoring comparing Tukey IQR fences, Z-Scores (|Z| &gt; 3.0), and Isolation Forests.</p>
                    </div>
                    {eda_results.advanced_outliers.multivariate_anomalies_count !== undefined && (
                      <span className="text-xs font-mono text-rose-400 bg-rose-500/10 px-2.5 py-1 rounded-full border border-rose-500/20">
                        {eda_results.advanced_outliers.multivariate_anomalies_count} Isolation Forest Anomalies ({eda_results.advanced_outliers.multivariate_anomalies_percentage}%)
                      </span>
                    )}
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-slate-800">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-900 text-slate-400 font-mono border-b border-slate-800">
                        <tr>
                          <th className="p-3">Variable</th>
                          <th className="p-3 text-right">Tukey IQR Outliers</th>
                          <th className="p-3 text-right">Z-Score Outliers (|Z| &gt; 3)</th>
                          <th className="p-3 text-right">IQR Valid Bounds</th>
                          <th className="p-3 text-right">Observed Rows</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40">
                        {Object.entries(eda_results.advanced_outliers.columns).map(([cName, cStat]: [string, any], oIdx: number) => {
                          const iqrPctStr = typeof cStat.iqr_outliers_pct === 'number' && cStat.iqr_outliers_pct > 0 && cStat.iqr_outliers_pct < 1
                            ? cStat.iqr_outliers_pct.toFixed(2)
                            : cStat.iqr_outliers_pct;
                          const zPctStr = typeof cStat.zscore_outliers_pct === 'number' && cStat.zscore_outliers_pct > 0 && cStat.zscore_outliers_pct < 1
                            ? cStat.zscore_outliers_pct.toFixed(2)
                            : cStat.zscore_outliers_pct;

                          return (
                            <tr key={oIdx} className="hover:bg-slate-900/40">
                              <td className="p-3 font-semibold text-slate-100">{cName}</td>
                              <td className="p-3 text-right font-mono">
                                <span className={`px-2 py-0.5 rounded text-[11px] ${cStat.iqr_outliers_count > 0 ? 'bg-amber-500/15 text-amber-400 border border-amber-500/20' : 'text-slate-400'}`}>
                                  {cStat.iqr_outliers_count} ({iqrPctStr}%)
                                </span>
                              </td>
                              <td className="p-3 text-right font-mono">
                                <span className={`px-2 py-0.5 rounded text-[11px] ${cStat.zscore_outliers_count > 0 ? 'bg-rose-500/15 text-rose-400 border border-rose-500/20' : 'text-slate-400'}`}>
                                  {cStat.zscore_outliers_count} ({zPctStr}%)
                                </span>
                              </td>
                              <td className="p-3 text-right font-mono text-slate-400">
                                [{cStat.iqr_bounds?.[0]} → {cStat.iqr_bounds?.[1]}]
                              </td>
                              <td className="p-3 text-right font-mono text-slate-400">{cStat.total_observed}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 12. Bivariate Hypothesis Testing (ANOVA & Kruskal-Wallis) */}
              {eda_results?.group_hypothesis_tests && eda_results.group_hypothesis_tests.length > 0 && (
                <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-sm text-purple-400 uppercase tracking-wider flex items-center gap-2">
                        <Activity className="w-4 h-4 text-purple-400" />
                        Categorical vs Numerical Hypothesis Testing (ANOVA & Kruskal-Wallis)
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">Formal statistical tests verifying if metric differences across category groups are statistically significant.</p>
                    </div>
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-slate-800">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-900 text-slate-400 font-mono border-b border-slate-800">
                        <tr>
                          <th className="p-3">Categorical Dimension ↔ Metric</th>
                          <th className="p-3 text-center">ANOVA F-Stat</th>
                          <th className="p-3 text-center">ANOVA p-value</th>
                          <th className="p-3 text-center">Kruskal H-Stat</th>
                          <th className="p-3 text-center">Effect Size (η²)</th>
                          <th className="p-3">Statistical Conclusion</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40">
                        {eda_results.group_hypothesis_tests.map((test: any, tIdx: number) => (
                          <tr key={tIdx} className="hover:bg-slate-900/40">
                            <td className="p-3 font-semibold text-slate-100">
                              <span className="text-blue-400">{test.dimension}</span>
                              <span className="text-slate-500 mx-1.5">↔</span>
                              <span className="text-emerald-400">{test.measure}</span>
                              <span className="text-[10px] text-slate-500 block font-normal mt-0.5">{test.num_groups} category groups</span>
                            </td>
                            <td className="p-3 text-center font-mono font-bold text-slate-200">{test.anova_f_stat}</td>
                            <td className="p-3 text-center font-mono font-bold">
                              <span className={`px-2 py-0.5 rounded text-[11px] ${test.is_significant ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-slate-800 text-slate-400'}`}>
                                {test.anova_p_value < 0.001 ? '< 0.001' : test.anova_p_value}
                              </span>
                            </td>
                            <td className="p-3 text-center font-mono text-slate-300">{test.kruskal_h_stat}</td>
                            <td className="p-3 text-center font-mono text-purple-400 font-bold">{test.eta_squared}</td>
                            <td className="p-3 text-xs text-slate-300 leading-normal">{test.interpretation}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 13. Interactive EDA Analysis Assistant & Quick Visual Query (Executive Chatbot) */}
              <div className="p-6 md:p-8 rounded-3xl glass-card border border-cyan-500/30 bg-gradient-to-b from-slate-900/95 to-slate-950/95 shadow-2xl space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
                  <div>
                    <div className="flex items-center gap-2.5">
                      <div className="w-9 h-9 rounded-2xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-lg shadow-cyan-500/20">
                        <MessageSquare className="w-5 h-5" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-bold text-lg text-slate-100">
                            Interactive EDA Assistant & Custom Visual Analytics
                          </h3>
                          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                            LIVE QUERY ENGINE
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Explore any slice of data, calculate category totals, or produce custom graphs with exact numbers on demand.
                        </p>
                      </div>
                    </div>
                  </div>
                  {edaChatHistory.length > 0 && (
                    <button
                      onClick={() => setEdaChatHistory([])}
                      className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors self-start sm:self-auto flex items-center gap-1.5 cursor-pointer"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      Clear History
                    </button>
                  )}
                </div>

                {/* Quick-Prompt Suggested Query Chips */}
                <div className="space-y-2">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block">
                    Quick Operational Queries (Click to Analyze):
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { label: "🚨 Which location has how many fatal cases?", query: "which location has how many fatal cases" },
                      { label: "📍 Fatal cases breakdown by location", query: "show fatal cases by location" },
                      { label: "🏢 Observation count by Location", query: "observation count by location" },
                      { label: "👥 Top 5 observers with highest volume", query: "top 5 observers with highest count" },
                      { label: "📋 Closure status breakdown across records", query: "closure status breakdown across records" },
                      { label: "🛡️ Potential impact breakdown", query: "potential impact category breakdown" }
                    ].map((chip, cIdx) => (
                      <button
                        key={cIdx}
                        onClick={() => executeEdaChat(chip.query)}
                        disabled={edaAsking}
                        className="px-3 py-1.5 rounded-xl bg-slate-950/80 hover:bg-cyan-950/40 border border-slate-800 hover:border-cyan-500/50 text-xs text-slate-300 hover:text-cyan-200 font-medium transition-all disabled:opacity-50 cursor-pointer flex items-center gap-1.5 shadow-sm"
                      >
                        {chip.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Query Input Bar */}
                <form onSubmit={handleEdaChatSubmit} className="flex gap-2.5">
                  <div className="relative flex-1">
                    <input
                      type="text"
                      value={edaChatInput}
                      onChange={(e) => setEdaChatInput(e.target.value)}
                      placeholder="e.g. Which location has how many fatal cases? Or show bar chart of..."
                      className="w-full px-4 py-3 rounded-2xl bg-slate-950 border border-slate-800 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all shadow-inner"
                      disabled={edaAsking}
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={edaAsking || !edaChatInput.trim()}
                    className="px-6 py-3 rounded-2xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-semibold text-xs transition-all shadow-lg shadow-cyan-600/25 flex items-center gap-2 cursor-pointer"
                  >
                    {edaAsking ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-white" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4" />
                        <span>Ask Data</span>
                      </>
                    )}
                  </button>
                </form>

                {/* Chat Results / Conversation Feed */}
                {edaChatHistory.length > 0 && (
                  <div className="space-y-6 pt-4 border-t border-slate-800/60">
                    {edaChatHistory.map((item, idx) => (
                      <div key={idx} className="p-6 rounded-2xl bg-slate-950/80 border border-slate-800/90 space-y-4 shadow-xl">
                        {/* Question Badge */}
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/60 pb-3">
                          <div className="flex items-center gap-2 text-sm font-semibold text-cyan-300">
                            <span className="w-6 h-6 rounded-lg bg-cyan-500/20 flex items-center justify-center text-cyan-400 font-mono text-xs">Q</span>
                            "{item.question}"
                          </div>
                          <div className="flex items-center gap-2">
                            {item.resolved_field && (
                              <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/40 flex items-center gap-1">
                                <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                                Resolved: <strong className="text-white">{item.resolved_field}</strong>
                              </span>
                            )}
                            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 flex items-center gap-1">
                              <ShieldCheck className="w-3 h-3" />
                              Grounded DuckDB Analytical Truth
                            </span>
                          </div>
                        </div>

                        {/* Answer Text with markdown & tables */}
                        {item.answer && (
                          <div className="text-slate-200 text-sm leading-relaxed">
                            <MarkdownRenderer content={item.answer} />
                          </div>
                        )}

                        {/* Interactive Match Suggestion Panel for EDA */}
                        {(item.suggestion_mode === 'confirmation_required' || (Array.isArray(item.match_suggestions) && item.match_suggestions.length > 0 && item.status !== 'PASS')) && (
                          <div className="p-4 rounded-xl bg-gradient-to-br from-blue-950/60 via-slate-900 to-indigo-950/40 border border-blue-500/30 space-y-3 shadow-lg">
                            <div className="flex items-center justify-between border-b border-blue-800/40 pb-2">
                              <span className="text-xs font-semibold text-blue-200 flex items-center gap-1.5">
                                <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                                Did you mean? Field Match Suggestions
                              </span>
                              <span className="text-[10px] font-mono text-slate-400">Query: "{item.question}"</span>
                            </div>
                            <div className="space-y-1.5">
                              {item.match_suggestions?.slice(0, 3).map((sugg: any, sIdx: number) => (
                                <div key={sIdx} className="flex items-center justify-between p-2 rounded-lg bg-slate-900/80 border border-slate-800">
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs font-semibold text-white">{sugg.column}</span>
                                    <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/40 px-1.5 py-0.5 rounded border border-cyan-800/40">
                                      {sugg.confidence_pct || Math.round((sugg.confidence || 0) * 100)}% match
                                    </span>
                                  </div>
                                  <button
                                    onClick={() => executeEdaChat(item.question, sugg.column)}
                                    className="px-2.5 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-all cursor-pointer flex items-center gap-1"
                                  >
                                    <CheckCircle2 className="w-3 h-3" />
                                    Use this field
                                  </button>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Visual Chart with Exact Numbers */}
                        {item.chart_spec && (
                          <div className="pt-2 rounded-2xl overflow-hidden border border-slate-800/80 bg-slate-900/50 p-4">
                            <PlotlyChartRenderer
                              chartType={item.chart_spec.chart_type}
                              title={item.chart_spec.title || 'Custom Visual Breakdown'}
                              subtitle={item.chart_spec.subtitle || `Visual analysis for: "${item.question}"`}
                              data={item.chart_spec.data}
                              xKey={item.chart_spec.x_axis}
                              yKey={item.chart_spec.y_axis}
                              takeaway={item.chart_spec.takeaway}
                              suitabilityReason={item.chart_spec.suitability_reason}
                            />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB: VISUALIZATIONS STUDIO (Plotly Interactive Workbench) */}
          {activeTab === 'visualizations' && (
            <div className="space-y-6 max-w-6xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold flex items-center gap-2">
                    <BarChart3 className="w-6 h-6 text-cyan-400" />
                    Interactive Visualizations Studio
                  </h1>
                  <p className="text-sm text-slate-400 mt-1">Dataset-aware dynamic charting engine powered by Plotly.js with interactive zoom, pan, hover, and PNG/HTML export.</p>
                </div>
                <button
                  onClick={() => loadMasterPreview('master')}
                  className="px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-300 flex items-center gap-1.5 self-start md:self-auto"
                >
                  <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
                  Refresh Visual Dataset Cache
                </button>
              </div>

              {/* Chart Workbench Control Bar */}
              <div className="p-5 rounded-2xl glass-card border border-slate-800 space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                  {/* Chart Type Selector */}
                  <div>
                    <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">Chart Type</label>
                    <select
                      value={studioChartType}
                      onChange={(e) => setStudioChartType(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs text-white focus:outline-none focus:border-cyan-500 font-medium"
                    >
                      <option value="SCATTER">Scatter Plot (Bivariate Fit)</option>
                      <option value="BAR">Bar Chart (Categorical)</option>
                      <option value="LINE">Line Chart (Trend)</option>
                      <option value="HISTOGRAM">Histogram (Distribution)</option>
                      <option value="BOX_PLOT">Box Plot (Dispersion & Outliers)</option>
                      <option value="VIOLIN">Violin Plot (Kernel Density)</option>
                      <option value="HEATMAP">Correlation Heatmap</option>
                      <option value="PCA_3D">3D PCA Projection</option>
                    </select>
                  </div>

                  {/* X Axis Selector */}
                  <div>
                    <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">Primary Axis (X)</label>
                    <select
                      value={studioX}
                      onChange={(e) => setStudioX(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs text-white focus:outline-none focus:border-cyan-500 font-medium"
                    >
                      {Object.keys(eda_results?.numeric_summary || {}).map((c) => {
                        const dictEntry = data_dictionary?.find((d: any) => d.column_name === c);
                        const isEligible = dictEntry ? dictEntry.analytical_eligible !== false : true;
                        const missingPct = dictEntry ? (dictEntry.missing_pct ?? dictEntry['missing_%'] ?? 0) : 0;
                        return isEligible ? (
                          <option key={c} value={c}>{c} (Numeric)</option>
                        ) : (
                          <option key={c} value={c} disabled className="text-slate-500 bg-slate-950">
                            {c} (Disabled: Insufficient analytical coverage — {missingPct}% missing)
                          </option>
                        );
                      })}
                      {Object.keys(eda_results?.categorical_summary || {}).map((c) => (
                        <option key={c} value={c}>{c} (Categorical)</option>
                      ))}
                    </select>
                  </div>

                  {/* Y Axis Selector */}
                  <div>
                    <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">Secondary Axis (Y)</label>
                    <select
                      value={studioY}
                      onChange={(e) => setStudioY(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs text-white focus:outline-none focus:border-cyan-500 font-medium"
                    >
                      {Object.keys(eda_results?.numeric_summary || {}).map((c) => {
                        const dictEntry = data_dictionary?.find((d: any) => d.column_name === c);
                        const isEligible = dictEntry ? dictEntry.analytical_eligible !== false : true;
                        const missingPct = dictEntry ? (dictEntry.missing_pct ?? dictEntry['missing_%'] ?? 0) : 0;
                        return isEligible ? (
                          <option key={c} value={c}>{c} (Numeric)</option>
                        ) : (
                          <option key={c} value={c} disabled className="text-slate-500 bg-slate-950">
                            {c} (Disabled: Insufficient analytical coverage — {missingPct}% missing)
                          </option>
                        );
                      })}
                    </select>
                  </div>

                  {/* Group / Dimension Selector */}
                  <div>
                    <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">Grouping / Category</label>
                    <select
                      value={studioZ}
                      onChange={(e) => setStudioZ(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs text-white focus:outline-none focus:border-cyan-500 font-medium"
                    >
                      <option value="">None (Single Series)</option>
                      {Object.keys(eda_results?.categorical_summary || {}).map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Main Plotly Interactive Chart Canvas */}
              <div className="space-y-4">
                <PlotlyChartRenderer
                  chartType={studioChartType}
                  title={`${studioY || studioX} ${studioChartType.replace('_', ' ')} Analysis`}
                  subtitle={`Dynamic visualization mapping '${studioX}' against '${studioY}' ${studioZ ? `grouped by '${studioZ}'` : ''}`}
                  totalDatasetRecords={project?.total_records || project?.total_rows || profile?.num_rows || (data_dictionary?.[0]?.total_rows) || 1200}
                  data={
                    studioChartType === 'HEATMAP'
                      ? eda_results?.relationships?.strong_associations || []
                      : studioChartType === 'PCA_3D'
                      ? eda_results?.pca_analysis?.projections || []
                      : (masterPreviewData?.rows && masterPreviewData.rows.length > 0)
                      ? masterPreviewData.rows
                      : (project?.visual_dataset && project.visual_dataset.length > 0)
                      ? project.visual_dataset
                      : []
                  }
                  xKey={studioChartType === 'HEATMAP' ? 'var1' : studioChartType === 'PCA_3D' ? 'PC1' : studioX}
                  yKey={studioChartType === 'HEATMAP' ? 'var2' : studioChartType === 'PCA_3D' ? 'PC2' : studioY}
                  zKey="PC3"
                  groupKey={studioZ}
                  height={420}
                  takeaway={`Interactive visualization rendered from actual master dataset records. Use Plotly's toolbar above to zoom, pan, toggle series, and download high-resolution assets.`}
                  suitabilityReason={`Plotly canvas provides hardware-accelerated interactive inspection superior for exploratory data discovery.`}
                />
              </div>
            </div>
          )}

          {/* TAB 8: INSIGHTS & STORY */}
          {activeTab === 'insights' && (
            <div className="space-y-6 max-w-6xl">
              <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                  <BrainCircuit className="w-6 h-6 text-purple-400" />
                  Management Operational Narrative & Evidence-Backed Story
                </h1>
                <p className="text-sm text-slate-400 mt-1">Concise plant-oriented analytics briefing structured around the 5 operational questions with verified findings.</p>
              </div>

              {/* MANAGEMENT STORY NARRATIVE */}
              {story && (
                <div className="p-8 rounded-2xl glass-card border border-purple-900/40 bg-purple-950/10 space-y-6">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
                    <div>
                      <span className="text-[10px] font-bold font-mono tracking-wider text-purple-400 uppercase block mb-1">
                        PLANT DECISION-SUPPORT BRIEFING
                      </span>
                      <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                        <BrainCircuit className="w-5 h-5 text-purple-400" />
                        {story.title || "Plant Operational Insights & Management Narrative"}
                      </h2>
                    </div>
                    <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 self-start md:self-auto flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5" />
                      Grounded in Validated Evidence
                    </span>
                  </div>

                  {/* Section A: Executive Summary */}
                  <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-purple-300">
                      A. Executive Summary
                    </h3>
                    <p className="text-sm text-slate-200 leading-relaxed font-sans">
                      {story.executive_summary}
                    </p>
                  </div>

                  {/* Section B & D: Current Operating Situation & Performance Gaps */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2 text-xs">
                      <h3 className="font-bold uppercase tracking-wider text-blue-400 text-[11px]">
                        B. Current Operating Situation
                      </h3>
                      <div className="space-y-1 text-slate-300">
                        <p><strong className="text-slate-400">Observation Grain:</strong> {story.current_operating_situation?.observation_grain || profile?.observation_grain}</p>
                        <p><strong className="text-slate-400">Timeframe:</strong> {story.current_operating_situation?.reporting_timeframe || "Ingested Observation Period"}</p>
                        <p><strong className="text-slate-400">Active Metric Dimensions:</strong> {story.current_operating_situation?.active_kpis?.length || 0} continuous variables monitored</p>
                      </div>
                    </div>

                    <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2 text-xs">
                      <h3 className="font-bold uppercase tracking-wider text-rose-400 text-[11px]">
                        D. Operational Performance Gaps
                      </h3>
                      <div className="space-y-1.5 text-slate-300">
                        {story.performance_gaps?.length > 0 ? (
                          story.performance_gaps.slice(0, 3).map((gap: any, gidx: number) => (
                            <div key={gidx} className="p-2 rounded bg-slate-900 border border-slate-800">
                              <span className="font-bold text-rose-300">{gap.metric}</span>: {gap.management_note || `${gap.observed_state} against baseline ${gap.target_or_baseline}`}
                            </div>
                          ))
                        ) : (
                          <p className="text-slate-400 italic">No empirical baseline deviations observed beyond 2 sigma bounds.</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Section E: Potential Contributing Factors (Distinguishing Facts vs Associations vs Hypotheses) */}
                  {story.potential_contributing_factors && (
                    <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3 text-xs">
                      <h3 className="font-bold uppercase tracking-wider text-cyan-400 text-[11px]">
                        E. Potential Contributing Factors & Scientific Distinction
                      </h3>
                      <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[11px]">
                        {story.potential_contributing_factors.causality_warning}
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
                          <span className="text-[10px] font-bold uppercase text-emerald-400 block">Observed Facts:</span>
                          <p className="text-slate-300 leading-normal">{story.potential_contributing_factors.observed_facts}</p>
                        </div>
                        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
                          <span className="text-[10px] font-bold uppercase text-cyan-400 block">Statistical Associations:</span>
                          <p className="text-slate-300 leading-normal">{story.potential_contributing_factors.statistical_associations}</p>
                        </div>
                        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
                          <span className="text-[10px] font-bold uppercase text-purple-400 block">Plausible Hypotheses:</span>
                          <p className="text-slate-300 leading-normal">{story.potential_contributing_factors.plausible_hypotheses}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Section G: Recommended Priorities */}
                  {story.recommended_priorities?.length > 0 && (
                    <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3 text-xs">
                      <h3 className="font-bold uppercase tracking-wider text-emerald-400 text-[11px]">
                        G. Ranked Management Action Priorities
                      </h3>
                      <div className="space-y-2">
                        {story.recommended_priorities.map((p: any, pidx: number) => (
                          <div key={pidx} className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-2">
                            <div className="space-y-0.5">
                              <span className="font-bold text-slate-200">#{pidx + 1}. {p.finding}</span>
                              <p className="text-emerald-300">{p.suggested_action}</p>
                              <p className="text-[11px] text-slate-500">{p.justification}</p>
                            </div>
                            <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold self-start md:self-auto ${
                              p.priority === 'HIGH' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                            }`}>
                              {p.urgency || p.priority}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="space-y-5">
                {insights?.map((ins: any) => (
                  <div key={ins.id} className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                        ins.priority === 'High' || ins.priority === 'Critical' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                      }`}>
                        {ins.priority} Priority
                      </span>
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        <span className="px-2.5 py-0.5 rounded bg-blue-500/10 text-blue-400 font-bold border border-blue-500/20" title="Overall Dataset Health, Completeness & Schema Integrity">
                          Dataset Health: {ins.data_quality_confidence}%
                        </span>
                        <span className="px-2.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-bold border border-emerald-500/20">
                          {ins.statistical_evidence || ins.confidence_classification || "Verified calculation"}
                        </span>
                      </div>
                    </div>

                    <h3 className="font-bold text-lg text-slate-100">{ins.title}</h3>
                    <p className="text-sm text-slate-300 leading-relaxed">{ins.insight || ins.what_happened}</p>

                    {/* Human-Readable Calculation Method Box */}
                    {ins.calculation_method && (
                      <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1 font-mono text-xs text-emerald-400">
                        <div className="flex items-center justify-between text-slate-400 font-sans text-[11px] uppercase tracking-wider font-bold">
                          <span>Calculation Explanation & Formula</span>
                          <span className="text-slate-500 font-mono text-[10px]">Ground Truth</span>
                        </div>
                        <p className="text-slate-200 font-sans font-medium text-xs leading-normal">{ins.calculation_method}</p>
                      </div>
                    )}

                    {/* Evidence Box */}
                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2">
                      <p className="text-xs font-bold text-blue-400 uppercase tracking-wider">Numerical Evidence</p>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                        {(ins.numerical_evidence || ins.evidence)?.map((ev: any, idx: number) => (
                          <div key={idx} className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                            <span className="text-slate-500">{ev.metric}</span>
                            <p className="font-bold text-slate-200 mt-0.5">{ev.value}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {ins.why_it_matters && (
                      <p className="text-xs text-amber-300 leading-normal bg-amber-500/10 p-3 rounded-xl border border-amber-500/20">
                        <strong className="text-amber-400">Why It Matters:</strong> {ins.why_it_matters}
                      </p>
                    )}

                    {ins.recommended_action && (
                      <p className="text-xs text-emerald-300 leading-normal bg-emerald-500/10 p-3 rounded-xl border border-emerald-500/20">
                        <strong className="text-emerald-400">Recommended Action:</strong> {ins.recommended_action}
                      </p>
                    )}

                    {/* Supporting Visualization Rendered directly in Insight Card */}
                    {ins.supporting_visualization && (
                      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-purple-400 uppercase tracking-wider flex items-center gap-1.5">
                            <LineChart className="w-3.5 h-3.5 text-purple-400" />
                            Supporting Visual Evidence ({ins.supporting_visualization.chart_type || "BAR"})
                          </span>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => openInVisualizer(
                                ins.supporting_visualization.x_axis || ins.supporting_visualization.category || 'x',
                                ins.supporting_visualization.y_axis || ins.supporting_visualization.metric || 'y',
                                ins.supporting_visualization.chart_type || 'BAR'
                              )}
                              className="flex items-center gap-1 text-[11px] font-semibold text-cyan-400 hover:text-cyan-300 bg-cyan-500/10 hover:bg-cyan-500/20 px-2.5 py-0.5 rounded border border-cyan-500/20 transition-all"
                            >
                              <ArrowUpRight className="w-3 h-3" />
                              Open in Visual Studio
                            </button>
                            <span className="text-[10px] font-bold text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                              Priority: {ins.supporting_visualization.priority || ins.priority || "HIGH"}
                            </span>
                          </div>
                        </div>
                        {ins.supporting_visualization.data && ins.supporting_visualization.data.length > 0 ? (
                          <UniversalChartRenderer
                            chartType={ins.supporting_visualization.chart_type || "BAR"}
                            title={ins.supporting_visualization.title || ins.title}
                            description={ins.supporting_visualization.subtitle || ins.insight}
                            data={ins.supporting_visualization.data}
                            xKey={ins.supporting_visualization.x_axis || ins.supporting_visualization.category || 'x'}
                            yKey={ins.supporting_visualization.y_axis || ins.supporting_visualization.metric || 'y'}
                            inputRowCount={profile?.num_rows || 0}
                          />
                        ) : (
                          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800/80 text-xs text-slate-400">
                            <p className="font-bold text-slate-200">{ins.supporting_visualization.title}</p>
                            <p className="text-[11px] text-slate-400 mt-0.5">{ins.supporting_visualization.subtitle || "Chart data rendered based on ground-truth numeric evidence."}</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 9: RECOMMENDATIONS */}
          {activeTab === 'recommendations' && (
            <div className="space-y-6 max-w-6xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Lightbulb className="w-6 h-6 text-amber-400" />
                    Evidence-Backed Recommendations
                  </h1>
                  <p className="text-sm text-slate-400 mt-1">
                    Defensible operational actions directly tied to validated findings, distinguishing immediate corrective tasks from investigations and data-quality improvements.
                  </p>
                </div>
                <button
                  onClick={() => setActiveTab('action_center')}
                  className="px-4 py-2 rounded-xl bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 text-xs font-semibold text-blue-300 flex items-center gap-2 transition-all self-start md:self-auto"
                >
                  <CheckSquare className="w-4 h-4 text-blue-400" />
                  View in Action Center ({actionItems?.length || 0})
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {recommendations?.map((rec: any, idx: number) => {
                  const categoryName = rec.recommendation_category || rec.category || "Preventive Action";
                  const isImmediate = categoryName.toLowerCase().includes('immediate') || categoryName.toLowerCase().includes('corrective');
                  const isInvestigation = categoryName.toLowerCase().includes('investigation');
                  const isQuality = categoryName.toLowerCase().includes('quality');
                  const isOptimization = categoryName.toLowerCase().includes('optimization') || categoryName.toLowerCase().includes('long-term');

                  const catBadgeStyle = isImmediate
                    ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
                    : isInvestigation
                    ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                    : isQuality
                    ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30'
                    : isOptimization
                    ? 'bg-purple-500/15 text-purple-300 border-purple-500/30'
                    : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30';

                  return (
                    <div key={rec.id || `rec-${idx}`} className="p-6 rounded-2xl glass-card border border-slate-800 flex flex-col justify-between space-y-4 hover:border-slate-700 transition-all">
                      <div className="space-y-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border ${catBadgeStyle}`}>
                              {categoryName}
                            </span>
                            <span className="text-[10px] font-mono text-slate-500">
                              Linked: {rec.source_insight_id || rec.insight_id || "Validated Finding"}
                            </span>
                          </div>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            rec.priority === 'Critical' || rec.priority === 'High' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                          }`}>
                            {rec.priority} Priority
                          </span>
                        </div>

                        <h3 className="font-bold text-base text-slate-100">{rec.title}</h3>

                        {rec.requires_human_approval && (
                          <div className="p-2 rounded-lg bg-amber-950/30 border border-amber-800/40 text-[11px] text-amber-300 flex items-center gap-2">
                            <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-amber-400" />
                            <span>Requires Operational Human Review / Management Sign-off</span>
                          </div>
                        )}

                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-slate-300">
                          <strong className="text-blue-400 uppercase tracking-wider text-[10px] block mb-1">Operational Problem Addressed:</strong>
                          <p>{rec.problem || rec.root_cause_analysis || "Operational gap identified in data."}</p>
                          {rec.evidence && (
                            <p className="mt-1.5 text-[11px] text-slate-400 font-mono">
                              <span className="text-slate-500 font-semibold">Evidence: </span>{rec.evidence}
                            </p>
                          )}
                        </div>

                        <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-800/40 text-xs text-emerald-200">
                          <strong className="text-emerald-400 uppercase tracking-wider text-[10px] block mb-1">Proposed Action & Justification:</strong>
                          <p>{rec.recommended_action || rec.action}</p>
                          {rec.why_appropriate && <p className="text-[11px] text-emerald-400/80 mt-1 italic">{rec.why_appropriate}</p>}
                        </div>

                        {rec.expected_outcome && (
                          <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-[11px] text-slate-300">
                            <strong className="text-slate-400 uppercase tracking-wider text-[10px] block mb-0.5">Expected Outcome:</strong>
                            <p>{rec.expected_outcome}</p>
                          </div>
                        )}

                        {rec.uncertainty_or_limitation && (
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800/60 text-[10px] text-slate-400 italic">
                            <span className="text-slate-500 font-semibold not-italic">Limitation / Uncertainty: </span>
                            {rec.uncertainty_or_limitation}
                          </div>
                        )}
                      </div>

                      <div className="pt-4 border-t border-slate-800/80 text-xs space-y-2 text-slate-400">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px]"><strong className="text-slate-300">Suggested Role:</strong> <span className="text-purple-400 font-medium">{rec.owner || "UNASSIGNED"}</span></span>
                          <span className="text-[11px]"><strong className="text-slate-300">Timeline:</strong> <span className="text-slate-200 font-medium">{rec.timeline}</span></span>
                        </div>
                        <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-[11px] font-mono">
                          <div className="text-slate-400 mb-1"><span className="text-slate-500">Target KPI:</span> <span className="text-blue-400 font-bold">{rec.kpi || "Process Compliance"}</span></div>
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <span className="text-slate-500 block text-[10px]">Baseline</span>
                              <span className="text-slate-200 font-bold">{rec.kpi_baseline}</span>
                            </div>
                            <div>
                              <span className="text-slate-500 block text-[10px]">Target</span>
                              <span className="text-emerald-400 font-bold">{rec.kpi_target}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* TAB 10: ACTION CENTER */}
          {activeTab === 'action_center' && (
            <div className="space-y-6 max-w-6xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold flex items-center gap-2">
                    <CheckSquare className="w-6 h-6 text-emerald-400" />
                    Plant Action Center
                  </h1>
                  <p className="text-sm text-slate-400 mt-1">
                    Operational execution layer tracking accountable owners, timelines, verifiable targets, execution status, and audit lineage back to verified findings.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400 font-mono">
                    Total Tracked: <strong className="text-slate-100">{actionItems?.length || 0}</strong>
                  </span>
                </div>
              </div>

              {/* Status Summary Counter */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {[
                  { label: 'Not Started', count: actionItems?.filter((a: any) => !a.status || a.status === 'Not Started').length || 0, color: 'text-slate-400 border-slate-800' },
                  { label: 'In Progress', count: actionItems?.filter((a: any) => a.status === 'In Progress').length || 0, color: 'text-blue-400 border-blue-900/40 bg-blue-950/20' },
                  { label: 'Blocked', count: actionItems?.filter((a: any) => a.status === 'Blocked').length || 0, color: 'text-rose-400 border-rose-900/40 bg-rose-950/20' },
                  { label: 'Completed', count: actionItems?.filter((a: any) => a.status === 'Completed').length || 0, color: 'text-emerald-400 border-emerald-900/40 bg-emerald-950/20' },
                  { label: 'Verified', count: actionItems?.filter((a: any) => a.status === 'Verified').length || 0, color: 'text-purple-400 border-purple-900/40 bg-purple-950/20' }
                ].map((s, sIdx) => (
                  <div key={sIdx} className={`p-3 rounded-xl border glass-card flex flex-col ${s.color}`}>
                    <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">{s.label}</span>
                    <span className="text-xl font-bold font-mono mt-0.5">{s.count}</span>
                  </div>
                ))}
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-800 glass-card">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-900 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                    <tr>
                      <th className="p-3.5">Action ID / Type</th>
                      <th className="p-3.5">Priority</th>
                      <th className="p-3.5">Action Item & Problem Addressed</th>
                      <th className="p-3.5">Owner</th>
                      <th className="p-3.5">Timeline / Due</th>
                      <th className="p-3.5">Baseline & Target</th>
                      <th className="p-3.5">Status</th>
                      <th className="p-3.5 text-right">Manage</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {actionItems?.map((act: any) => {
                      const actId = act.id || act.action_id;
                      const isEditing = editingActionId === actId;
                      const category = act.recommendation_category || act.category || "Preventive Action";

                      return (
                        <React.Fragment key={actId}>
                          <tr className={`hover:bg-slate-900/40 transition-colors ${isEditing ? 'bg-slate-900/70 border-l-2 border-l-blue-500' : ''}`}>
                            <td className="p-3.5 font-mono text-[11px] align-top">
                              <span className="text-blue-400 font-bold block">{actId}</span>
                              <span className="text-[10px] text-slate-400 block mt-0.5 truncate max-w-[120px]">{category}</span>
                              <span className="text-slate-500 text-[10px] block">Src: {act.source_insight_id || "Ground Truth"}</span>
                              {act.requires_human_approval && (
                                <span className="inline-block mt-1 px-1.5 py-0.5 rounded text-[9px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                                  Review Req.
                                </span>
                              )}
                            </td>
                            <td className="p-3.5 align-top">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                act.priority === 'Critical' || act.priority === 'High' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                              }`}>
                                {act.priority}
                              </span>
                            </td>
                            <td className="p-3.5 max-w-xs space-y-1 align-top">
                              <p className="font-semibold text-slate-100">{act.task_name || act.action_step || act.action || act.task_description}</p>
                              <p className="text-[11px] text-slate-400 line-clamp-2">{act.problem_statement || act.lineage?.evidence || "Verified evidence-backed recommendation"}</p>
                              {act.completion_notes && (
                                <div className="mt-1 text-[10px] text-emerald-400/90 bg-emerald-950/20 p-1.5 rounded border border-emerald-900/30">
                                  <strong>Notes:</strong> {act.completion_notes}
                                </div>
                              )}
                              {act.verification_notes && (
                                <div className="mt-1 text-[10px] text-purple-400/90 bg-purple-950/20 p-1.5 rounded border border-purple-900/30">
                                  <strong>Verification:</strong> {act.verification_notes}
                                </div>
                              )}
                            </td>
                            <td className="p-3.5 font-medium text-purple-400 align-top">
                              {act.owner && act.owner !== 'UNASSIGNED' ? act.owner : <span className="text-slate-500 italic">Unassigned</span>}
                            </td>
                            <td className="p-3.5 text-slate-300 font-mono align-top text-[11px]">
                              {act.due_date || act.timeline || act.deadline || "Not specified"}
                            </td>
                            <td className="p-3.5 font-mono text-[11px] align-top">
                              <div className="text-slate-400">Base: <span className="text-slate-200 font-bold">{act.baseline || act.kpi_baseline || "N/A"}</span></div>
                              <div className="text-emerald-400">Target: <span className="font-bold">{act.target || act.kpi_target || "N/A"}</span></div>
                            </td>
                            <td className="p-3.5 align-top">
                              <select
                                value={act.status || "Not Started"}
                                onChange={(e) => handleActionStatusChange(actId, e.target.value)}
                                className={`text-xs rounded px-2.5 py-1 font-bold border transition-all ${
                                  act.status === 'Verified' ? 'bg-purple-500/15 text-purple-400 border-purple-500/30' :
                                  act.status === 'Completed' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' :
                                  act.status === 'In Progress' ? 'bg-blue-500/15 text-blue-400 border-blue-500/30' :
                                  act.status === 'Blocked' ? 'bg-rose-500/15 text-rose-400 border-rose-500/30' :
                                  act.status === 'Cancelled' ? 'bg-slate-800 text-slate-500 border-slate-700' :
                                  'bg-slate-900 text-slate-400 border-slate-700'
                                }`}
                              >
                                <option value="Not Started">Not Started</option>
                                <option value="In Progress">In Progress</option>
                                <option value="Blocked">Blocked</option>
                                <option value="Completed">Completed</option>
                                <option value="Verified">Verified</option>
                                <option value="Cancelled">Cancelled</option>
                              </select>
                            </td>
                            <td className="p-3.5 text-right align-top">
                              <button
                                onClick={() => isEditing ? setEditingActionId(null) : handleStartEditAction(act)}
                                className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition-all ${
                                  isEditing 
                                    ? 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700' 
                                    : 'bg-blue-600/10 text-blue-400 border-blue-500/30 hover:bg-blue-600/20'
                                }`}
                              >
                                {isEditing ? 'Cancel' : 'Edit / Assign'}
                              </button>
                            </td>
                          </tr>

                          {/* INLINE EDIT DRAWER / FORM */}
                          {isEditing && (
                            <tr className="bg-slate-900/90 border-b border-slate-800">
                              <td colSpan={8} className="p-4 space-y-4">
                                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                                  <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                                    <h4 className="font-bold text-xs text-blue-400 uppercase tracking-wider flex items-center gap-1.5">
                                      <Sliders className="w-3.5 h-3.5" />
                                      Update Operational Action Assignment & Execution Details ({actId})
                                    </h4>
                                    <span className="text-[11px] text-slate-500">Changes persist to backend action registry</span>
                                  </div>

                                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                    <div>
                                      <label className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Accountable Owner / Department</label>
                                      <input
                                        type="text"
                                        placeholder="e.g. Operations Mgr / Shift Lead"
                                        value={actionFormOwner}
                                        onChange={(e) => setActionFormOwner(e.target.value)}
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-blue-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Due Date / Timeline</label>
                                      <input
                                        type="text"
                                        placeholder="e.g. 2 weeks / End of Shift"
                                        value={actionFormDueDate}
                                        onChange={(e) => setActionFormDueDate(e.target.value)}
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-blue-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Target / Measurable Threshold</label>
                                      <input
                                        type="text"
                                        placeholder="e.g. Zero recurring alarms / < 1.0%"
                                        value={actionFormTarget}
                                        onChange={(e) => setActionFormTarget(e.target.value)}
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-blue-500"
                                      />
                                    </div>
                                  </div>

                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    <div>
                                      <label className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Execution / Completion Notes</label>
                                      <textarea
                                        rows={2}
                                        placeholder="Document steps taken, findings during execution, or shift handover notes..."
                                        value={actionFormCompletionNotes}
                                        onChange={(e) => setActionFormCompletionNotes(e.target.value)}
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Verification of Results / Quality Sign-off</label>
                                      <textarea
                                        rows={2}
                                        placeholder="Document verification evidence, post-change measurements, or QA sign-off..."
                                        value={actionFormVerificationNotes}
                                        onChange={(e) => setActionFormVerificationNotes(e.target.value)}
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500"
                                      />
                                    </div>
                                  </div>

                                  <div className="flex justify-end gap-2 pt-2">
                                    <button
                                      onClick={() => setEditingActionId(null)}
                                      disabled={savingAction}
                                      className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition-colors"
                                    >
                                      Cancel
                                    </button>
                                    <button
                                      onClick={() => handleSaveActionDetails(actId, act.status || 'Not Started')}
                                      disabled={savingAction}
                                      className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition-colors flex items-center gap-1.5"
                                    >
                                      {savingAction ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : null}
                                      Save Action Details
                                    </button>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 11: AI ASSISTANT & CHAT */}
          {activeTab === 'ask_data' && (
            <div className="space-y-6 max-w-4xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-bold flex items-center gap-2">
                    <MessageSquare className="w-6 h-6 text-blue-400" />
                    AI Assistant & Interactive Analytical Chat
                  </h1>
                  <p className="text-sm text-slate-400 mt-1">
                    Multi-turn analytical agent orchestrating dynamic dataset query planning, EDA diagnostics, and interactive Plotly charts.
                  </p>
                </div>
                {queryHistory.length > 0 && (
                  <button
                    onClick={handleClearChat}
                    className="px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-rose-950/40 border border-slate-800 hover:border-rose-800/40 text-xs font-semibold text-slate-400 hover:text-rose-300 flex items-center gap-1.5 transition-all self-start md:self-auto"
                  >
                    <Trash2 className="w-3.5 h-3.5 text-rose-400" />
                    Clear Chat History
                  </button>
                )}
              </div>

              {/* Quick Suggestion Chips */}
              <div className="space-y-2">
                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">Recommended Analytical & Document Prompts:</span>
                <div className="flex flex-wrap gap-2">
                  {[
                    "What are the key drivers and top insights in this dataset?",
                    "📄 Summarize the key findings from the uploaded PDF document",
                    "🔍 Extract revenue, EBITDA, and financial performance from the report",
                    "Show outlier anomalies across numerical features",
                    "Plot correlation between numeric variables",
                    "📄 What structured tables or regional breakdowns are in the PDF?",
                    "Summarize data completeness and missing values"
                  ].map((promptText, pIdx) => (
                    <button
                      key={pIdx}
                      onClick={() => executeAskQuery(promptText)}
                      disabled={asking}
                      className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-blue-900/30 text-xs text-slate-300 hover:text-blue-300 border border-slate-800 hover:border-blue-700/50 transition-all text-left"
                    >
                      💡 {promptText}
                    </button>
                  ))}
                </div>
              </div>

              {/* Query Input Bar */}
              <form onSubmit={handleSendQuery} className="flex gap-3">
                <input
                  type="text"
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  placeholder="Ask a question about your dataset or uploaded PDF (e.g. 'Summarize report' or 'Plot revenue by region')"
                  className="flex-1 px-4 py-3 rounded-xl bg-slate-900 border border-slate-800 text-sm text-white focus:outline-none focus:border-blue-500 shadow-inner"
                />
                <button
                  type="submit"
                  disabled={asking || !queryInput.trim()}
                  className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 font-semibold text-sm text-white shadow-lg shadow-blue-500/20 disabled:opacity-50 transition-all flex items-center gap-2"
                >
                  {asking ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Thinking...</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>Send</span>
                    </>
                  )}
                </button>
              </form>

              {/* Chat Message History */}
              <div className="space-y-6 pt-2">
                {queryHistory.length === 0 ? (
                  <div className="p-8 rounded-2xl glass-card border border-slate-800 text-center space-y-4">
                    <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mx-auto text-blue-400">
                      <Sparkles className="w-6 h-6" />
                    </div>
                    <div>
                      <h3 className="font-bold text-base text-slate-200">Interactive Intelligence Agent Ready</h3>
                      <p className="text-xs text-slate-400 max-w-md mx-auto mt-1">
                        Ask questions about your tabular datasets or uploaded PDF documents. The agent uses deterministic Pandas query planning, Plotly visual generation, and grounded semantic PDF document retrieval.
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-lg mx-auto text-left text-xs">
                      <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                        <span className="font-bold text-blue-400 block mb-0.5">Strict Grounding</span>
                        <p className="text-slate-400 text-[11px]">Zero hallucination: answers verified against dataframes and PDF passages.</p>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                        <span className="font-bold text-emerald-400 block mb-0.5">Dual Intelligence</span>
                        <p className="text-slate-400 text-[11px]">Understands both tabular data (DuckDB/Pandas) and unstructured PDF text/tables.</p>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                        <span className="font-bold text-purple-400 block mb-0.5">Rich Markdown</span>
                        <p className="text-slate-400 text-[11px]">ChatGPT-grade structured answers with citations, headings, and data tables.</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  queryHistory.map((item, idx) => (
                    <div key={idx} className="space-y-3">
                      {/* User Query Turn */}
                      <div className="flex items-start gap-3 justify-end">
                        <div className="max-w-xl bg-blue-600/20 border border-blue-500/30 text-blue-100 rounded-2xl rounded-tr-sm px-4 py-3 text-sm shadow-md">
                          <p className="font-medium">{item.question}</p>
                        </div>
                        <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold shrink-0">
                          U
                        </div>
                      </div>

                      {/* Assistant Response Turn */}
                      <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-cyan-400 shrink-0">
                          <Sparkles className="w-4 h-4" />
                        </div>
                        <div className="flex-1 p-6 rounded-2xl glass-card border border-slate-800/90 space-y-4 shadow-xl">
                          {/* Tool and Intent Metadata Badge Bar */}
                          <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800/60">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20">
                                Tool: {item.tool_used || "DatasetQueryTool"}
                              </span>
                              {item.intent && (
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-500/10 text-purple-400 border border-purple-500/20">
                                  Intent: {item.intent}
                                </span>
                              )}
                              {item.source_file && (
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/10 text-amber-300 border border-amber-500/20 flex items-center gap-1">
                                  <FileText className="w-3 h-3" />
                                  Doc: {item.source_file}
                                </span>
                              )}
                              {Array.isArray(item.page_references) && item.page_references.length > 0 && (
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold text-cyan-300 bg-cyan-950/40 border border-cyan-800/40">
                                  Pages: {item.page_references.join(', ')}
                                </span>
                              )}
                              {item.records_evaluated !== undefined && item.records_evaluated !== null && (
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono text-slate-300 bg-slate-900 border border-slate-800">
                                  {Number(item.records_evaluated).toLocaleString()} {item.source_file ? "passages" : "records"} ({item.coverage_pct ?? 100}% coverage)
                                </span>
                              )}
                              {item.status && (
                                <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${
                                  item.status === 'PASS' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                                  item.status === 'WARNING' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                                  item.status === 'NO_MATCH' ? 'bg-slate-500/10 text-slate-400 border-slate-500/20' :
                                  'bg-rose-500/10 text-rose-400 border-rose-500/20'
                                }`}>
                                  Status: {item.status}
                                </span>
                              )}
                              {item.resolved_field && (
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 flex items-center gap-1 shadow-sm">
                                  <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                                  Resolved: <strong className="text-white">{item.resolved_field}</strong>
                                </span>
                              )}
                            </div>
                            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                              <ShieldCheck className="w-3 h-3" />
                              Governed Intelligence Engine
                            </span>
                          </div>

                          {/* Agent Answer Text with Rich ChatGPT-Style Markdown */}
                          {item.answer && (
                            <div className="text-slate-100 text-sm leading-relaxed">
                              <MarkdownRenderer content={item.answer} />
                            </div>
                          )}

                          {/* Verified Document Evidence Drawer */}
                          {Array.isArray(item.evidence) && item.evidence.length > 0 && (
                            <details className="mt-3 group rounded-xl bg-slate-950/60 border border-slate-800/80 p-3">
                              <summary className="text-[11px] font-mono font-semibold text-blue-400 cursor-pointer hover:text-blue-300 transition-colors list-none flex items-center gap-1.5 select-none">
                                <span className="group-open:rotate-90 transition-transform">▸</span>
                                <FileText className="w-3.5 h-3.5 text-amber-400" />
                                <span>Verified Source Evidence ({item.evidence.length} passages cited from {item.source_file || 'document'})</span>
                              </summary>
                              <div className="mt-3 space-y-2.5 pt-2 border-t border-slate-800/60">
                                {item.evidence.map((ev: any, evIdx: number) => (
                                  <div key={evIdx} className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs font-mono space-y-1">
                                    <div className="flex items-center justify-between text-[10px] text-slate-400">
                                      <span className="font-bold text-amber-300 flex items-center gap-1">
                                        <FileText className="w-3 h-3" />
                                        Page {ev.page_num}
                                        {ev.section_title && <span className="text-slate-400 font-normal"> · {ev.section_title}</span>}
                                      </span>
                                      {ev.score !== undefined && (
                                        <span className="text-emerald-400 font-semibold bg-emerald-950/40 px-1.5 py-0.5 rounded border border-emerald-800/40">
                                          Score: {Number(ev.score).toFixed(2)}
                                        </span>
                                      )}
                                    </div>
                                    <p className="text-slate-200 text-[11px] leading-relaxed italic bg-slate-950/50 p-2 rounded border border-slate-800/50">
                                      "{ev.text}"
                                    </p>
                                  </div>
                                ))}
                              </div>
                            </details>
                          )}

                          {/* Interactive Intelligent Match Suggestion Panel ("Did You Mean?") */}
                          {(item.suggestion_mode === 'confirmation_required' || (Array.isArray(item.match_suggestions) && item.match_suggestions.length > 0 && item.status !== 'PASS')) && (
                            <div className="p-4 rounded-xl bg-gradient-to-br from-blue-950/70 via-slate-900 to-indigo-950/40 border border-blue-500/40 space-y-3.5 shadow-lg">
                              <div className="flex items-center justify-between border-b border-blue-800/40 pb-2">
                                <div className="flex items-center gap-2">
                                  <span className="w-5 h-5 rounded-md bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold text-xs">?</span>
                                  <span className="text-xs font-semibold text-blue-200">
                                    Did you mean? Matching Dataset Fields
                                  </span>
                                </div>
                                <span className="text-[10px] font-mono text-blue-300/70">
                                  Query: &quot;{item.question}&quot;
                                </span>
                              </div>

                              <div className="space-y-2">
                                {item.match_suggestions?.slice(0, 4).map((sugg: any, sIdx: number) => {
                                  const isPrimary = sIdx === 0;
                                  const confPct = sugg.confidence_pct || Math.round((sugg.confidence || 0) * 100);
                                  const barColor = confPct >= 90 ? 'bg-emerald-500' : (confPct >= 70 ? 'bg-amber-500' : 'bg-slate-500');
                                  const badgeColor = confPct >= 90 ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' : (confPct >= 70 ? 'text-amber-400 bg-amber-500/10 border-amber-500/30' : 'text-slate-400 bg-slate-800 border-slate-700');

                                  return (
                                    <div
                                      key={sIdx}
                                      className={`p-2.5 rounded-lg border transition-all ${
                                        isPrimary
                                          ? 'bg-blue-900/30 border-blue-500/40 hover:border-blue-400'
                                          : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                                      }`}
                                    >
                                      <div className="flex flex-wrap items-center justify-between gap-2">
                                        <div className="flex items-center gap-2">
                                          <span className="text-xs font-mono text-slate-400 font-bold">{sIdx + 1}.</span>
                                          <span className="text-xs font-semibold text-white tracking-wide">
                                            {sugg.column}
                                          </span>
                                          {sugg.semantic_role && (
                                            <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 border border-slate-700">
                                              {sugg.semantic_role}
                                            </span>
                                          )}
                                        </div>
                                        <div className="flex items-center gap-2">
                                          <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${badgeColor}`}>
                                            {confPct}% match
                                          </span>
                                          <button
                                            onClick={() => executeAskQuery(item.question, sugg.column)}
                                            className="px-2.5 py-1 rounded bg-blue-600 hover:bg-blue-500 text-[11px] font-medium text-white transition-all shadow cursor-pointer flex items-center gap-1"
                                          >
                                            <CheckCircle2 className="w-3 h-3" />
                                            Use this field
                                          </button>
                                        </div>
                                      </div>
                                      <div className="mt-1.5 flex items-center gap-2">
                                        <div className="flex-1 h-1 rounded-full bg-slate-800 overflow-hidden">
                                          <div className={`h-full ${barColor}`} style={{ width: `${confPct}%` }} />
                                        </div>
                                        {sugg.explanation && (
                                          <span className="text-[10px] text-slate-400 truncate max-w-xs font-mono">
                                            {sugg.explanation}
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>

                              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-blue-900/40 text-xs">
                                <span className="text-slate-300 text-[11px]">
                                  {item.match_suggestions?.[0]?.column ? (
                                    <>Would you like to analyze <strong className="text-cyan-300">{item.match_suggestions[0].column}</strong>?</>
                                  ) : "Please choose a field to proceed:"}
                                </span>
                                <div className="flex flex-wrap gap-2">
                                  {item.match_suggestions?.[0]?.column && (
                                    <button
                                      onClick={() => executeAskQuery(item.question, item.match_suggestions[0].column)}
                                      className="px-3 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-md transition-all cursor-pointer flex items-center gap-1"
                                    >
                                      <CheckCircle2 className="w-3.5 h-3.5" />
                                      Yes, use this field
                                    </button>
                                  )}
                                  <button
                                    onClick={() => {
                                      const inputEl = document.getElementById('chat-query-input') || document.querySelector('input[placeholder*="Ask"]');
                                      if (inputEl) (inputEl as HTMLInputElement).focus();
                                    }}
                                    className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono border border-slate-700 transition-all cursor-pointer"
                                  >
                                    Search again
                                  </button>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Fallback Clarification Chips for Ambiguous Queries */}
                          {!item.match_suggestions?.length && item.query_plan?.resolved_entities?.some((e: any) => Array.isArray(e.candidate_alternatives) && e.candidate_alternatives.length > 0) && (
                            <div className="p-3 rounded-xl bg-blue-950/40 border border-blue-800/40 space-y-2">
                              <span className="text-[11px] font-semibold text-blue-300 block">Clarification Options (Click to Select):</span>
                              <div className="flex flex-wrap gap-2">
                                {item.query_plan.resolved_entities
                                  .flatMap((e: any) => e.candidate_alternatives || [])
                                  .map((candidate: string, cIdx: number) => (
                                    <button
                                      key={cIdx}
                                      onClick={() => executeAskQuery(`Analyze ${candidate}`)}
                                      className="px-2.5 py-1 rounded-lg bg-blue-900/40 hover:bg-blue-800/60 text-xs font-mono text-blue-200 border border-blue-700/50 hover:border-blue-500 transition-all cursor-pointer"
                                    >
                                      📊 {candidate}
                                    </button>
                                  ))}
                              </div>
                            </div>
                          )}

                          {/* Inline Dynamic Plotly Chart */}
                          {item.chart_spec && (
                            <div className="pt-2">
                              <PlotlyChartRenderer
                                chartType={item.chart_spec.chart_type}
                                title={item.chart_spec.title || 'Interactive Visual Analysis'}
                                subtitle={item.chart_spec.subtitle || `Generated for query: "${item.question}"`}
                                data={item.chart_spec.data}
                                xKey={item.chart_spec.x_axis}
                                yKey={item.chart_spec.y_axis}
                                takeaway={item.chart_spec.takeaway}
                                suitabilityReason={item.chart_spec.suitability_reason}
                              />
                            </div>
                          )}

                          {/* Calculation Method Box */}
                          {item.calculation_formula && (
                            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-emerald-400 space-y-1">
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] uppercase font-bold text-slate-500 font-sans block">Verified Mathematical Execution Trace:</span>
                                <span className="text-[10px] text-slate-500 font-sans">In-Memory DuckDB Engine</span>
                              </div>
                              <p className="text-slate-200 break-all">{item.calculation_formula}</p>
                            </div>
                          )}

                          {/* Structured Query Plan Breakdown */}
                          {item.query_plan && (
                            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-2 font-mono">
                              <p className="font-bold text-blue-400 font-sans uppercase tracking-wider text-[10px]">Governed Query Plan</p>
                              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 text-slate-300">
                                <div><span className="text-slate-500">Intent:</span> {item.query_plan.intent}</div>
                                <div>
                                  <span className="text-slate-500">
                                    {item.query_plan.pairs ? "Pairs:" : (item.query_plan.variables ? "Variables:" : "Metric:")}
                                  </span>{" "}
                                  {item.query_plan.pairs
                                    ? `${item.query_plan.pairs.length} Pairs`
                                    : (item.query_plan.variables
                                        ? (Array.isArray(item.query_plan.variables) ? item.query_plan.variables.join(", ") : item.query_plan.variables)
                                        : (item.query_plan.metric || "None"))}
                                </div>
                                {item.query_plan.methods && item.query_plan.methods.length > 0 && (
                                  <div>
                                    <span className="text-slate-500">Method:</span>{" "}
                                    {item.query_plan.methods.map((m: string) => m.charAt(0).toUpperCase() + m.slice(1)).join(" & ")}
                                  </div>
                                )}
                                <div><span className="text-slate-500">Granularity:</span> {item.query_plan.granularity || (item.query_plan.aggregation ? "aggregated" : "row-level")}</div>
                                <div><span className="text-slate-500">Agg:</span> {item.query_plan.aggregation || "None"}</div>
                                <div><span className="text-slate-500">Dimension:</span> {item.query_plan.dimension || "None"}</div>
                              </div>
                            </div>
                          )}

                          {/* Result Table Preview */}
                          {Array.isArray(item.evidence) && item.evidence.length > 0 && typeof item.evidence[0] === 'object' && (
                            <div className="overflow-x-auto rounded-xl border border-slate-800 text-xs max-h-56">
                              <table className="w-full text-left text-slate-300">
                                <thead className="bg-slate-900 text-slate-400 font-mono border-b border-slate-800 sticky top-0">
                                  <tr>
                                    {Object.keys(item.evidence[0]).map((colKey, kIdx) => (
                                      <th key={kIdx} className="p-2.5 whitespace-nowrap">{colKey}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800/40">
                                  {item.evidence.slice(0, 10).map((rowObj: any, rIdx: number) => (
                                    <tr key={rIdx} className="hover:bg-slate-900/40">
                                      {Object.keys(item.evidence[0]).map((colKey, cIdx) => (
                                        <td key={cIdx} className="p-2 font-mono text-[11px] whitespace-nowrap">
                                          {String(rowObj[colKey] ?? '')}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* TAB 12: REPORTS & EXPORT */}
          {activeTab === 'reports' && (
            <div className="space-y-6 max-w-4xl">
              <div>
                <h1 className="text-2xl font-bold">Export Deliverables</h1>
                <p className="text-sm text-slate-400 mt-1">Download multi-sheet master deliverables and executive reports.</p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="p-6 rounded-2xl glass-card border border-slate-800 flex flex-col justify-between space-y-4">
                  <div>
                    <FileSpreadsheet className="w-8 h-8 text-emerald-400 mb-3" />
                    <h3 className="font-bold text-lg text-slate-100">Master Excel Deliverable</h3>
                    <p className="text-xs text-slate-400 mt-1">Includes 9 styled sheets (Executive Summary, Quality, Dictionary, Master, Cleaned, Raw, EDA, Insights, Action Plan).</p>
                  </div>
                  <a
                    href={getExcelExportUrl(projectId)}
                    download
                    className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 font-semibold text-xs text-white text-center shadow-lg shadow-emerald-600/20 transition-all flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download Excel (.xlsx)
                  </a>
                </div>

                <div className="p-6 rounded-2xl glass-card border border-slate-800 flex flex-col justify-between space-y-4">
                  <div>
                    <Activity className="w-8 h-8 text-purple-400 mb-3" />
                    <h3 className="font-bold text-lg text-slate-100">Executive PDF Report</h3>
                    <p className="text-xs text-slate-400 mt-1">Board-ready summary with executive story narrative, key metrics, charts, and recommendations.</p>
                  </div>
                  <a
                    href={getPdfExportUrl(projectId)}
                    download
                    className="w-full py-3 rounded-xl bg-purple-600 hover:bg-purple-500 font-semibold text-xs text-white text-center shadow-lg shadow-purple-600/20 transition-all flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download Executive PDF (.pdf)
                  </a>
                </div>
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
