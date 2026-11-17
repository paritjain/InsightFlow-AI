'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Download, Sparkles, Code2, ZoomIn, RefreshCw, AlertTriangle } from 'lucide-react';

export function toNumericValue(val: any): number | null {
  if (val === null || val === undefined) return null;
  if (typeof val === 'number') {
    return Number.isFinite(val) ? val : null;
  }
  if (typeof val === 'boolean') return null;
  if (typeof val === 'string') {
    const trimmed = val.trim();
    if (trimmed === '' || trimmed.toLowerCase() === 'nan' || trimmed.toLowerCase() === 'null' || trimmed.toLowerCase() === 'undefined') {
      return null;
    }
    const cleanStr = trimmed.replace(/[$€£¥₹,\s]/g, '');
    const num = Number(cleanStr);
    return Number.isFinite(num) ? num : null;
  }
  return null;
}

interface PlotlyChartProps {
  chartType: string; // 'HISTOGRAM' | 'BOX_PLOT' | 'VIOLIN' | 'SCATTER' | 'BAR' | 'LINE' | 'HEATMAP' | 'STACKED_BAR' | 'PCA_3D' | 'PCA_2D'
  title: string;
  subtitle?: string;
  data: any[];
  xKey?: string;
  yKey?: string;
  zKey?: string;
  groupKey?: string;
  takeaway?: string;
  suitabilityReason?: string;
  height?: number;
  totalDatasetRecords?: number;
}

export default function PlotlyChartRenderer({
  chartType,
  title,
  subtitle,
  data,
  xKey = 'x',
  yKey = 'y',
  zKey = 'z',
  groupKey,
  takeaway,
  suitabilityReason,
  height = 360,
  totalDatasetRecords
}: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [observationStats, setObservationStats] = useState<{ valid: number; total: number } | null>(null);
  const [zeroWarning, setZeroWarning] = useState<string | null>(null);
  const plotlyInstanceRef = useRef<any>(null);

  useEffect(() => {
    let isMounted = true;

    async function renderChart() {
      if (!containerRef.current || !Array.isArray(data) || data.length === 0) {
        setLoading(false);
        setObservationStats(null);
        setZeroWarning(null);
        return;
      }

      const effectiveTotal = (typeof totalDatasetRecords === 'number' && totalDatasetRecords > 0) ? totalDatasetRecords : data.length;

      try {
        setLoading(true);
        setError(null);
        setZeroWarning(null);

        // Dynamically load Plotly on client-side only
        const Plotly = (await import('plotly.js-dist-min')).default;
        plotlyInstanceRef.current = Plotly;

        if (!isMounted || !containerRef.current) return;

        const typeUpper = (chartType || 'BAR').toUpperCase();
        let traces: any[] = [];

        // Build traces based on chart type
        if (typeUpper === 'HISTOGRAM') {
          const xVals = data.map(d => toNumericValue(d[xKey] ?? d.value ?? d.y)).filter((v): v is number => v !== null);
          setObservationStats({ valid: xVals.length, total: effectiveTotal });
          traces = [{
            type: 'histogram',
            x: xVals,
            marker: {
              color: '#3b82f6',
              line: { color: '#1e40af', width: 1 }
            },
            opacity: 0.85,
            name: xKey
          }];
        } else if (typeUpper === 'BOX_PLOT' || typeUpper === 'BOX') {
          const actualGroupKey = (groupKey && groupKey !== '') ? groupKey : (xKey && xKey !== yKey ? xKey : null);
          if (actualGroupKey && data.some(d => d[actualGroupKey] !== undefined && d[actualGroupKey] !== null)) {
            const rawCategories = data.map(d => d[actualGroupKey]).filter(v => v !== null && v !== undefined && String(v).trim() !== '' && String(v).trim().toLowerCase() !== 'nan');
            const groups = Array.from(new Set(rawCategories.map(String)));
            let totalValid = 0;
            traces = groups.slice(0, 10).map((grp) => {
              const grpVals = data.filter(d => String(d[actualGroupKey] ?? '').trim() === grp)
                .map(d => toNumericValue(d[yKey] ?? d.value))
                .filter((v): v is number => v !== null);
              totalValid += grpVals.length;
              return {
                type: 'box',
                y: grpVals,
                name: grp,
                x0: grp,
                boxpoints: 'outliers',
                marker: { size: 4 }
              };
            });
            setObservationStats({ valid: totalValid, total: effectiveTotal });
          } else {
            const yVals = data.map(d => toNumericValue(d[yKey] ?? d.value)).filter((v): v is number => v !== null);
            setObservationStats({ valid: yVals.length, total: effectiveTotal });
            traces = [{
              type: 'box',
              y: yVals,
              name: yKey,
              boxpoints: 'outliers',
              marker: { color: '#8b5cf6', size: 4 }
            }];
          }
        } else if (typeUpper === 'VIOLIN') {
          const actualGroupKey = (groupKey && groupKey !== '') ? groupKey : (xKey && xKey !== yKey ? xKey : null);
          if (actualGroupKey && data.some(d => d[actualGroupKey] !== undefined && d[actualGroupKey] !== null)) {
            const rawCategories = data.map(d => d[actualGroupKey]).filter(v => v !== null && v !== undefined && String(v).trim() !== '' && String(v).trim().toLowerCase() !== 'nan');
            const groups = Array.from(new Set(rawCategories.map(String)));
            if (groups.length >= 2) {
              const palette = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#6366f1', '#14b8a6'];
              let totalValid = 0;
              traces = groups.slice(0, 10).map((grp, idx) => {
                const grpRows = data.filter(d => String(d[actualGroupKey] ?? '').trim() === grp);
                const grpVals = grpRows.map(d => toNumericValue(d[yKey] ?? d.value)).filter((v): v is number => v !== null);
                totalValid += grpVals.length;
                const col = palette[idx % palette.length];
                return {
                  type: 'violin',
                  y: grpVals,
                  name: grp,
                  x0: grp,
                  box: { visible: true, width: 0.2 },
                  meanline: { visible: true },
                  points: 'outliers',
                  marker: { size: 4, color: col },
                  line: { color: col, width: 1.5 }
                };
              });
              setObservationStats({ valid: totalValid, total: effectiveTotal });
            } else {
              const yVals = data.map(d => toNumericValue(d[yKey] ?? d.value)).filter((v): v is number => v !== null);
              setObservationStats({ valid: yVals.length, total: effectiveTotal });
              traces = [{
                type: 'violin',
                y: yVals,
                name: yKey,
                box: { visible: true, width: 0.2 },
                meanline: { visible: true },
                points: 'outliers',
                fillcolor: 'rgba(139, 92, 246, 0.4)',
                line: { color: '#8b5cf6', width: 1.5 },
                marker: { color: '#c084fc', size: 4 }
              }];
            }
          } else {
            const yVals = data.map(d => toNumericValue(d[yKey] ?? d.value)).filter((v): v is number => v !== null);
            setObservationStats({ valid: yVals.length, total: effectiveTotal });
            traces = [{
              type: 'violin',
              y: yVals,
              name: yKey,
              box: { visible: true, width: 0.2 },
              meanline: { visible: true },
              points: 'outliers',
              fillcolor: 'rgba(139, 92, 246, 0.4)',
              line: { color: '#8b5cf6', width: 1.5 },
              marker: { color: '#c084fc', size: 4 }
            }];
          }
        } else if (typeUpper === 'SCATTER') {
          const rawXSample = data.slice(0, 10).map(d => d[xKey] !== undefined ? d[xKey] : d.x);
          const rawYSample = data.slice(0, 10).map(d => d[yKey] !== undefined ? d[yKey] : d.y);

          const allYRaw = data.map(d => d[yKey] !== undefined ? d[yKey] : d.y);
          const allXRaw = data.map(d => d[xKey] !== undefined ? d[xKey] : d.x);

          const allYNumeric = allYRaw.map(toNumericValue);
          const allXNumeric = allXRaw.map(toNumericValue);

          const nullCount = allYRaw.filter(v => v === null || v === undefined).length;
          const nanCount = allYRaw.filter(v => {
            if (v === null || v === undefined) return false;
            if (typeof v === 'number') return isNaN(v);
            if (typeof v === 'string') {
              const clean = v.trim().replace(/[$€£¥₹,\s]/g, '');
              return clean === '' || isNaN(Number(clean));
            }
            return true;
          }).length;
          const zeroCount = allYNumeric.filter(v => v === 0).length;

          const nonNullY = allYNumeric.filter((v): v is number => v !== null);
          const yMin = nonNullY.length > 0 ? Math.min(...nonNullY) : null;
          const yMax = nonNullY.length > 0 ? Math.max(...nonNullY) : null;

          const debugLog = {
            selectedXColumn: xKey,
            selectedYColumn: yKey,
            xSample: rawXSample,
            ySample: rawYSample,
            xType: data.length > 0 ? typeof (data[0]?.[xKey] ?? data[0]?.x) : 'undefined',
            yType: data.length > 0 ? typeof (data[0]?.[yKey] ?? data[0]?.y) : 'undefined',
            nullCount,
            nanCount,
            zeroCount,
            min: yMin,
            max: yMax
          };
          console.log('[VisualizationsStudio Debug]', debugLog);

          const paired = data.map((d, idx) => {
            const rawX = d[xKey] !== undefined ? d[xKey] : d.x;
            const rawY = d[yKey] !== undefined ? d[yKey] : d.y;
            const numX = toNumericValue(rawX);
            const numY = toNumericValue(rawY);
            const grp = groupKey && d[groupKey] !== undefined ? String(d[groupKey]) : 'All Records';
            return {
              numX,
              numY,
              grp,
              idx,
              isValid: numX !== null && numY !== null
            };
          });

          const validPairs = paired.filter(p => p.isValid);
          const validCount = validPairs.length;
          const totalCount = effectiveTotal;

          setObservationStats({ valid: validCount, total: totalCount });

          if (validCount > 0) {
            const validZerosY = validPairs.filter(p => p.numY === 0).length;
            const zeroRatio = validZerosY / validCount;
            if (zeroRatio > 0.9) {
              const pct = (zeroRatio * 100).toFixed(1);
              const warn = `Warning: Selected Y variable "${yKey}" contains ${pct}% zero values (${validZerosY}/${validCount}). Data may be zero-dominated or invalid.`;
              setZeroWarning(warn);
              console.warn(`[VisualizationsStudio Warning] ${warn}`);
            } else {
              setZeroWarning(null);
            }
          } else {
            setZeroWarning(null);
            setError(`No valid numeric observations found for X: "${xKey}" and Y: "${yKey}". (Total rows: ${totalCount})`);
            setLoading(false);
            return;
          }

          if (groupKey && validPairs.some(p => p.grp !== 'All Records')) {
            const groups = Array.from(new Set(validPairs.map(p => p.grp)));
            traces = groups.slice(0, 10).map((grp) => {
              const grpPairs = validPairs.filter(p => p.grp === grp);
              return {
                type: 'scatter',
                mode: 'markers',
                x: grpPairs.map(p => p.numX),
                y: grpPairs.map(p => p.numY),
                name: grp,
                marker: { size: 6, opacity: 0.85 }
              };
            });
          } else {
            traces = [{
              type: 'scatter',
              mode: 'markers',
              x: validPairs.map(p => p.numX),
              y: validPairs.map(p => p.numY),
              marker: {
                color: '#06b6d4',
                size: 6,
                opacity: 0.85,
                line: { color: '#0891b2', width: 1 }
              },
              name: `${xKey} vs ${yKey}`
            }];
          }
        } else if (typeUpper === 'PCA_3D') {
          const xVals = data.map(d => Number(d.PC1 ?? d[xKey] ?? 0));
          const yVals = data.map(d => Number(d.PC2 ?? d[yKey] ?? 0));
          const zVals = data.map(d => Number(d.PC3 ?? d[zKey] ?? 0));
          traces = [{
            type: 'scatter3d',
            mode: 'markers',
            x: xVals,
            y: yVals,
            z: zVals,
            marker: {
              size: 4,
              color: xVals,
              colorscale: 'Viridis',
              opacity: 0.85
            },
            name: 'PCA 3D Space'
          }];
        } else if (typeUpper === 'LINE') {
          const xVals = data.map(d => String(d[xKey] ?? d.period ?? d.date ?? ''));
          const yVals = data.map(d => Number(d[yKey] ?? d.value ?? 0));
          traces = [{
            type: 'scatter',
            mode: 'lines+markers',
            x: xVals,
            y: yVals,
            line: { color: '#3b82f6', width: 2.5, shape: 'spline' },
            marker: { color: '#60a5fa', size: 5 },
            name: yKey
          }];
        } else if (typeUpper === 'HEATMAP') {
          const xCats = Array.from(new Set(data.map(d => String(d.x || d.var1 || '')))).filter(Boolean);
          const yCats = Array.from(new Set(data.map(d => String(d.y || d.var2 || '')))).filter(Boolean);
          const zMatrix = yCats.map(yc => {
            return xCats.map(xc => {
              const match = data.find(d => String(d.x || d.var1) === xc && String(d.y || d.var2) === yc);
              return match ? Number(match.value ?? match.val ?? match.pearson ?? 0) : 0;
            });
          });
          traces = [{
            type: 'heatmap',
            x: xCats,
            y: yCats,
            z: zMatrix,
            colorscale: [
              [0, '#ef4444'],
              [0.5, '#1e293b'],
              [1, '#10b981']
            ],
            zmin: -1,
            zmax: 1
          }];
        } else if (typeUpper === 'HORIZONTAL_BAR') {
          const yCats = data.map(d => String(d[xKey] ?? d.category ?? d.Location ?? d[yKey] ?? ''));
          const xNums = data.map(d => Number(d[yKey] ?? d.value ?? d.count ?? d[xKey] ?? 0));
          const isFatal = (title || '').toLowerCase().includes('fatal') || (subtitle || '').toLowerCase().includes('fatal');
          traces = [{
            type: 'bar',
            orientation: 'h',
            y: yCats,
            x: xNums,
            text: xNums.map(v => String(v)),
            textposition: 'auto',
            textfont: { color: '#ffffff', size: 12, weight: 'bold' },
            marker: {
              color: isFatal ? '#f43f5e' : '#3b82f6',
              line: { color: isFatal ? '#be123c' : '#1d4ed8', width: 1 }
            },
            name: yKey
          }];
        } else {
          // Standard Bar
          const xVals = data.map(d => String(d[xKey] ?? d.category ?? d.Location ?? ''));
          const yVals = data.map(d => Number(d[yKey] ?? d.value ?? d.count ?? 0));
          const isFatal = (title || '').toLowerCase().includes('fatal') || (subtitle || '').toLowerCase().includes('fatal');
          traces = [{
            type: 'bar',
            x: xVals,
            y: yVals,
            text: yVals.map(v => String(v)),
            textposition: 'auto',
            textfont: { color: '#ffffff', size: 12, weight: 'bold' },
            marker: {
              color: isFatal ? '#f43f5e' : '#10b981',
              line: { color: isFatal ? '#be123c' : '#047857', width: 1 }
            },
            name: yKey
          }];
        }

        const layout: any = {
          margin: { l: 50, r: 20, t: 30, b: 50 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: {
            color: '#94a3b8',
            family: 'ui-sans-serif, system-ui, sans-serif',
            size: 11
          },
          xaxis: {
            gridcolor: '#1e293b',
            linecolor: '#334155',
            tickcolor: '#334155',
            title: xKey
          },
          yaxis: {
            gridcolor: '#1e293b',
            linecolor: '#334155',
            tickcolor: '#334155',
            title: yKey
          },
          showlegend: traces.length > 1,
          legend: {
            font: { color: '#cbd5e1', size: 10 },
            orientation: 'h',
            y: -0.2
          }
        };

        if (typeUpper === 'PCA_3D') {
          layout.scene = {
            xaxis: { title: 'PC1', gridcolor: '#1e293b', backgroundcolor: 'rgba(0,0,0,0)' },
            yaxis: { title: 'PC2', gridcolor: '#1e293b', backgroundcolor: 'rgba(0,0,0,0)' },
            zaxis: { title: 'PC3', gridcolor: '#1e293b', backgroundcolor: 'rgba(0,0,0,0)' }
          };
        } else if (typeUpper === 'VIOLIN') {
          layout.violinmode = 'group';
          layout.violingap = 0.2;
        } else if (typeUpper === 'BOX_PLOT' || typeUpper === 'BOX') {
          layout.boxmode = 'group';
          layout.boxgap = 0.2;
        }

        const config: any = {
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        await Plotly.react(containerRef.current, traces, layout, config);
        setLoading(false);
      } catch (err: any) {
        if (isMounted) {
          console.error('[Plotly Render Error]', err);
          setError(err?.message || 'Failed to render Plotly chart');
          setLoading(false);
        }
      }
    }

    renderChart();

    return () => {
      isMounted = false;
      if (containerRef.current && plotlyInstanceRef.current) {
        try {
          plotlyInstanceRef.current.purge(containerRef.current);
        } catch (_) {}
      }
    };
  }, [chartType, data, xKey, yKey, zKey, groupKey]);

  async function handleDownloadPNG() {
    if (!containerRef.current || !plotlyInstanceRef.current) return;
    try {
      const filename = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_chart.png`;
      await plotlyInstanceRef.current.downloadImage(containerRef.current, {
        format: 'png',
        filename: filename,
        width: 1000,
        height: 600,
        scale: 2
      });
    } catch (e) {
      console.error('Download failed:', e);
    }
  }

  function handleDownloadHTML() {
    if (!data || data.length === 0) return;
    const cleanTitle = title.replace(/"/g, '&quot;');
    const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${cleanTitle} - Interactive Chart</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { background-color: #0b1120; color: #f8fafc; font-family: system-ui, sans-serif; margin: 0; padding: 24px; }
    h2 { margin-top: 0; color: #60a5fa; }
    #chart { width: 100%; height: 75vh; }
    .meta { font-size: 13px; color: #94a3b8; margin-top: 12px; }
  </style>
</head>
<body>
  <h2>${cleanTitle}</h2>
  ${subtitle ? `<p style="color:#94a3b8;">${subtitle}</p>` : ''}
  <div id="chart"></div>
  ${takeaway ? `<div class="meta"><strong>Key Takeaway:</strong> ${takeaway}</div>` : ''}
  <script>
    var payload = ${JSON.stringify(data)};
    var trace = {
      type: '${chartType.toLowerCase().includes('scatter') ? 'scatter' : chartType.toLowerCase().includes('hist') ? 'histogram' : 'bar'}',
      x: payload.map(function(d){ return d['${xKey}'] || d.x || d.category; }),
      y: payload.map(function(d){ return d['${yKey}'] || d.y || d.value; }),
      marker: { color: '#3b82f6' }
    };
    var layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#94a3b8' },
      xaxis: { title: '${xKey}', gridcolor: '#1e293b' },
      yaxis: { title: '${yKey}', gridcolor: '#1e293b' }
    };
    Plotly.newPlot('chart', [trace], layout, { responsive: true });
  </script>
</body>
</html>`;

    const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_interactive.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  return (
    <div className="p-6 rounded-2xl glass-card border border-slate-800 space-y-4 flex flex-col justify-between">
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="font-bold text-sm text-slate-100">{title}</h3>
              <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-bold border border-cyan-500/20 font-mono">
                PLOTLY INTERACTIVE
              </span>
              {observationStats && (
                <span className="text-[10px] px-2 py-0.5 rounded bg-slate-900 text-slate-300 border border-slate-800 font-mono">
                  Observations: <strong className="text-cyan-400">{observationStats.valid}</strong> / {observationStats.total}
                </span>
              )}
            </div>
            {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
          </div>

          <div className="flex items-center gap-1.5 self-end sm:self-auto">
            <button
              onClick={handleDownloadPNG}
              title="Download High-Res PNG"
              className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[11px] font-semibold text-slate-300 hover:text-white flex items-center gap-1 transition-colors"
            >
              <Download className="w-3 h-3 text-cyan-400" />
              PNG
            </button>
            <button
              onClick={handleDownloadHTML}
              title="Export Standalone Interactive HTML"
              className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[11px] font-semibold text-slate-300 hover:text-white flex items-center gap-1 transition-colors"
            >
              <Code2 className="w-3 h-3 text-purple-400" />
              HTML
            </button>
          </div>
        </div>

        {zeroWarning && (
          <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>{zeroWarning}</span>
          </div>
        )}

        <div className="w-full relative" style={{ height: `${height}px` }}>
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-950/40 rounded-xl z-10">
              <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                Rendering interactive Plotly canvas...
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center p-4 text-center bg-rose-950/20 rounded-xl border border-rose-900/40">
              <p className="text-xs text-rose-300">{error}</p>
            </div>
          )}

          <div ref={containerRef} className="w-full h-full" />
        </div>
      </div>

      {(takeaway || suitabilityReason) && (
        <div className="pt-3 border-t border-slate-800/80 space-y-1.5 text-xs">
          {takeaway && (
            <div className="flex items-start gap-2 bg-slate-950/70 p-2.5 rounded-lg border border-slate-800/80">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400 mt-0.5 shrink-0" />
              <p className="text-slate-200 leading-relaxed"><strong className="text-cyan-300">Analytical Takeaway:</strong> {takeaway}</p>
            </div>
          )}
          {suitabilityReason && (
            <p className="text-[11px] text-slate-400 italic px-1">
              <span className="text-slate-500 font-semibold not-italic">Visualization Suitability: </span>
              {suitabilityReason}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
