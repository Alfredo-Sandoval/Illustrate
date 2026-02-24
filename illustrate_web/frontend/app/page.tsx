'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { AxisIndicator } from '../components/AxisIndicator';
import { OutlinePanel } from '../components/OutlinePanel';
import { PdbUploader } from '../components/PdbUploader';
import { PresetPicker } from '../components/PresetPicker';
import { RenderView } from '../components/RenderView';
import { RuleEditor } from '../components/RuleEditor';
import { TransformPanel } from '../components/TransformPanel';
import { WorldPanel } from '../components/WorldPanel';
import { fetchPresets, renderImage } from '../lib/api';
import type { PresetPayload, RenderPayload, RulePayload } from '../lib/types';

type RgbTriplet = [number, number, number];

type WorkspaceSnapshot = {
  version: number;
  savedAt: number;
  rulesOpen: boolean;
  transformOpen: boolean;
  worldOpen: boolean;
  outlinesOpen: boolean;
  autoRender: boolean;
  pdbId: string | null;
  activePreset: string;
  rules: RulePayload[];
  renderSizeMode: 'auto' | 'custom';
  renderWidth: number;
  renderHeight: number;
  scale: number;
  xrot: number;
  yrot: number;
  zrot: number;
  xtran: number;
  ytran: number;
  ztran: number;
  background: RgbTriplet;
  fogColor: RgbTriplet;
  fogFront: number;
  fogBack: number;
  shadows: boolean;
  shadowStrength: number;
  shadowAngle: number;
  shadowMinZ: number;
  shadowMaxDark: number;
  outlineEnabled: boolean;
  kernel: number;
  contourLow: number;
  contourHigh: number;
  zDiffMin: number;
  zDiffMax: number;
  subunitLow: number;
  subunitHigh: number;
  residueLow: number;
  residueHigh: number;
  residueDiff: number;
};

const WORKSPACE_STORAGE_KEY = 'illustrate_web_workspace_v1';
const WORKSPACE_STORAGE_VERSION = 1;

const defaultRule: RulePayload = {
  record_name: 'ATOM  ',
  descriptor: '----------',
  res_low: 0,
  res_high: 9999,
  color: [1.0, 0.7, 0.5],
  radius: 1.5,
};

function clamp(value: number, minValue: number, maxValue: number): number {
  return Math.min(Math.max(value, minValue), maxValue);
}

function asNumber(value: unknown, fallback: number): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function asText(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value : fallback;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function normalizeDimension(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(2, Math.min(4000, Math.round(value)));
}

function asRgb(value: [number, number, number] | number[] | undefined, fallback: RgbTriplet = [1, 1, 1]): RgbTriplet {
  if (!value || value.length < 3) {
    return fallback;
  }
  return [
    clamp(Number(value[0] ?? fallback[0]), 0, 1),
    clamp(Number(value[1] ?? fallback[1]), 0, 1),
    clamp(Number(value[2] ?? fallback[2]), 0, 1),
  ];
}

function asRgbUnknown(value: unknown, fallback: RgbTriplet = [1, 1, 1]): RgbTriplet {
  if (!Array.isArray(value) || value.length < 3) {
    return fallback;
  }
  return asRgb([asNumber(value[0], fallback[0]), asNumber(value[1], fallback[1]), asNumber(value[2], fallback[2])], fallback);
}

function sanitizeRule(rule: RulePayload): RulePayload {
  const descriptorRaw = (rule.descriptor || '').toUpperCase();
  const descriptor = `${descriptorRaw}----------`.slice(0, 10);
  return {
    record_name: rule.record_name === 'HETATM' ? 'HETATM' : 'ATOM  ',
    descriptor,
    res_low: Number.isFinite(rule.res_low) ? Math.max(0, Math.floor(rule.res_low)) : 0,
    res_high: Number.isFinite(rule.res_high) ? Math.max(0, Math.floor(rule.res_high)) : 9999,
    color: asRgb(rule.color, [1.0, 1.0, 1.0]),
    radius: Number.isFinite(rule.radius) ? Math.max(0, rule.radius) : 1.5,
  };
}

function sanitizeRulesUnknown(value: unknown): RulePayload[] {
  if (!Array.isArray(value) || value.length === 0) {
    return [{ ...defaultRule }];
  }
  const sanitized = value
    .map((entry) => asRecord(entry))
    .filter((entry): entry is Record<string, unknown> => entry !== null)
    .map((entry) => {
      const colorRaw = Array.isArray(entry.color) ? entry.color : [];
      const color: [number, number, number] = [
        asNumber(colorRaw[0], 1.0),
        asNumber(colorRaw[1], 1.0),
        asNumber(colorRaw[2], 1.0),
      ];
      return sanitizeRule({
        record_name: asText(entry.record_name, 'ATOM  '),
        descriptor: asText(entry.descriptor, '----------'),
        res_low: asNumber(entry.res_low, 0),
        res_high: asNumber(entry.res_high, 9999),
        color,
        radius: asNumber(entry.radius, 1.5),
      });
    });
  return sanitized.length > 0 ? sanitized : [{ ...defaultRule }];
}

function rotationForAxis(rotations: Array<[string, number]>, axis: string): number {
  const match = rotations.find((entry) => entry[0].toLowerCase() === axis.toLowerCase());
  return match ? match[1] : 0;
}

export default function HomePage() {
  const [rulesOpen, setRulesOpen] = useState(false);
  const [transformOpen, setTransformOpen] = useState(true);
  const [worldOpen, setWorldOpen] = useState(false);
  const [outlinesOpen, setOutlinesOpen] = useState(false);
  const [pdbId, setPdbId] = useState<string | null>(null);
  const [status, setStatus] = useState('Upload a PDB to start');
  const [rendering, setRendering] = useState(false);
  const [autoRender, setAutoRender] = useState(true);
  const [imageUrl, setImageUrl] = useState<string | undefined>();
  const [presets, setPresets] = useState<PresetPayload[]>([]);
  const [activePreset, setActivePreset] = useState('');
  const [rules, setRules] = useState<RulePayload[]>([{ ...defaultRule }]);
  const [renderSizeMode, setRenderSizeMode] = useState<'auto' | 'custom'>('auto');
  const [renderWidth, setRenderWidth] = useState(1200);
  const [renderHeight, setRenderHeight] = useState(900);
  const renderRequestIdRef = useRef(0);
  const renderAbortRef = useRef<AbortController | null>(null);

  const [scale, setScale] = useState(12);
  const [xrot, setXrot] = useState(0);
  const [yrot, setYrot] = useState(0);
  const [zrot, setZrot] = useState(90);
  const [xtran, setXtran] = useState(0);
  const [ytran, setYtran] = useState(0);
  const [ztran, setZtran] = useState(0);

  const [background, setBackground] = useState<RgbTriplet>([1.0, 1.0, 1.0]);
  const [fogColor, setFogColor] = useState<RgbTriplet>([1.0, 1.0, 1.0]);
  const [fogFront, setFogFront] = useState(1.0);
  const [fogBack, setFogBack] = useState(1.0);
  const [shadows, setShadows] = useState(true);
  const [shadowStrength, setShadowStrength] = useState(0.0023);
  const [shadowAngle, setShadowAngle] = useState(2.0);
  const [shadowMinZ, setShadowMinZ] = useState(1.0);
  const [shadowMaxDark, setShadowMaxDark] = useState(0.2);

  const [outlineEnabled, setOutlineEnabled] = useState(true);
  const [kernel, setKernel] = useState(4);
  const [contourLow, setContourLow] = useState(3);
  const [contourHigh, setContourHigh] = useState(10);
  const [zDiffMin, setZDiffMin] = useState(0);
  const [zDiffMax, setZDiffMax] = useState(5);
  const [subunitLow, setSubunitLow] = useState(3);
  const [subunitHigh, setSubunitHigh] = useState(10);
  const [residueLow, setResidueLow] = useState(3);
  const [residueHigh, setResidueHigh] = useState(8);
  const [residueDiff, setResidueDiff] = useState(6000);
  const [workspaceHydrated, setWorkspaceHydrated] = useState(false);
  const [workspaceRestored, setWorkspaceRestored] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(WORKSPACE_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsedUnknown = JSON.parse(raw);
      const parsed = asRecord(parsedUnknown);
      if (!parsed) {
        return;
      }
      const version = asNumber(parsed.version, 0);
      if (version !== WORKSPACE_STORAGE_VERSION) {
        return;
      }

      setWorkspaceRestored(true);
      setRulesOpen(asBoolean(parsed.rulesOpen, false));
      setTransformOpen(asBoolean(parsed.transformOpen, true));
      setWorldOpen(asBoolean(parsed.worldOpen, false));
      setOutlinesOpen(asBoolean(parsed.outlinesOpen, false));
      setAutoRender(asBoolean(parsed.autoRender, true));

      const restoredPdbId = asText(parsed.pdbId, '').trim();
      setPdbId(restoredPdbId.length > 0 ? restoredPdbId : null);
      setActivePreset(asText(parsed.activePreset, ''));
      setRules(sanitizeRulesUnknown(parsed.rules));

      const restoredMode = parsed.renderSizeMode === 'custom' ? 'custom' : 'auto';
      setRenderSizeMode(restoredMode);
      setRenderWidth(normalizeDimension(asNumber(parsed.renderWidth, 1200), 1200));
      setRenderHeight(normalizeDimension(asNumber(parsed.renderHeight, 900), 900));

      setScale(asNumber(parsed.scale, 12));
      setXrot(asNumber(parsed.xrot, 0));
      setYrot(asNumber(parsed.yrot, 0));
      setZrot(asNumber(parsed.zrot, 90));
      setXtran(asNumber(parsed.xtran, 0));
      setYtran(asNumber(parsed.ytran, 0));
      setZtran(asNumber(parsed.ztran, 0));

      setBackground(asRgbUnknown(parsed.background, [1.0, 1.0, 1.0]));
      setFogColor(asRgbUnknown(parsed.fogColor, [1.0, 1.0, 1.0]));
      setFogFront(asNumber(parsed.fogFront, 1.0));
      setFogBack(asNumber(parsed.fogBack, 1.0));
      setShadows(asBoolean(parsed.shadows, true));
      setShadowStrength(asNumber(parsed.shadowStrength, 0.0023));
      setShadowAngle(asNumber(parsed.shadowAngle, 2.0));
      setShadowMinZ(asNumber(parsed.shadowMinZ, 1.0));
      setShadowMaxDark(asNumber(parsed.shadowMaxDark, 0.2));

      setOutlineEnabled(asBoolean(parsed.outlineEnabled, true));
      setKernel(asNumber(parsed.kernel, 4));
      setContourLow(asNumber(parsed.contourLow, 3));
      setContourHigh(asNumber(parsed.contourHigh, 10));
      setZDiffMin(asNumber(parsed.zDiffMin, 0));
      setZDiffMax(asNumber(parsed.zDiffMax, 5));
      setSubunitLow(asNumber(parsed.subunitLow, 3));
      setSubunitHigh(asNumber(parsed.subunitHigh, 10));
      setResidueLow(asNumber(parsed.residueLow, 3));
      setResidueHigh(asNumber(parsed.residueHigh, 8));
      setResidueDiff(asNumber(parsed.residueDiff, 6000));
      setStatus('Restored previous session');
    } catch {
      // Ignore malformed browser state and continue with defaults.
    } finally {
      setWorkspaceHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!workspaceHydrated) {
      return;
    }
    void (async () => {
      try {
        const next = await fetchPresets();
        setPresets(next);
        if (!workspaceRestored && next.length > 0) {
          const firstPreset = next[0];
          setRules(firstPreset.rules.map(sanitizeRule));
          setActivePreset(firstPreset.name);
        }
        if (!workspaceRestored) {
          setStatus('Ready');
        }
      } catch (error) {
        setStatus((error as Error).message);
      }
    })();
  }, [workspaceHydrated, workspaceRestored]);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  useEffect(() => {
    return () => {
      if (renderAbortRef.current !== null) {
        renderAbortRef.current.abort();
        renderAbortRef.current = null;
      }
    };
  }, []);

  const buildPayload = useCallback(
    (outputFormat: 'png' | 'ppm' = 'png'): RenderPayload => {
      if (!pdbId) {
        throw new Error('No PDB loaded');
      }

      return {
        pdb_id: pdbId,
        rules: rules.map(sanitizeRule),
        transform: {
          scale,
          translate: [xtran, ytran, ztran],
          rotations: [
            ['z', zrot],
            ['y', yrot],
            ['x', xrot],
          ],
          autocenter: 'auto',
        },
        world: {
          background,
          fog_color: fogColor,
          fog_front: fogFront,
          fog_back: fogBack,
          shadows,
          shadow_strength: shadowStrength,
          shadow_angle: shadowAngle,
          shadow_min_z: shadowMinZ,
          shadow_max_dark: shadowMaxDark,
          width: renderSizeMode === 'custom' ? normalizeDimension(renderWidth, 1200) : -30,
          height: renderSizeMode === 'custom' ? normalizeDimension(renderHeight, 900) : -30,
        },
        outlines: {
          enabled: outlineEnabled,
          contour_low: contourLow,
          contour_high: contourHigh,
          kernel,
          z_diff_min: zDiffMin,
          z_diff_max: zDiffMax,
          subunit_low: subunitLow,
          subunit_high: subunitHigh,
          residue_low: residueLow,
          residue_high: residueHigh,
          residue_diff: residueDiff,
        },
        output_format: outputFormat,
      };
    },
    [
      background,
      contourHigh,
      contourLow,
      fogBack,
      fogColor,
      fogFront,
      kernel,
      outlineEnabled,
      pdbId,
      residueDiff,
      renderHeight,
      renderSizeMode,
      renderWidth,
      residueHigh,
      residueLow,
      rules,
      scale,
      shadowAngle,
      shadowMaxDark,
      shadowMinZ,
      shadowStrength,
      shadows,
      subunitHigh,
      subunitLow,
      xrot,
      xtran,
      yrot,
      ytran,
      zDiffMax,
      zDiffMin,
      zrot,
      ztran,
    ],
  );

  const requestRender = useCallback(
    async (outputFormat: 'png' | 'ppm' = 'png', signal?: AbortSignal): Promise<Blob> => {
      const payload = buildPayload(outputFormat);
      return await renderImage(payload, { signal });
    },
    [buildPayload],
  );

  const showImage = useCallback(
    async (outputFormat: 'png' | 'ppm' = 'png'): Promise<void> => {
      if (!pdbId) {
        setStatus('Upload a PDB first');
        return;
      }
      const requestId = renderRequestIdRef.current + 1;
      renderRequestIdRef.current = requestId;
      if (renderAbortRef.current !== null) {
        renderAbortRef.current.abort();
      }
      const controller = new AbortController();
      renderAbortRef.current = controller;

      setRendering(true);
      setStatus('Rendering…');
      try {
        const blob = await requestRender(outputFormat, controller.signal);
        if (requestId !== renderRequestIdRef.current) {
          return;
        }
        const nextUrl = URL.createObjectURL(blob);
        setImageUrl((previous) => {
          if (previous) {
            URL.revokeObjectURL(previous);
          }
          return nextUrl;
        });
        setStatus(outputFormat === 'ppm' ? 'PPM render complete' : 'Render complete');
      } catch (error) {
        if ((error as Error).name === 'AbortError') {
          return;
        }
        if (requestId !== renderRequestIdRef.current) {
          return;
        }
        setStatus((error as Error).message);
      } finally {
        if (renderAbortRef.current === controller) {
          renderAbortRef.current = null;
        }
        if (requestId === renderRequestIdRef.current) {
          setRendering(false);
        }
      }
    },
    [pdbId, requestRender],
  );

  const downloadImage = useCallback(
    async (outputFormat: 'png' | 'ppm'): Promise<void> => {
      if (!pdbId) {
        setStatus('Upload a PDB first');
        return;
      }
      setStatus(`Preparing ${outputFormat.toUpperCase()} download`);
      try {
        const blob = await requestRender(outputFormat);
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = outputFormat === 'ppm' ? 'render.ppm' : 'render.png';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(link.href);
        setStatus('Download ready');
      } catch (error) {
        setStatus((error as Error).message);
      }
    },
    [pdbId, requestRender],
  );

  useEffect(() => {
    if (!autoRender || !pdbId) {
      return;
    }
    const timeout = window.setTimeout(() => {
      void showImage('png');
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [
    autoRender,
    pdbId,
    showImage,
    rules,
    scale,
    xrot,
    yrot,
    zrot,
    xtran,
    ytran,
    ztran,
    background,
    fogColor,
    fogFront,
    fogBack,
    shadows,
    shadowStrength,
    shadowAngle,
    shadowMinZ,
    shadowMaxDark,
    outlineEnabled,
    kernel,
    contourLow,
    contourHigh,
    zDiffMin,
    zDiffMax,
    subunitLow,
    subunitHigh,
    residueLow,
    residueHigh,
    residueDiff,
    renderHeight,
    renderSizeMode,
    renderWidth,
  ]);

  const handleFile = async (fileId: string): Promise<void> => {
    setPdbId(fileId);
    setStatus(`Loaded PDB ${fileId}`);
  };

  const applyPreset = (preset: PresetPayload) => {
    setActivePreset(preset.name);
    setRules(preset.rules.map(sanitizeRule));

    setScale(preset.transform.scale);
    setXrot(rotationForAxis(preset.transform.rotations, 'x'));
    setYrot(rotationForAxis(preset.transform.rotations, 'y'));
    setZrot(rotationForAxis(preset.transform.rotations, 'z'));
    setXtran(Number(preset.transform.translate[0] ?? 0));
    setYtran(Number(preset.transform.translate[1] ?? 0));
    setZtran(Number(preset.transform.translate[2] ?? 0));

    setBackground(asRgb(preset.world.background));
    setFogColor(asRgb(preset.world.fog_color));
    setFogFront(preset.world.fog_front);
    setFogBack(preset.world.fog_back);
    setShadows(preset.world.shadows);
    setShadowStrength(preset.world.shadow_strength);
    setShadowAngle(preset.world.shadow_angle);
    setShadowMinZ(preset.world.shadow_min_z);
    setShadowMaxDark(preset.world.shadow_max_dark);
    if (preset.world.width > 0 && preset.world.height > 0) {
      setRenderSizeMode('custom');
      setRenderWidth(normalizeDimension(preset.world.width, 1200));
      setRenderHeight(normalizeDimension(preset.world.height, 900));
    } else {
      setRenderSizeMode('auto');
    }

    setOutlineEnabled(preset.outlines.enabled);
    setKernel(preset.outlines.kernel);
    setContourLow(preset.outlines.contour_low);
    setContourHigh(preset.outlines.contour_high);
    setZDiffMin(preset.outlines.z_diff_min);
    setZDiffMax(preset.outlines.z_diff_max);
    setSubunitLow(preset.outlines.subunit_low);
    setSubunitHigh(preset.outlines.subunit_high);
    setResidueLow(preset.outlines.residue_low);
    setResidueHigh(preset.outlines.residue_high);
    setResidueDiff(preset.outlines.residue_diff);

    setStatus(`Applied preset: ${preset.name}`);
  };

  const applyDrag = (dx: number, dy: number) => {
    setXrot((current) => clamp(current + dy * 0.5, -180, 180));
    setYrot((current) => clamp(current + dx * 0.5, -180, 180));
  };

  const [loadingPdb, setLoadingPdb] = useState(false);

  const handleUpload = async (fileId: string): Promise<void> => {
    setLoadingPdb(true);
    try {
      await handleFile(fileId);
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setLoadingPdb(false);
    }
  };

  useEffect(() => {
    if (!workspaceHydrated) {
      return;
    }
    const snapshot: WorkspaceSnapshot = {
      version: WORKSPACE_STORAGE_VERSION,
      savedAt: Date.now(),
      rulesOpen,
      transformOpen,
      worldOpen,
      outlinesOpen,
      autoRender,
      pdbId,
      activePreset,
      rules: rules.map(sanitizeRule),
      renderSizeMode,
      renderWidth: normalizeDimension(renderWidth, 1200),
      renderHeight: normalizeDimension(renderHeight, 900),
      scale,
      xrot,
      yrot,
      zrot,
      xtran,
      ytran,
      ztran,
      background,
      fogColor,
      fogFront,
      fogBack,
      shadows,
      shadowStrength,
      shadowAngle,
      shadowMinZ,
      shadowMaxDark,
      outlineEnabled,
      kernel,
      contourLow,
      contourHigh,
      zDiffMin,
      zDiffMax,
      subunitLow,
      subunitHigh,
      residueLow,
      residueHigh,
      residueDiff,
    };
    const saveTimeout = window.setTimeout(() => {
      try {
        window.localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(snapshot));
      } catch {
        // Storage can fail in private mode or quota exhaustion.
      }
    }, 300);
    return () => window.clearTimeout(saveTimeout);
  }, [
    activePreset,
    autoRender,
    background,
    contourHigh,
    contourLow,
    fogBack,
    fogColor,
    fogFront,
    kernel,
    outlineEnabled,
    outlinesOpen,
    pdbId,
    renderHeight,
    renderSizeMode,
    renderWidth,
    residueDiff,
    residueHigh,
    residueLow,
    rules,
    rulesOpen,
    scale,
    shadowAngle,
    shadowMaxDark,
    shadowMinZ,
    shadowStrength,
    shadows,
    subunitHigh,
    subunitLow,
    transformOpen,
    worldOpen,
    workspaceHydrated,
    xrot,
    xtran,
    yrot,
    ytran,
    zDiffMax,
    zDiffMin,
    zrot,
    ztran,
  ]);

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1>Illustrate Web</h1>
          {status !== 'Ready' && <p className="status-line">{status}</p>}
        </div>
        {rendering && <span className="status-text">Rendering…</span>}
      </header>

      <section className="top-toolbar">
        <div className="toolbar-left">
          <PresetPicker presets={presets} value={activePreset} disabled={loadingPdb} onPick={applyPreset} compact />
          <PdbUploader disabled={loadingPdb} onUploaded={handleUpload} compact />
        </div>
        <section className="action-row toolbar-actions">
          <div className="render-size-row">
            <select
              className="form-input render-size-mode"
              value={renderSizeMode}
              onChange={(event) => setRenderSizeMode(event.target.value as 'auto' | 'custom')}
            >
              <option value="auto">Auto</option>
              <option value="custom">Custom</option>
            </select>
            {renderSizeMode === 'custom' && (
              <>
                <input
                  className="form-input render-size-input"
                  type="number"
                  min={2}
                  max={4000}
                  step={2}
                  value={renderWidth}
                  onChange={(event) => setRenderWidth(normalizeDimension(Number(event.target.value), renderWidth))}
                />
                <span className="render-size-sep">×</span>
                <input
                  className="form-input render-size-input"
                  type="number"
                  min={2}
                  max={4000}
                  step={2}
                  value={renderHeight}
                  onChange={(event) => setRenderHeight(normalizeDimension(Number(event.target.value), renderHeight))}
                />
              </>
            )}
          </div>
          <label className="auto-toggle">
            <input type="checkbox" checked={autoRender} onChange={(event) => setAutoRender(event.target.checked)} />
            <span>Auto-render</span>
          </label>
          <button className="action-button primary" type="button" disabled={!pdbId || rendering} onClick={() => void showImage('png')}>
            {rendering ? 'Rendering…' : 'Render'}
          </button>
          <button
            className="action-button secondary"
            type="button"
            disabled={!pdbId || rendering}
            onClick={() => void downloadImage('png')}
          >
            Download PNG
          </button>
          <button
            className="action-button secondary"
            type="button"
            disabled={!pdbId || rendering}
            onClick={() => void downloadImage('ppm')}
          >
            Download PPM
          </button>
        </section>
      </section>

      <section className="page-layout">
        <aside className="control-column">
          <div className="sidebar-section">
            <button type="button" className="sidebar-section-toggle" onClick={() => setTransformOpen((v) => !v)}>
              <span>Transform</span>
              <span className={`sidebar-chevron ${transformOpen ? 'open' : ''}`}>&#9654;</span>
            </button>
            {transformOpen && (
              <div className="sidebar-section-body">
                <TransformPanel
                  scale={scale}
                  xrot={xrot}
                  yrot={yrot}
                  zrot={zrot}
                  xtran={xtran}
                  ytran={ytran}
                  ztran={ztran}
                  onChange={(next) => {
                    setScale(next.scale);
                    setXrot(next.xrot);
                    setYrot(next.yrot);
                    setZrot(next.zrot);
                    setXtran(next.xtran);
                    setYtran(next.ytran);
                    setZtran(next.ztran);
                  }}
                />
              </div>
            )}
          </div>

          <div className="sidebar-section">
            <button type="button" className="sidebar-section-toggle" onClick={() => setWorldOpen((v) => !v)}>
              <span>World / Lighting</span>
              <span className={`sidebar-chevron ${worldOpen ? 'open' : ''}`}>&#9654;</span>
            </button>
            {worldOpen && (
              <div className="sidebar-section-body">
                <WorldPanel
                  background={background}
                  fogColor={fogColor}
                  fogFront={fogFront}
                  fogBack={fogBack}
                  shadows={shadows}
                  shadowStrength={shadowStrength}
                  shadowAngle={shadowAngle}
                  shadowMinZ={shadowMinZ}
                  shadowMaxDark={shadowMaxDark}
                  onChange={(next) => {
                    setBackground(next.background);
                    setFogColor(next.fogColor);
                    setFogFront(next.fogFront);
                    setFogBack(next.fogBack);
                    setShadows(next.shadows);
                    setShadowStrength(next.shadowStrength);
                    setShadowAngle(next.shadowAngle);
                    setShadowMinZ(next.shadowMinZ);
                    setShadowMaxDark(next.shadowMaxDark);
                  }}
                />
              </div>
            )}
          </div>

          <div className="sidebar-section">
            <button type="button" className="sidebar-section-toggle" onClick={() => setOutlinesOpen((v) => !v)}>
              <span>Outlines</span>
              <span className={`sidebar-chevron ${outlinesOpen ? 'open' : ''}`}>&#9654;</span>
            </button>
            {outlinesOpen && (
              <div className="sidebar-section-body">
                <OutlinePanel
                  enabled={outlineEnabled}
                  kernel={kernel}
                  contourLow={contourLow}
                  contourHigh={contourHigh}
                  zDiffMin={zDiffMin}
                  zDiffMax={zDiffMax}
                  subunitLow={subunitLow}
                  subunitHigh={subunitHigh}
                  residueLow={residueLow}
                  residueHigh={residueHigh}
                  residueDiff={residueDiff}
                  onChange={(next) => {
                    setOutlineEnabled(next.enabled);
                    setKernel(next.kernel);
                    setContourLow(next.contourLow);
                    setContourHigh(next.contourHigh);
                    setZDiffMin(next.zDiffMin);
                    setZDiffMax(next.zDiffMax);
                    setSubunitLow(next.subunitLow);
                    setSubunitHigh(next.subunitHigh);
                    setResidueLow(next.residueLow);
                    setResidueHigh(next.residueHigh);
                    setResidueDiff(next.residueDiff);
                  }}
                />
              </div>
            )}
          </div>
        </aside>

        <section className="preview-panel">
          <RenderView imageUrl={imageUrl} busy={rendering} status={status} onRotate={applyDrag} />
          <AxisIndicator xrot={xrot} yrot={yrot} zrot={zrot} />
        </section>
      </section>

      <section className={`bottom-panel ${rulesOpen ? 'open' : 'collapsed'}`}>
        <button type="button" className="bottom-panel-toggle" onClick={() => setRulesOpen((v) => !v)}>
          <span className="bottom-panel-toggle-label">Selection Rules</span>
          <span className={`bottom-panel-chevron ${rulesOpen ? 'open' : ''}`}>&#9650;</span>
        </button>
        {rulesOpen && (
          <RuleEditor
            rules={rules}
            onChange={(nextRules) => {
              setActivePreset('');
              setRules(nextRules);
            }}
          />
        )}
      </section>
    </main>
  );
}
