import { ChangeEvent, FC } from 'react';

type RgbTriplet = [number, number, number];

type WorldPanelProps = {
  background: RgbTriplet;
  fogColor: RgbTriplet;
  fogFront: number;
  fogBack: number;
  shadows: boolean;
  shadowStrength: number;
  shadowAngle: number;
  shadowMinZ: number;
  shadowMaxDark: number;
  onChange: (next: {
    background: RgbTriplet;
    fogColor: RgbTriplet;
    fogFront: number;
    fogBack: number;
    shadows: boolean;
    shadowStrength: number;
    shadowAngle: number;
    shadowMinZ: number;
    shadowMaxDark: number;
  }) => void;
};

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function normalizeRgb(value: RgbTriplet | number): RgbTriplet {
  if (Array.isArray(value) && value.length >= 3) {
    return [clamp01(Number(value[0])), clamp01(Number(value[1])), clamp01(Number(value[2]))];
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    const grayscale = clamp01(value);
    return [grayscale, grayscale, grayscale];
  }
  return [1, 1, 1];
}

function rgbToHex(value: RgbTriplet | number): string {
  const [r, g, b] = normalizeRgb(value);
  const toHex = (value: number): string => {
    const normalized = Math.round(clamp01(value) * 255);
    return normalized.toString(16).padStart(2, '0');
  };
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function hexToRgb(value: string): RgbTriplet {
  const cleaned = value.replace('#', '');
  if (cleaned.length !== 6) {
    return [1, 1, 1];
  }
  const r = Number.parseInt(cleaned.slice(0, 2), 16) / 255;
  const g = Number.parseInt(cleaned.slice(2, 4), 16) / 255;
  const b = Number.parseInt(cleaned.slice(4, 6), 16) / 255;
  return [r, g, b];
}

export const WorldPanel: FC<WorldPanelProps> = ({
  background,
  fogColor,
  fogFront,
  fogBack,
  shadows,
  shadowStrength,
  shadowAngle,
  shadowMinZ,
  shadowMaxDark,
  onChange,
}) => {
  const update = (patch: Partial<WorldPanelProps>) => {
    onChange({
      background: (patch.background as RgbTriplet) ?? background,
      fogColor: (patch.fogColor as RgbTriplet) ?? fogColor,
      fogFront: patch.fogFront ?? fogFront,
      fogBack: patch.fogBack ?? fogBack,
      shadows: patch.shadows ?? shadows,
      shadowStrength: patch.shadowStrength ?? shadowStrength,
      shadowAngle: patch.shadowAngle ?? shadowAngle,
      shadowMinZ: patch.shadowMinZ ?? shadowMinZ,
      shadowMaxDark: patch.shadowMaxDark ?? shadowMaxDark,
    });
  };

  const handleCheckbox = (event: ChangeEvent<HTMLInputElement>) => {
    update({ shadows: event.target.checked });
  };

  return (
    <div className="field-stack">
      <label className="check-field">
        <input type="checkbox" checked={shadows} onChange={handleCheckbox} />
        <span>Enable shadows</span>
      </label>

      <label className="form-row">
        <span className="form-row-label">Background</span>
        <span className="form-row-color-wrap">
          <input type="color" value={rgbToHex(background)} onChange={(event) => update({ background: hexToRgb(event.target.value) })} />
          <span className="form-row-hex">{rgbToHex(background)}</span>
        </span>
      </label>

      <label className="form-row">
        <span className="form-row-label">Fog color</span>
        <span className="form-row-color-wrap">
          <input type="color" value={rgbToHex(fogColor)} onChange={(event) => update({ fogColor: hexToRgb(event.target.value) })} />
          <span className="form-row-hex">{rgbToHex(fogColor)}</span>
        </span>
      </label>

      <label className="form-row">
        <span className="form-row-label">Fog front</span>
        <input className="form-input" type="number" min={0} max={1} step={0.01} value={fogFront} onChange={(event) => update({ fogFront: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Fog back</span>
        <input className="form-input" type="number" min={0} max={1} step={0.01} value={fogBack} onChange={(event) => update({ fogBack: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Shadow strength</span>
        <input className="form-input" type="number" min={0} max={1} step={0.0001} value={shadowStrength} onChange={(event) => update({ shadowStrength: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Shadow angle</span>
        <input className="form-input" type="number" min={0} max={10} step={0.1} value={shadowAngle} onChange={(event) => update({ shadowAngle: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Shadow start Z</span>
        <input className="form-input" type="number" min={0} max={20} step={0.1} value={shadowMinZ} onChange={(event) => update({ shadowMinZ: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Max shadow dark</span>
        <input className="form-input" type="number" min={0} max={1} step={0.01} value={shadowMaxDark} onChange={(event) => update({ shadowMaxDark: Number(event.target.value) })} />
      </label>
    </div>
  );
};
