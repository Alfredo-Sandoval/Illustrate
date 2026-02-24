import { FC } from 'react';

type OutlinePanelProps = {
  enabled: boolean;
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
  onChange: (next: {
    enabled: boolean;
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
  }) => void;
};

export const OutlinePanel: FC<OutlinePanelProps> = ({
  enabled,
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
  onChange,
}) => {
  const update = (patch: Partial<OutlinePanelProps>) => {
    onChange({
      enabled: patch.enabled ?? enabled,
      kernel: patch.kernel ?? kernel,
      contourLow: patch.contourLow ?? contourLow,
      contourHigh: patch.contourHigh ?? contourHigh,
      zDiffMin: patch.zDiffMin ?? zDiffMin,
      zDiffMax: patch.zDiffMax ?? zDiffMax,
      subunitLow: patch.subunitLow ?? subunitLow,
      subunitHigh: patch.subunitHigh ?? subunitHigh,
      residueLow: patch.residueLow ?? residueLow,
      residueHigh: patch.residueHigh ?? residueHigh,
      residueDiff: patch.residueDiff ?? residueDiff,
    });
  };

  return (
    <div className="field-stack">
      <label className="check-field">
        <input type="checkbox" checked={enabled} onChange={(event) => update({ enabled: event.target.checked })} />
        <span>Enable outlines</span>
      </label>

      <label className="form-row">
        <span className="form-row-label">Kernel</span>
        <select className="form-input" value={kernel} onChange={(event) => update({ kernel: Number(event.target.value) })}>
          <option value={1}>1</option>
          <option value={2}>2</option>
          <option value={3}>3</option>
          <option value={4}>4</option>
        </select>
      </label>

      <label className="form-row">
        <span className="form-row-label">Contour low</span>
        <input className="form-input" type="number" min={0} max={30} step={0.5} value={contourLow} onChange={(event) => update({ contourLow: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Contour high</span>
        <input className="form-input" type="number" min={0} max={30} step={0.5} value={contourHigh} onChange={(event) => update({ contourHigh: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Z diff min</span>
        <input className="form-input" type="number" min={0} max={50} step={0.1} value={zDiffMin} onChange={(event) => update({ zDiffMin: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Z diff max</span>
        <input className="form-input" type="number" min={0} max={50} step={0.1} value={zDiffMax} onChange={(event) => update({ zDiffMax: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Subunit low</span>
        <input className="form-input" type="number" min={0} max={50} step={0.5} value={subunitLow} onChange={(event) => update({ subunitLow: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Subunit high</span>
        <input className="form-input" type="number" min={0} max={50} step={0.5} value={subunitHigh} onChange={(event) => update({ subunitHigh: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Residue low</span>
        <input className="form-input" type="number" min={0} max={50} step={0.5} value={residueLow} onChange={(event) => update({ residueLow: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Residue high</span>
        <input className="form-input" type="number" min={0} max={50} step={0.5} value={residueHigh} onChange={(event) => update({ residueHigh: Number(event.target.value) })} />
      </label>

      <label className="form-row">
        <span className="form-row-label">Residue diff</span>
        <input className="form-input" type="number" min={0} max={10000} step={1} value={residueDiff} onChange={(event) => update({ residueDiff: Number(event.target.value) })} />
      </label>
    </div>
  );
};
