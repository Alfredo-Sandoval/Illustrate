import { ChangeEvent, FC } from 'react';

type TransformPanelProps = {
  scale: number;
  xrot: number;
  yrot: number;
  zrot: number;
  xtran: number;
  ytran: number;
  ztran: number;
  onChange: (next: {
    scale: number;
    xrot: number;
    yrot: number;
    zrot: number;
    xtran: number;
    ytran: number;
    ztran: number;
  }) => void;
};

export const TransformPanel: FC<TransformPanelProps> = ({
  scale,
  xrot,
  yrot,
  zrot,
  xtran,
  ytran,
  ztran,
  onChange,
}) => {
  const update = (key: keyof Omit<TransformPanelProps, 'onChange'>, value: number) => {
    onChange({
      scale: key === 'scale' ? value : scale,
      xrot: key === 'xrot' ? value : xrot,
      yrot: key === 'yrot' ? value : yrot,
      zrot: key === 'zrot' ? value : zrot,
      xtran: key === 'xtran' ? value : xtran,
      ytran: key === 'ytran' ? value : ytran,
      ztran: key === 'ztran' ? value : ztran,
    });
  };

  const handleNumber = (key: keyof Omit<TransformPanelProps, 'onChange'>) => (event: ChangeEvent<HTMLInputElement>) => {
    update(key, Number(event.target.value));
  };

  return (
    <div className="field-stack">
      <label className="form-row">
        <span className="form-row-label">Scale</span>
        <input className="form-input" type="number" min={1} max={50} step={0.1} value={scale} onChange={handleNumber('scale')} />
      </label>
      <label className="form-row">
        <span className="form-row-label">X rotation</span>
        <input className="form-input" type="number" min={-180} max={180} step={1} value={xrot} onChange={handleNumber('xrot')} />
      </label>
      <label className="form-row">
        <span className="form-row-label">Y rotation</span>
        <input className="form-input" type="number" min={-180} max={180} step={1} value={yrot} onChange={handleNumber('yrot')} />
      </label>
      <label className="form-row">
        <span className="form-row-label">Z rotation</span>
        <input className="form-input" type="number" min={-180} max={180} step={1} value={zrot} onChange={handleNumber('zrot')} />
      </label>
      <label className="form-row">
        <span className="form-row-label">X translation</span>
        <input className="form-input" type="number" min={-10000} max={10000} step={0.5} value={xtran} onChange={handleNumber('xtran')} />
      </label>
      <label className="form-row">
        <span className="form-row-label">Y translation</span>
        <input className="form-input" type="number" min={-10000} max={10000} step={0.5} value={ytran} onChange={handleNumber('ytran')} />
      </label>
      <label className="form-row">
        <span className="form-row-label">Z translation</span>
        <input className="form-input" type="number" min={-10000} max={10000} step={0.5} value={ztran} onChange={handleNumber('ztran')} />
      </label>
      <button
        className="action-button secondary"
        type="button"
        style={{ width: '100%' }}
        onClick={() =>
          onChange({
            scale: 12,
            xrot: 0,
            yrot: 0,
            zrot: 90,
            xtran: 0,
            ytran: 0,
            ztran: 0,
          })
        }
      >
        Reset
      </button>
    </div>
  );
};
