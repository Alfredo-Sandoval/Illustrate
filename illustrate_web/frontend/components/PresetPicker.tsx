import { FC, ChangeEvent } from 'react';
import type { PresetPayload } from '../lib/types';

type PresetPickerProps = {
  presets: PresetPayload[];
  onPick: (preset: PresetPayload) => void;
  value?: string;
  disabled?: boolean;
  compact?: boolean;
};

export const PresetPicker: FC<PresetPickerProps> = ({ presets, onPick, value, disabled, compact }) => {
  const handleChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const name = event.target.value;
    const preset = presets.find((item) => item.name === name);
    if (preset) {
      onPick(preset);
    }
  };

  return (
    <section className={compact ? 'toolbar-card' : 'utility-card'}>
      <h3 className="panel-title">{compact ? 'Preset' : 'Preset'}</h3>
      <select className="form-input" value={value} onChange={handleChange} disabled={disabled || presets.length === 0}>
        <option value="">Select preset</option>
        {presets.map((preset) => (
          <option value={preset.name} key={preset.name}>
            {preset.name}
          </option>
        ))}
      </select>
    </section>
  );
};
