import { FC } from 'react';
import type { RulePayload } from '../lib/types';

type RuleEditorProps = {
  rules: RulePayload[];
  onChange: (next: RulePayload[]) => void;
};

const newRule: RulePayload = {
  record_name: 'ATOM  ',
  descriptor: '----------',
  res_low: 0,
  res_high: 9999,
  color: [1.0, 0.7, 0.5],
  radius: 1.5,
};

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function rgbToHex([r, g, b]: [number, number, number]): string {
  const toHex = (value: number): string => Math.round(clamp01(value) * 255).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function hexToRgb(value: string): [number, number, number] {
  const cleaned = value.replace('#', '');
  if (cleaned.length !== 6) {
    return [1.0, 1.0, 1.0];
  }
  return [
    Number.parseInt(cleaned.slice(0, 2), 16) / 255,
    Number.parseInt(cleaned.slice(2, 4), 16) / 255,
    Number.parseInt(cleaned.slice(4, 6), 16) / 255,
  ];
}

export const RuleEditor: FC<RuleEditorProps> = ({ rules, onChange }) => {
  const updateRow = (index: number, patch: Partial<RulePayload>) => {
    onChange(
      rules.map((rule, ruleIndex) => {
        if (ruleIndex !== index) {
          return rule;
        }
        return {
          ...rule,
          ...patch,
        };
      }),
    );
  };

  const addRule = () => {
    onChange([...rules, { ...newRule }]);
  };

  const removeRule = (index: number) => {
    const next = rules.filter((_, ruleIndex) => ruleIndex !== index);
    onChange(next.length > 0 ? next : [{ ...newRule }]);
  };

  const moveRule = (index: number, direction: -1 | 1) => {
    const target = index + direction;
    if (target < 0 || target >= rules.length) {
      return;
    }
    const next = [...rules];
    const current = next[index];
    next[index] = next[target];
    next[target] = current;
    onChange(next);
  };

  return (
    <section className="control-panel full-width-panel">
      <div className="rule-header-row">
        <h3 className="panel-title">Rules</h3>
        <button className="action-button secondary" type="button" onClick={addRule}>
          Add Rule
        </button>
      </div>

      <div className="rules-table-wrap">
        <table className="rules-table">
          <thead>
            <tr>
              <th>Record</th>
              <th>Descriptor</th>
              <th>Res low</th>
              <th>Res high</th>
              <th>Color</th>
              <th>Radius</th>
              <th>Matches</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((rule, index) => (
              <tr key={`${rule.record_name}-${rule.descriptor}-${index}`}>
                <td>
                  <select
                    className="form-input"
                    value={rule.record_name}
                    onChange={(event) => updateRow(index, { record_name: event.target.value })}
                  >
                    <option value="ATOM  ">ATOM</option>
                    <option value="HETATM">HETATM</option>
                  </select>
                </td>
                <td>
                  <input
                    className="form-input mono-input"
                    type="text"
                    maxLength={10}
                    value={rule.descriptor}
                    onChange={(event) => updateRow(index, { descriptor: event.target.value.toUpperCase() })}
                  />
                </td>
                <td>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={9999}
                    step={1}
                    value={rule.res_low}
                    onChange={(event) => updateRow(index, { res_low: Number(event.target.value) })}
                  />
                </td>
                <td>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={9999}
                    step={1}
                    value={rule.res_high}
                    onChange={(event) => updateRow(index, { res_high: Number(event.target.value) })}
                  />
                </td>
                <td>
                  <input
                    className="rule-color-input"
                    type="color"
                    value={rgbToHex(rule.color)}
                    onChange={(event) => updateRow(index, { color: hexToRgb(event.target.value) })}
                  />
                </td>
                <td>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={20}
                    step={0.1}
                    value={rule.radius}
                    onChange={(event) => updateRow(index, { radius: Number(event.target.value) })}
                  />
                </td>
                <td>
                  <span className="match-pill">-</span>
                </td>
                <td>
                  <div className="rule-actions">
                    <button className="mini-button" type="button" onClick={() => moveRule(index, -1)} disabled={index === 0}>
                      Up
                    </button>
                    <button
                      className="mini-button"
                      type="button"
                      onClick={() => moveRule(index, 1)}
                      disabled={index === rules.length - 1}
                    >
                      Down
                    </button>
                    <button className="mini-button danger" type="button" onClick={() => removeRule(index)}>
                      Del
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};
