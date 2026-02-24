export type Empty = Record<string, never>;
export type ApiRoute = '/api/upload-pdb' | '/api/render';

export type PresetPayload = {
  name: string;
  pdb_path: string;
  rules: RulePayload[];
  transform: TransformPayload;
  world: WorldPayload;
  outlines: OutlinePayload;
};

export type TransformPayload = {
  scale: number;
  translate: [number, number, number];
  rotations: Array<[string, number]>;
  autocenter: string;
};

export type WorldPayload = {
  background: [number, number, number];
  fog_color: [number, number, number];
  fog_front: number;
  fog_back: number;
  shadows: boolean;
  shadow_strength: number;
  shadow_angle: number;
  shadow_min_z: number;
  shadow_max_dark: number;
  width: number;
  height: number;
};

export type OutlinePayload = {
  enabled: boolean;
  contour_low: number;
  contour_high: number;
  kernel: number;
  z_diff_min: number;
  z_diff_max: number;
  subunit_low: number;
  subunit_high: number;
  residue_low: number;
  residue_high: number;
  residue_diff: number;
};

export type RulePayload = {
  record_name: string;
  descriptor: string;
  res_low: number;
  res_high: number;
  color: [number, number, number];
  radius: number;
};

export type RenderPayload = {
  pdb_id: string;
  rules: RulePayload[];
  transform: TransformPayload;
  world: WorldPayload;
  outlines: OutlinePayload;
  output_format?: 'png' | 'ppm';
};

export type PresetListResponse = PresetPayload[];
