import { FC, useMemo } from 'react';

type AxisIndicatorProps = {
  xrot: number;
  yrot: number;
  zrot: number;
  size?: number;
};

function degToRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

/** Multiply two 3×3 matrices (row-major). */
function mul(a: number[], b: number[]): number[] {
  return [
    a[0]*b[0]+a[1]*b[3]+a[2]*b[6], a[0]*b[1]+a[1]*b[4]+a[2]*b[7], a[0]*b[2]+a[1]*b[5]+a[2]*b[8],
    a[3]*b[0]+a[4]*b[3]+a[5]*b[6], a[3]*b[1]+a[4]*b[4]+a[5]*b[7], a[3]*b[2]+a[4]*b[5]+a[5]*b[8],
    a[6]*b[0]+a[7]*b[3]+a[8]*b[6], a[6]*b[1]+a[7]*b[4]+a[8]*b[7], a[6]*b[2]+a[7]*b[5]+a[8]*b[8],
  ];
}

function rotX(a: number): number[] {
  const c = Math.cos(a), s = Math.sin(a);
  return [1,0,0, 0,c,-s, 0,s,c];
}
function rotY(a: number): number[] {
  const c = Math.cos(a), s = Math.sin(a);
  return [c,0,s, 0,1,0, -s,0,c];
}
function rotZ(a: number): number[] {
  const c = Math.cos(a), s = Math.sin(a);
  return [c,-s,0, s,c,0, 0,0,1];
}

function transform(m: number[], v: [number, number, number]): [number, number, number] {
  return [
    m[0]*v[0]+m[1]*v[1]+m[2]*v[2],
    m[3]*v[0]+m[4]*v[1]+m[5]*v[2],
    m[6]*v[0]+m[7]*v[1]+m[8]*v[2],
  ];
}

const AXES: { dir: [number, number, number]; label: string; color: string }[] = [
  { dir: [1, 0, 0], label: 'X', color: '#ff4444' },
  { dir: [0, 1, 0], label: 'Y', color: '#44cc44' },
  { dir: [0, 0, 1], label: 'Z', color: '#4488ff' },
];

export const AxisIndicator: FC<AxisIndicatorProps> = ({ xrot, yrot, zrot, size = 64 }) => {
  const axes = useMemo(() => {
    // Match the render pipeline: Z first, then Y, then X
    const m = mul(rotX(degToRad(xrot)), mul(rotY(degToRad(yrot)), rotZ(degToRad(zrot))));
    const len = size * 0.34;
    return AXES.map(({ dir, label, color }) => {
      const [px, py, pz] = transform(m, dir);
      return { x: px * len, y: -py * len, z: pz, label, color };
    })
      .sort((a, b) => a.z - b.z); // paint back-to-front
  }, [xrot, yrot, zrot, size]);

  const cx = size / 2;
  const cy = size / 2;

  return (
    <div className="axis-indicator">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {axes.map(({ x, y, z, label, color }) => {
          const opacity = 0.4 + 0.6 * ((z + 1) / 2); // fade axes pointing away
          return (
            <g key={label} opacity={opacity}>
              <line
                x1={cx} y1={cy}
                x2={cx + x} y2={cy + y}
                stroke={color}
                strokeWidth={2}
                strokeLinecap="round"
              />
              <circle cx={cx + x} cy={cy + y} r={3} fill={color} />
              <text
                x={cx + x * 1.25}
                y={cy + y * 1.25}
                textAnchor="middle"
                dominantBaseline="central"
                fill={color}
                fontSize={10}
                fontWeight={600}
                fontFamily="var(--font-body)"
              >
                {label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};
