import { FC, useRef } from 'react';
import type { MouseEvent } from 'react';

type RenderViewProps = {
  imageUrl?: string;
  busy: boolean;
  status: string;
  onRotate?: (dx: number, dy: number) => void;
};

export const RenderView: FC<RenderViewProps> = ({ imageUrl, busy, status, onRotate }) => {
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);

  const handleMouseDown = (event: MouseEvent<HTMLImageElement>) => {
    if (!onRotate) {
      return;
    }
    lastPointRef.current = { x: event.clientX, y: event.clientY };
  };

  const handleMouseMove = (event: MouseEvent<HTMLImageElement>) => {
    if (!onRotate || !event.buttons) {
      lastPointRef.current = null;
      return;
    }
    if (lastPointRef.current === null) {
      lastPointRef.current = { x: event.clientX, y: event.clientY };
      return;
    }

    const dx = event.clientX - lastPointRef.current.x;
    const dy = event.clientY - lastPointRef.current.y;
    lastPointRef.current = { x: event.clientX, y: event.clientY };
    if (Math.abs(dx) > 0 || Math.abs(dy) > 0) {
      onRotate(dx, dy);
    }
  };

  const handleMouseUp = () => {
    lastPointRef.current = null;
  };

  const handleDragStart = (event: MouseEvent<HTMLImageElement>) => {
    event.preventDefault();
  };

  if (!imageUrl) {
    return (
      <div className="preview-empty">
        <p>{busy ? 'Rendering…' : status || 'Load a structure and render to see output.'}</p>
        <p className="subtle-text">Preview appears here once a render completes.</p>
      </div>
    );
  }

  return (
    <figure className="preview-figure">
      <img
        src={imageUrl}
        alt="Rendered image"
        className="preview-image"
        draggable={false}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDragStart={handleDragStart}
        style={{ cursor: onRotate ? 'grab' : 'default' }}
      />
      <figcaption className="preview-caption">
        {busy ? 'Rendering…' : onRotate ? 'Drag image to rotate' : status}
      </figcaption>
    </figure>
  );
};
