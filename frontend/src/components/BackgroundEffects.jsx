import { useEffect, useRef } from 'react';

/**
 * Full-screen background: animated gradient orbs + interactive dot-grid canvas.
 * The grid dots glow and grow when the mouse approaches them.
 */
export default function BackgroundEffects() {
  const canvasRef = useRef(null);
  const mouseRef = useRef({ x: -9999, y: -9999 });
  const rafRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const SPACING = 36;
    const BASE_R = 1.2;
    const MAX_R = 5;
    const RADIUS = 140; // influence radius

    let cols, rows, dots;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      cols = Math.ceil(canvas.width / SPACING) + 1;
      rows = Math.ceil(canvas.height / SPACING) + 1;
      dots = [];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          dots.push({ x: c * SPACING, y: r * SPACING, radius: BASE_R });
        }
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;

      for (const d of dots) {
        const dx = d.x - mx;
        const dy = d.y - my;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const t = Math.max(0, 1 - dist / RADIUS);
        const r = BASE_R + (MAX_R - BASE_R) * t * t;
        const alpha = 0.12 + 0.55 * t * t;

        ctx.beginPath();
        ctx.arc(d.x, d.y, r, 0, Math.PI * 2);
        // Use brand cyan color
        ctx.fillStyle = `rgba(56, 189, 248, ${alpha})`;
        ctx.fill();
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    const onMouseMove = (e) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', onMouseMove);
    draw();

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', onMouseMove);
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      {/* Interactive dot-grid canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ opacity: 0.6 }}
      />

      {/* Gradient orb — top-left brand */}
      <div
        className="absolute rounded-full bg-brand-600/20 blur-[140px] animate-orb-1"
        style={{ width: '45vw', height: '45vw', top: '-8%', left: '-8%' }}
      />

      {/* Gradient orb — bottom-right accent */}
      <div
        className="absolute rounded-full bg-accent-600/12 blur-[160px] animate-orb-2"
        style={{ width: '55vw', height: '55vw', bottom: '-12%', right: '-12%' }}
      />

      {/* Small emerald orb — mid */}
      <div
        className="absolute rounded-full bg-emerald-600/10 blur-[100px] animate-orb-3"
        style={{ width: '30vw', height: '30vw', top: '38%', left: '58%' }}
      />
    </div>
  );
}
