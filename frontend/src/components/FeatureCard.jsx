import { useState, useRef } from 'react';

export default function FeatureCard({ icon: Icon, title, description, colorClass }) {
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [opacity, setOpacity] = useState(0);
    const cardRef = useRef(null);

    const handleMouseMove = (e) => {
        if (!cardRef.current) return;
        const rect = cardRef.current.getBoundingClientRect();
        setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    };

    const handleMouseEnter = () => setOpacity(1);
    const handleMouseLeave = () => setOpacity(0);

    return (
        <div
            ref={cardRef}
            onMouseMove={handleMouseMove}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            className="relative overflow-hidden glass-card p-6 rounded-2xl flex flex-col items-center hover:bg-gray-800/80 transition-colors group cursor-default shadow-lg hover:shadow-xl hover:-translate-y-1 duration-300"
        >
            <div
                className="pointer-events-none absolute -inset-px opacity-0 transition duration-300"
                style={{
                    opacity,
                    background: `radial-gradient(400px circle at ${position.x}px ${position.y}px, rgba(255,255,255,0.06), transparent 40%)`
                }}
            />
            <div className={`w-14 h-14 rounded-full flex items-center justify-center mb-5 transition-transform duration-300 group-hover:scale-110 shadow-inner ${colorClass}`}>
                <Icon className="w-7 h-7" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-brand-300 transition-colors">{title}</h3>
            <p className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors leading-relaxed">{description}</p>
        </div>
    );
}
