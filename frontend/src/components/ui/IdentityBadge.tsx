import React from 'react';

export const IdentityBadge = () => {
    return (
        <div className="flex items-center gap-3 px-4 py-2 rounded-full glass-panel border-cyber-cyan/30 neon-glow-cyan hover:bg-white/10 transition-all duration-300 group cursor-pointer">
            <div className="relative">
                <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-cyber-cyan to-cyber-purple p-[2px]">
                    <div className="w-full h-full rounded-full bg-black flex items-center justify-center overflow-hidden">
                        <span className="text-xs font-bold text-white">MD</span>
                    </div>
                </div>
                <div className="absolute bottom-0 right-0 w-3 h-3 bg-cyber-green rounded-full border-2 border-black shadow-[0_0_5px_#10b981]"></div>
            </div>
            <div className="flex flex-col">
                <span className="text-sm font-bold text-white group-hover:text-cyber-cyan transition-colors font-orbitron tracking-wide">
                    MIKE DOMINIC
                </span>
                <span className="text-[10px] text-gray-400 uppercase tracking-wider">
                    Cybersecurity Eng.
                </span>
            </div>
        </div>
    );
};
