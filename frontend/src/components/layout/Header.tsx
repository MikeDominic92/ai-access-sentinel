'use client';

import React from 'react';
import { Bell, Search, HelpCircle } from 'lucide-react';
import { IdentityBadge } from '../ui/IdentityBadge';

export function Header() {
    return (
        <header className="h-20 glass-panel border-b-0 sticky top-0 z-30 px-6 flex items-center justify-between m-4 rounded-2xl">
            {/* Search Bar */}
            <div className="relative w-96 group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Search className="h-4 w-4 text-cyber-cyan/50 group-focus-within:text-cyber-cyan transition-colors" />
                </div>
                <input
                    type="text"
                    className="block w-full pl-10 pr-3 py-2 border border-white/10 rounded-xl leading-5 bg-black/20 text-white placeholder-gray-500 focus:outline-none focus:bg-black/40 focus:border-cyber-cyan/50 focus:ring-1 focus:ring-cyber-cyan/50 sm:text-sm transition-all backdrop-blur-sm"
                    placeholder="Search users, resources, or events... (Cmd+K)"
                />
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                    <span className="text-xs text-gray-500 border border-white/10 rounded px-1.5 py-0.5">⌘K</span>
                </div>
            </div>

            {/* Right Actions */}
            <div className="flex items-center space-x-6">
                {/* Notifications */}
                <button className="relative p-2 rounded-full hover:bg-white/5 text-gray-400 hover:text-cyber-cyan transition-all duration-300 group">
                    <Bell className="h-5 w-5 group-hover:animate-pulse" />
                    <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-cyber-purple border border-black shadow-[0_0_5px_#7c3aed]"></span>
                </button>

                <div className="h-8 w-px bg-white/10 mx-2"></div>

                {/* User Menu -> Identity Badge */}
                <IdentityBadge />
            </div>
        </header>
    );
}

