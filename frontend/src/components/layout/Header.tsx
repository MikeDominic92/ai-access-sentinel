'use client';

import React from 'react';
import { Bell, Search, User } from 'lucide-react';
import { cn } from '@/utils/cn';

export function Header() {
    return (
        <header className="h-16 border-b border-white/10 bg-deep-navy/80 backdrop-blur-md sticky top-0 z-30 px-6 flex items-center justify-between">
            {/* Search Bar */}
            <div className="relative w-96">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Search className="h-4 w-4 text-silver" />
                </div>
                <input
                    type="text"
                    className="block w-full pl-10 pr-3 py-2 border border-white/10 rounded-lg leading-5 bg-slate-gray/50 text-white placeholder-silver focus:outline-none focus:bg-slate-gray focus:border-electric-cyan/50 focus:ring-1 focus:ring-electric-cyan/50 sm:text-sm transition-colors"
                    placeholder="Search users, resources, or events... (Cmd+K)"
                />
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                    <span className="text-xs text-silver border border-white/10 rounded px-1.5 py-0.5">⌘K</span>
                </div>
            </div>

            {/* Right Actions */}
            <div className="flex items-center space-x-4">
                {/* Notifications */}
                <button className="relative p-2 rounded-full hover:bg-white/5 text-silver hover:text-white transition-colors">
                    <Bell className="h-5 w-5" />
                    <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-coral-red border border-deep-navy"></span>
                </button>

                {/* User Menu */}
                <button className="flex items-center space-x-2 p-1.5 rounded-lg hover:bg-white/5 transition-colors">
                    <div className="h-8 w-8 rounded-full bg-slate-gray flex items-center justify-center border border-white/10">
                        <User className="h-4 w-4 text-silver" />
                    </div>
                </button>
            </div>
        </header>
    );
}
