'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/utils/cn';
import {
    LayoutDashboard,
    Activity,
    ShieldAlert,
    Users,
    BrainCircuit,
    Settings,
    ChevronLeft,
    Menu,
    LogOut
} from 'lucide-react';

const navItems = [
    { name: 'Overview', href: '/', icon: LayoutDashboard },
    { name: 'Anomalies', href: '/anomalies', icon: Activity },
    { name: 'Risk Scoring', href: '/risk', icon: ShieldAlert },
    { name: 'Role Mining', href: '/roles', icon: Users },
    { name: 'Predictions', href: '/predictions', icon: BrainCircuit },
];

export function Sidebar() {
    const pathname = usePathname();
    const [collapsed, setCollapsed] = useState(false);

    return (
        <aside
            className={cn(
                "h-[calc(100vh-2rem)] fixed left-4 top-4 z-40 glass-panel border-r-0 transition-all duration-300 ease-in-out flex flex-col rounded-2xl",
                collapsed ? "w-20" : "w-64"
            )}
        >
            {/* Logo Area */}
            <div className="h-20 flex items-center justify-between px-4 border-b border-white/10">
                {!collapsed && (
                    <span className="font-orbitron font-bold text-lg tracking-wider text-white neon-text-cyan">
                        SENTINEL
                    </span>
                )}
                <button
                    onClick={() => setCollapsed(!collapsed)}
                    className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors ml-auto"
                >
                    {collapsed ? <Menu size={20} /> : <ChevronLeft size={20} />}
                </button>
            </div>

            {/* Navigation */}
            <nav className="flex-1 py-6 px-3 space-y-2 overflow-y-auto">
                {navItems.map((item) => {
                    const isActive = pathname === item.href;
                    const Icon = item.icon;

                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                                "flex items-center px-3 py-3 rounded-xl transition-all duration-300 group",
                                isActive
                                    ? "bg-cyber-cyan/10 text-cyber-cyan border border-cyber-cyan/20 shadow-[0_0_10px_rgba(6,182,212,0.1)]"
                                    : "text-gray-400 hover:bg-white/5 hover:text-white hover:pl-4"
                            )}
                        >
                            <Icon
                                size={20}
                                className={cn(
                                    "transition-colors",
                                    isActive ? "text-cyber-cyan" : "text-gray-500 group-hover:text-white"
                                )}
                            />
                            {!collapsed && (
                                <span className="ml-3 font-medium text-sm">
                                    {item.name}
                                </span>
                            )}
                        </Link>
                    );
                })}
            </nav>

            {/* Footer */}
            <div className="p-4 border-t border-white/10 space-y-2">
                <Link
                    href="/settings"
                    className="flex items-center px-3 py-2 rounded-lg text-gray-400 hover:bg-white/5 hover:text-white transition-colors"
                >
                    <Settings size={20} />
                    {!collapsed && <span className="ml-3 text-sm font-medium">Settings</span>}
                </Link>
                <button className="flex items-center w-full px-3 py-2 rounded-lg text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-colors">
                    <LogOut size={20} />
                    {!collapsed && <span className="ml-3 text-sm font-medium">Sign Out</span>}
                </button>
            </div>
        </aside>
    );
}

