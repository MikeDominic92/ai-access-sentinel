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
    ChevronRight,
    Menu
} from 'lucide-react';

const navItems = [
    { name: 'Overview', href: '/', icon: LayoutDashboard },
    { name: 'Anomalies', href: '/anomalies', icon: Activity },
    { name: 'Risk Scoring', href: '/risk', icon: ShieldAlert },
    { name: 'Role Mining', href: '/roles', icon: Users },
    { name: 'Predictions', href: '/predictions', icon: BrainCircuit },
    { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar() {
    const pathname = usePathname();
    const [collapsed, setCollapsed] = useState(false);

    return (
        <aside
            className={cn(
                "h-screen fixed left-0 top-0 z-40 bg-deep-navy border-r border-white/10 transition-all duration-300 ease-in-out flex flex-col",
                collapsed ? "w-20" : "w-64"
            )}
        >
            {/* Logo Area */}
            <div className="h-16 flex items-center justify-between px-4 border-b border-white/10">
                {!collapsed && (
                    <span className="font-bold text-xl tracking-wider text-electric-cyan">
                        SENTINEL
                    </span>
                )}
                <button
                    onClick={() => setCollapsed(!collapsed)}
                    className="p-2 rounded-lg hover:bg-white/5 text-silver transition-colors"
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
                                "flex items-center px-3 py-3 rounded-lg transition-all duration-200 group",
                                isActive
                                    ? "bg-electric-cyan/10 text-electric-cyan"
                                    : "text-silver hover:bg-white/5 hover:text-white"
                            )}
                        >
                            <Icon
                                size={20}
                                className={cn(
                                    "transition-colors",
                                    isActive ? "text-electric-cyan" : "text-silver group-hover:text-white"
                                )}
                            />
                            {!collapsed && (
                                <span className="ml-3 font-medium text-sm">
                                    {item.name}
                                </span>
                            )}

                            {/* Active Indicator */}
                            {isActive && !collapsed && (
                                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-electric-cyan shadow-[0_0_8px_#00D9FF]" />
                            )}
                        </Link>
                    );
                })}
            </nav>

            {/* User Profile / Footer */}
            <div className="p-4 border-t border-white/10">
                <div className={cn("flex items-center", collapsed ? "justify-center" : "space-x-3")}>
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-electric-cyan to-deep-navy border border-white/20 flex items-center justify-center text-xs font-bold text-white">
                        JD
                    </div>
                    {!collapsed && (
                        <div className="flex flex-col">
                            <span className="text-sm font-medium text-white">Jane Doe</span>
                            <span className="text-xs text-silver">SOC Analyst</span>
                        </div>
                    )}
                </div>
            </div>
        </aside>
    );
}
