'use client';

import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

interface DashboardLayoutProps {
    children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
    return (
        <div className="min-h-screen bg-deep-navy text-white font-sans selection:bg-electric-cyan/30">
            <Sidebar />
            <div className="pl-20 lg:pl-64 transition-all duration-300 ease-in-out">
                <Header />
                <main className="p-6 max-w-7xl mx-auto space-y-6">
                    {children}
                </main>
            </div>
        </div>
    );
}
