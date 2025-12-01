'use client';

import React from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Settings as SettingsIcon, Moon, Bell, Shield } from 'lucide-react';

export default function SettingsPage() {
    return (
        <DashboardLayout>
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-white">Settings</h1>
                <p className="text-silver">Manage platform configuration and preferences</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card variant="glass">
                    <CardHeader>
                        <CardTitle className="flex items-center">
                            <Moon size={20} className="mr-2 text-electric-cyan" /> Appearance
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-white">Theme</span>
                            <select className="bg-slate-gray/50 border border-white/10 rounded px-3 py-1 text-white text-sm">
                                <option>Dark (Default)</option>
                                <option>Light</option>
                                <option>System</option>
                            </select>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-white">Density</span>
                            <select className="bg-slate-gray/50 border border-white/10 rounded px-3 py-1 text-white text-sm">
                                <option>Compact</option>
                                <option>Comfortable</option>
                            </select>
                        </div>
                    </CardContent>
                </Card>

                <Card variant="glass">
                    <CardHeader>
                        <CardTitle className="flex items-center">
                            <Bell size={20} className="mr-2 text-amber-gold" /> Notifications
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-white">Email Alerts</span>
                            <div className="w-10 h-5 bg-electric-cyan rounded-full relative cursor-pointer">
                                <div className="absolute right-1 top-1 w-3 h-3 bg-deep-navy rounded-full"></div>
                            </div>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-white">Slack Integration</span>
                            <div className="w-10 h-5 bg-slate-gray rounded-full relative cursor-pointer">
                                <div className="absolute left-1 top-1 w-3 h-3 bg-white rounded-full"></div>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card variant="glass">
                    <CardHeader>
                        <CardTitle className="flex items-center">
                            <Shield size={20} className="mr-2 text-coral-red" /> Security
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-white">MFA Enforcement</span>
                            <Button size="sm" variant="outline">Configure</Button>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-white">API Keys</span>
                            <Button size="sm" variant="outline">Manage</Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </DashboardLayout>
    );
}
