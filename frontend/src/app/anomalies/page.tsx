'use client';

import React from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { AnomalyScoreDistribution } from '@/components/anomalies/AnomalyScoreDistribution';
import { DetectedAnomaliesTable } from '@/components/anomalies/DetectedAnomaliesTable';
import { Card, CardContent } from '@/components/ui/Card';
import { Activity, ShieldAlert, Cpu } from 'lucide-react';

export default function AnomalyDetectionPage() {
    return (
        <DashboardLayout>
            <div className="flex flex-col space-y-6">
                {/* Header Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <Card variant="glass" className="flex items-center p-6">
                        <div className="p-3 rounded-full bg-coral-red/10 text-coral-red mr-4">
                            <ShieldAlert size={24} />
                        </div>
                        <div>
                            <p className="text-silver text-sm uppercase tracking-wider">Total Anomalies</p>
                            <h3 className="text-2xl font-bold text-white">127</h3>
                            <p className="text-xs text-coral-red mt-1">+12% from yesterday</p>
                        </div>
                    </Card>

                    <Card variant="glass" className="flex items-center p-6">
                        <div className="p-3 rounded-full bg-amber-gold/10 text-amber-gold mr-4">
                            <Activity size={24} />
                        </div>
                        <div>
                            <p className="text-silver text-sm uppercase tracking-wider">Anomaly Rate</p>
                            <h3 className="text-2xl font-bold text-white">0.27%</h3>
                            <p className="text-xs text-emerald-green mt-1">-0.05% improvement</p>
                        </div>
                    </Card>

                    <Card variant="glass" className="flex items-center p-6">
                        <div className="p-3 rounded-full bg-electric-cyan/10 text-electric-cyan mr-4">
                            <Cpu size={24} />
                        </div>
                        <div>
                            <p className="text-silver text-sm uppercase tracking-wider">Active Model</p>
                            <h3 className="text-xl font-bold text-white">Isolation Forest</h3>
                            <p className="text-xs text-silver mt-1">v2.4.1 • Updated 2h ago</p>
                        </div>
                    </Card>
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[400px]">
                    <div className="lg:col-span-2 h-full">
                        <AnomalyScoreDistribution />
                    </div>
                    <div className="lg:col-span-1 h-full">
                        {/* Placeholder for Gauge or Radar Chart */}
                        <Card variant="glass" className="h-full flex flex-col items-center justify-center text-center p-6">
                            <div className="relative w-48 h-48 flex items-center justify-center">
                                <svg className="w-full h-full transform -rotate-90">
                                    <circle cx="96" cy="96" r="88" stroke="#1E2A3A" strokeWidth="12" fill="none" />
                                    <circle cx="96" cy="96" r="88" stroke="#FF4757" strokeWidth="12" fill="none" strokeDasharray="552" strokeDashoffset="400" strokeLinecap="round" />
                                </svg>
                                <div className="absolute inset-0 flex flex-col items-center justify-center">
                                    <span className="text-4xl font-bold text-white">127</span>
                                    <span className="text-sm text-silver">Detected</span>
                                </div>
                            </div>
                            <h3 className="mt-4 text-lg font-medium text-white">Anomaly Count</h3>
                            <p className="text-sm text-silver mt-2">Exceeding threshold -0.3</p>
                        </Card>
                    </div>
                </div>

                {/* Table Row */}
                <div>
                    <DetectedAnomaliesTable />
                </div>
            </div>
        </DashboardLayout>
    );
}
