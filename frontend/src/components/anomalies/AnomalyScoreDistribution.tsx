'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts';

const data = [
    { range: '-1.0', count: 5 },
    { range: '-0.9', count: 12 },
    { range: '-0.8', count: 25 },
    { range: '-0.7', count: 45 },
    { range: '-0.6', count: 80 },
    { range: '-0.5', count: 150 },
    { range: '-0.4', count: 300 },
    { range: '-0.3', count: 800 },
    { range: '-0.2', count: 2500 },
    { range: '-0.1', count: 15000 },
    { range: '0.0', count: 28000 },
];

export function AnomalyScoreDistribution() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>Anomaly Score Distribution</CardTitle>
            </CardHeader>
            <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <XAxis
                            dataKey="range"
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                        />
                        <YAxis
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                        />
                        <Tooltip
                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                            contentStyle={{ backgroundColor: '#1E2A3A', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        <ReferenceLine x="-0.3" stroke="#FF4757" strokeDasharray="3 3" label={{ position: 'top', value: 'High Risk Threshold', fill: '#FF4757', fontSize: 12 }} />
                        <Bar dataKey="count" fill="#00D9FF" radius={[4, 4, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
}
