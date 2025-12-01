'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

const data = [
    { factor: 'Failed Logins', score: 85, color: '#FF4757' },
    { factor: 'Off-Hours Access', score: 65, color: '#FF7F0E' },
    { factor: 'Sensitive Resources', score: 45, color: '#FFB800' },
    { factor: 'Location Anomalies', score: 30, color: '#00D9FF' },
    { factor: 'Privilege Level', score: 20, color: '#00D68F' },
];

export function RiskFactorBreakdown() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>Risk Factor Contribution</CardTitle>
            </CardHeader>
            <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        layout="vertical"
                        data={data}
                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                    >
                        <XAxis type="number" hide />
                        <YAxis
                            dataKey="factor"
                            type="category"
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            width={120}
                            axisLine={false}
                            tickLine={false}
                        />
                        <Tooltip
                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                            contentStyle={{ backgroundColor: '#1E2A3A', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={24}>
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
}
