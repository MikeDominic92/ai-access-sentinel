'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const data = [
    { day: '1', score: 20 }, { day: '5', score: 22 }, { day: '10', score: 25 },
    { day: '15', score: 45 }, { day: '20', score: 60 }, { day: '25', score: 55 },
    { day: '30', score: 73 },
];

export function RiskTimeline() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>30-Day Risk Trend</CardTitle>
            </CardHeader>
            <CardContent className="h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis
                            dataKey="day"
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                        />
                        <YAxis
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                            domain={[0, 100]}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1E2A3A', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        <Line
                            type="monotone"
                            dataKey="score"
                            stroke="#FF4757"
                            strokeWidth={3}
                            dot={{ fill: '#FF4757', r: 4 }}
                            activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
}
