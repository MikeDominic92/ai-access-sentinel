'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

const data = [
    { name: 'Engineering', risk: 85, events: 1200 },
    { name: 'Sales', risk: 45, events: 800 },
    { name: 'Marketing', risk: 30, events: 600 },
    { name: 'Finance', risk: 65, events: 400 },
    { name: 'HR', risk: 20, events: 300 },
    { name: 'IT', risk: 90, events: 1500 },
    { name: 'Legal', risk: 40, events: 200 },
];

export function DepartmentBreakdown() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>Department Risk Breakdown</CardTitle>
            </CardHeader>
            <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        layout="vertical"
                        data={data}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <XAxis type="number" hide />
                        <YAxis
                            dataKey="name"
                            type="category"
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            width={80}
                            axisLine={false}
                            tickLine={false}
                        />
                        <Tooltip
                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                            contentStyle={{ backgroundColor: '#1E2A3A', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        <Bar dataKey="risk" radius={[0, 4, 4, 0]} barSize={20}>
                            {data.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={
                                        entry.risk > 80 ? '#FF4757' :
                                            entry.risk > 50 ? '#FFB800' :
                                                '#00D68F'
                                    }
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
}
