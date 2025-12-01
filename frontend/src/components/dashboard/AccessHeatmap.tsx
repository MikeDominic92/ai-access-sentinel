'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, Cell } from 'recharts';

const data = [
    { day: 0, hour: 0, value: 10 }, { day: 0, hour: 1, value: 5 }, { day: 0, hour: 2, value: 2 },
    // ... (mock data would go here, simplified for brevity)
    { day: 1, hour: 10, value: 50 }, { day: 1, hour: 11, value: 80 },
    { day: 2, hour: 14, value: 100 },
];

// Generate full grid data
const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const hours = Array.from({ length: 24 }, (_, i) => i);
const fullData: { day: number; hour: number; value: number; dayName: string }[] = [];

days.forEach((day, dayIndex) => {
    hours.forEach((hour) => {
        // Random value with some pattern
        let value = Math.floor(Math.random() * 20);
        if (dayIndex < 5 && hour > 8 && hour < 18) value += Math.floor(Math.random() * 80); // Work hours
        fullData.push({ day: dayIndex, hour, value, dayName: day });
    });
});

export function AccessHeatmap() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>Access Heatmap</CardTitle>
            </CardHeader>
            <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                        margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                        <XAxis
                            type="number"
                            dataKey="hour"
                            name="Hour"
                            domain={[0, 23]}
                            tickCount={24}
                            tick={{ fill: '#94A3B8', fontSize: 10 }}
                            axisLine={false}
                            tickLine={false}
                        />
                        <YAxis
                            type="number"
                            dataKey="day"
                            name="Day"
                            domain={[0, 6]}
                            tickFormatter={(value) => days[value]}
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                            width={40}
                        />
                        <ZAxis type="number" dataKey="value" range={[0, 500]} />
                        <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{ backgroundColor: '#1E2A3A', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        <Scatter data={fullData} shape="square">
                            {fullData.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={`rgba(0, 217, 255, ${Math.min(0.1 + (entry.value / 100), 1)})`}
                                />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
}
