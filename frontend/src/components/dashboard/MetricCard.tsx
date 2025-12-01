'use client';

import React from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { cn } from '@/utils/cn';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface MetricCardProps {
    title: string;
    value: string;
    trend: number;
    data: { value: number }[];
    color?: string;
}

export function MetricCard({ title, value, trend, data, color = "#00D9FF" }: MetricCardProps) {
    const isPositive = trend >= 0;

    return (
        <Card variant="glass" noPadding className="relative overflow-hidden group">
            <CardContent className="p-6">
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <p className="text-silver text-sm font-medium uppercase tracking-wider">{title}</p>
                        <h3 className="text-3xl font-bold text-white mt-1 font-mono">{value}</h3>
                    </div>
                    <div className={cn(
                        "flex items-center px-2 py-1 rounded text-xs font-bold",
                        isPositive ? "bg-emerald-green/10 text-emerald-green" : "bg-coral-red/10 text-coral-red"
                    )}>
                        {isPositive ? <ArrowUpRight size={14} className="mr-1" /> : <ArrowDownRight size={14} className="mr-1" />}
                        {Math.abs(trend)}%
                    </div>
                </div>

                <div className="h-16 w-full absolute bottom-0 left-0 right-0 opacity-50 group-hover:opacity-80 transition-opacity">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke={color}
                                strokeWidth={2}
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    );
}
