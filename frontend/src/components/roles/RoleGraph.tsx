'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, Cell } from 'recharts';

// Simulating a graph layout with scatter plot for now
const nodes = Array.from({ length: 50 }, (_, i) => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
    z: Math.random() * 500 + 100, // Size
    cluster: Math.floor(Math.random() * 4), // 4 Roles
    name: `User ${i}`
}));

const colors = ['#00D9FF', '#FFB800', '#FF4757', '#00D68F'];

export function RoleGraph() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>Role Discovery Graph</CardTitle>
            </CardHeader>
            <CardContent className="h-[400px] relative">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <XAxis type="number" dataKey="x" hide />
                        <YAxis type="number" dataKey="y" hide />
                        <ZAxis type="number" dataKey="z" range={[50, 400]} />
                        <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{ backgroundColor: '#1E2A3A', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        <Scatter name="Users" data={nodes} fill="#8884d8">
                            {nodes.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={colors[entry.cluster]} />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>

                {/* Graph Overlay UI */}
                <div className="absolute bottom-4 right-4 bg-slate-gray/80 backdrop-blur p-3 rounded-lg border border-white/10 text-xs text-silver">
                    <div className="flex items-center mb-1"><span className="w-3 h-3 rounded-full bg-[#00D9FF] mr-2"></span> Data Analysts</div>
                    <div className="flex items-center mb-1"><span className="w-3 h-3 rounded-full bg-[#FFB800] mr-2"></span> DevOps</div>
                    <div className="flex items-center mb-1"><span className="w-3 h-3 rounded-full bg-[#FF4757] mr-2"></span> Shadow Admins</div>
                    <div className="flex items-center"><span className="w-3 h-3 rounded-full bg-[#00D68F] mr-2"></span> HR Staff</div>
                </div>
            </CardContent>
        </Card>
    );
}
