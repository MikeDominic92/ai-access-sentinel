'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Tooltip, ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Cell } from 'recharts';

const roles = ['Data Analysts', 'DevOps', 'Shadow Admins', 'HR Staff'];
const permissions = ['Read_PII', 'Write_DB', 'Deploy_Prod', 'View_Logs', 'Manage_Users'];

// Generate overlap data
const data: { x: number; y: number; z: number; role: string; perm: string }[] = [];
roles.forEach((role, rIndex) => {
    permissions.forEach((perm, pIndex) => {
        // Random overlap
        const value = Math.random() > 0.5 ? Math.floor(Math.random() * 100) : 0;
        if (value > 0) {
            data.push({ x: pIndex, y: rIndex, z: value, role, perm });
        }
    });
});

export function PermissionOverlapMatrix() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>Permission Overlap Matrix</CardTitle>
            </CardHeader>
            <CardContent className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <XAxis
                            type="number"
                            dataKey="x"
                            domain={[0, permissions.length - 1]}
                            tickFormatter={(val) => permissions[val]}
                            tick={{ fill: '#94A3B8', fontSize: 10 }}
                            interval={0}
                            angle={-45}
                            textAnchor="end"
                            height={60}
                        />
                        <YAxis
                            type="number"
                            dataKey="y"
                            domain={[0, roles.length - 1]}
                            tickFormatter={(val) => roles[val]}
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            width={100}
                        />
                        <ZAxis type="number" dataKey="z" range={[0, 400]} />
                        <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                        <div className="bg-slate-gray border border-white/10 p-2 rounded shadow-lg text-xs">
                                            <p className="font-bold text-white">{data.role} + {data.perm}</p>
                                            <p className="text-silver">Overlap Score: {data.z}</p>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <Scatter data={data} shape="circle">
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={`rgba(0, 217, 255, ${entry.z / 100})`} />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
}
