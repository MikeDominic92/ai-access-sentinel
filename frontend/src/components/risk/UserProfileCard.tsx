'use client';

import React from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { User, MapPin, Building, Mail } from 'lucide-react';

interface UserProfileProps {
    user: {
        name: string;
        role: string;
        department: string;
        location: string;
        email: string;
        riskScore: number;
        riskLevel: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'MINIMAL';
    };
}

export function UserProfileCard({ user }: UserProfileProps) {
    const getRiskColor = (level: string) => {
        switch (level) {
            case 'CRITICAL': return '#FF4757';
            case 'HIGH': return '#FF7F0E';
            case 'MEDIUM': return '#FFB800';
            case 'LOW': return '#00D68F';
            default: return '#1F77B4';
        }
    };

    const riskColor = getRiskColor(user.riskLevel);

    return (
        <Card variant="glass" className="h-full">
            <CardContent className="p-6 flex flex-col items-center text-center">
                <div className="relative mb-6">
                    {/* Risk Dial SVG */}
                    <svg className="w-40 h-40 transform -rotate-90">
                        <circle cx="80" cy="80" r="70" stroke="#1E2A3A" strokeWidth="10" fill="none" />
                        <circle
                            cx="80"
                            cy="80"
                            r="70"
                            stroke={riskColor}
                            strokeWidth="10"
                            fill="none"
                            strokeDasharray="440"
                            strokeDashoffset={440 - (440 * user.riskScore) / 100}
                            strokeLinecap="round"
                        />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <div className="w-24 h-24 rounded-full bg-slate-gray border-4 border-deep-navy overflow-hidden mb-2">
                            {/* Placeholder Avatar */}
                            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-electric-cyan to-deep-navy text-white text-2xl font-bold">
                                {user.name.split(' ').map(n => n[0]).join('')}
                            </div>
                        </div>
                    </div>
                    <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 bg-deep-navy px-3 py-1 rounded-full border border-white/10">
                        <span className="text-xl font-bold text-white">{user.riskScore}</span>
                        <span className="text-xs text-silver">/100</span>
                    </div>
                </div>

                <h2 className="text-2xl font-bold text-white mt-4">{user.name}</h2>
                <p className="text-silver mb-4">{user.role}</p>

                <Badge variant={user.riskLevel === 'CRITICAL' ? 'danger' : user.riskLevel === 'HIGH' ? 'warning' : 'success'} className="mb-6 px-4 py-1 text-sm">
                    {user.riskLevel} RISK
                </Badge>

                <div className="w-full space-y-3 text-sm text-silver">
                    <div className="flex items-center justify-between p-2 rounded bg-white/5">
                        <div className="flex items-center"><Building size={16} className="mr-2" /> Department</div>
                        <span className="text-white">{user.department}</span>
                    </div>
                    <div className="flex items-center justify-between p-2 rounded bg-white/5">
                        <div className="flex items-center"><MapPin size={16} className="mr-2" /> Location</div>
                        <span className="text-white">{user.location}</span>
                    </div>
                    <div className="flex items-center justify-between p-2 rounded bg-white/5">
                        <div className="flex items-center"><Mail size={16} className="mr-2" /> Email</div>
                        <span className="text-white">{user.email}</span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
