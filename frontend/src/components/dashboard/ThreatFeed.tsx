'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { AlertTriangle, ShieldCheck, UserX } from 'lucide-react';

const threats = [
    { id: 1, message: "Suspicious login attempt from Moscow", time: "2m ago", severity: "critical", icon: AlertTriangle },
    { id: 2, message: "Privilege escalation detected: user_bob", time: "15m ago", severity: "high", icon: UserX },
    { id: 3, message: "New admin role assigned to service_account", time: "1h ago", severity: "medium", icon: ShieldCheck },
    { id: 4, message: "Multiple failed MFA attempts: user_sarah", time: "2h ago", severity: "medium", icon: AlertTriangle },
    { id: 5, message: "Unusual data egress volume: 5GB", time: "3h ago", severity: "high", icon: AlertTriangle },
];

export function ThreatFeed() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle className="flex items-center">
                    <span className="w-2 h-2 rounded-full bg-coral-red animate-pulse mr-2"></span>
                    Real-Time Threat Feed
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                {threats.map((threat) => (
                    <div key={threat.id} className="flex items-start space-x-3 p-3 rounded-lg hover:bg-white/5 transition-colors border border-transparent hover:border-white/5 group animate-in slide-in-from-right fade-in duration-500">
                        <div className={`p-2 rounded-lg bg-${threat.severity === 'critical' ? 'coral-red' : threat.severity === 'high' ? 'amber-gold' : 'electric-cyan'}/10 text-${threat.severity === 'critical' ? 'coral-red' : threat.severity === 'high' ? 'amber-gold' : 'electric-cyan'}`}>
                            <threat.icon size={16} />
                        </div>
                        <div className="flex-1">
                            <p className="text-sm text-white font-medium group-hover:text-electric-cyan transition-colors">{threat.message}</p>
                            <p className="text-xs text-silver mt-1">{threat.time}</p>
                        </div>
                        <Badge variant={threat.severity === 'critical' ? 'danger' : threat.severity === 'high' ? 'warning' : 'info'} size="sm">
                            {threat.severity.toUpperCase()}
                        </Badge>
                    </div>
                ))}
            </CardContent>
        </Card>
    );
}
