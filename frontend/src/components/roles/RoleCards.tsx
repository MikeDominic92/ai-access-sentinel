'use client';

import React from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Users, Database, Server, ShieldAlert } from 'lucide-react';

const roles = [
    { id: 1, name: 'Data Analysts', count: 47, resources: ['BI_Tools', 'Sales_DB', 'Reports'], dept: 'Sales/Marketing', icon: Database, color: 'electric-cyan' },
    { id: 2, name: 'DevOps Engineers', count: 23, resources: ['K8s_Prod', 'CI_CD', 'AWS_Console'], dept: 'Engineering', icon: Server, color: 'amber-gold' },
    { id: 3, name: 'Shadow IT Admins', count: 8, resources: ['Admin_Panel', 'User_Mgmt', 'Logs'], dept: 'Various', icon: ShieldAlert, color: 'coral-red', warning: true },
];

export function RoleCards() {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {roles.map((role) => (
                <Card key={role.id} variant="glass-hover" className="cursor-pointer group">
                    <CardContent className="p-6">
                        <div className="flex justify-between items-start mb-4">
                            <div className={`p-3 rounded-lg bg-${role.color}/10 text-${role.color}`}>
                                <role.icon size={24} />
                            </div>
                            <Badge variant="outline" className="text-xs">
                                {role.count} Users
                            </Badge>
                        </div>

                        <h3 className="text-lg font-bold text-white mb-1 group-hover:text-electric-cyan transition-colors">{role.name}</h3>
                        <p className="text-sm text-silver mb-4">{role.dept}</p>

                        {role.warning && (
                            <div className="mb-4 p-2 bg-coral-red/10 border border-coral-red/20 rounded text-xs text-coral-red flex items-center">
                                <ShieldAlert size={12} className="mr-1.5" />
                                Potential security risk detected
                            </div>
                        )}

                        <div className="space-y-2">
                            <p className="text-xs text-silver uppercase tracking-wider font-semibold">Common Access</p>
                            <div className="flex flex-wrap gap-2">
                                {role.resources.map((res) => (
                                    <span key={res} className="px-2 py-1 rounded bg-white/5 text-xs text-silver border border-white/5">
                                        {res}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>
            ))}
        </div>
    );
}
