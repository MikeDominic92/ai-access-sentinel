'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { ChevronDown, ChevronUp, AlertTriangle, ShieldAlert } from 'lucide-react';
import { cn } from '@/utils/cn';

const anomalies = [
    { id: 1, timestamp: '2023-10-25 03:47:12', user: 'jane.doe', resource: 'AWS_PROD_DB', action: 'Query', score: -0.72, location: 'Moscow, RU', details: 'Unusual location for this user. Previous access was from New York, US.' },
    { id: 2, timestamp: '2023-10-25 04:15:00', user: 'svc_backup', resource: 'S3_Finance', action: 'Delete', score: -0.58, location: 'Internal', details: 'High volume of delete operations (847) in short window (5m).' },
    { id: 3, timestamp: '2023-10-25 05:22:18', user: 'admin_bob', resource: 'IAM_Role', action: 'Escalate', score: -0.81, location: 'London, UK', details: 'Simultaneous privilege escalation on 12 systems.' },
    { id: 4, timestamp: '2023-10-25 06:10:45', user: 'dev_sarah', resource: 'Prod_K8s', action: 'Exec', score: -0.65, location: 'Home IP', details: 'Execution of sensitive command "rm -rf" in production container.' },
    { id: 5, timestamp: '2023-10-25 07:05:30', user: 'contractor_mike', resource: 'Git_Repo', action: 'Clone', score: -0.45, location: 'Unknown', details: 'Cloning entire repository at 3 AM local time.' },
];

export function DetectedAnomaliesTable() {
    const [expandedId, setExpandedId] = useState<number | null>(null);

    const toggleExpand = (id: number) => {
        setExpandedId(expandedId === id ? null : id);
    };

    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>Detected Anomalies</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                        <thead className="text-silver border-b border-white/10">
                            <tr>
                                <th className="pb-3 pl-2">Timestamp</th>
                                <th className="pb-3">User</th>
                                <th className="pb-3">Resource</th>
                                <th className="pb-3">Action</th>
                                <th className="pb-3">Score</th>
                                <th className="pb-3">Location</th>
                                <th className="pb-3"></th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {anomalies.map((anomaly) => (
                                <React.Fragment key={anomaly.id}>
                                    <tr
                                        className={cn(
                                            "group hover:bg-white/5 transition-colors cursor-pointer",
                                            expandedId === anomaly.id && "bg-white/5"
                                        )}
                                        onClick={() => toggleExpand(anomaly.id)}
                                    >
                                        <td className="py-3 pl-2 text-silver font-mono">{anomaly.timestamp}</td>
                                        <td className="py-3 font-medium text-white">{anomaly.user}</td>
                                        <td className="py-3 text-silver">{anomaly.resource}</td>
                                        <td className="py-3 text-silver">{anomaly.action}</td>
                                        <td className="py-3">
                                            <Badge variant={anomaly.score < -0.7 ? 'danger' : 'warning'}>
                                                {anomaly.score}
                                            </Badge>
                                        </td>
                                        <td className="py-3 text-silver">{anomaly.location}</td>
                                        <td className="py-3 pr-2 text-right">
                                            {expandedId === anomaly.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                        </td>
                                    </tr>
                                    {expandedId === anomaly.id && (
                                        <tr className="bg-white/5 animate-in fade-in slide-in-from-top-1 duration-200">
                                            <td colSpan={7} className="p-4">
                                                <div className="flex items-start space-x-3 text-silver">
                                                    <AlertTriangle className="text-amber-gold shrink-0 mt-0.5" size={18} />
                                                    <div>
                                                        <span className="font-semibold text-white block mb-1">Analysis Details:</span>
                                                        {anomaly.details}
                                                    </div>
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </React.Fragment>
                            ))}
                        </tbody>
                    </table>
                </div>
            </CardContent>
        </Card>
    );
}
