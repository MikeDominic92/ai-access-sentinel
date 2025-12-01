'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Users } from 'lucide-react';

export function SimilarUsersPanel() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle className="flex items-center text-sm">
                    <Users size={16} className="mr-2 text-electric-cyan" /> Peer Analysis
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="text-center py-4">
                    <div className="text-4xl font-bold text-white mb-1">87%</div>
                    <p className="text-sm text-silver">
                        of Engineers in <span className="text-white">Engineering</span> have this access.
                    </p>
                </div>
                <div className="space-y-3 mt-4">
                    <div className="text-xs text-silver uppercase tracking-wider">Common Patterns</div>
                    <div className="flex items-center justify-between text-sm">
                        <span className="text-white">Same Job Title</span>
                        <span className="text-emerald-green">High Match</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-gray rounded-full overflow-hidden">
                        <div className="h-full bg-emerald-green w-[90%]"></div>
                    </div>

                    <div className="flex items-center justify-between text-sm">
                        <span className="text-white">Past Access History</span>
                        <span className="text-amber-gold">Medium Match</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-gray rounded-full overflow-hidden">
                        <div className="h-full bg-amber-gold w-[60%]"></div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
