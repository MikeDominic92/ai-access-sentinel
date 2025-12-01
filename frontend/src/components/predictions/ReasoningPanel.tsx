'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Lightbulb, Check, AlertTriangle } from 'lucide-react';

export function ReasoningPanel() {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle className="flex items-center text-sm">
                    <Lightbulb size={16} className="mr-2 text-amber-gold" /> AI Reasoning
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex items-start space-x-3">
                    <div className="mt-0.5 p-1 rounded-full bg-emerald-green/10 text-emerald-green">
                        <Check size={12} />
                    </div>
                    <p className="text-sm text-silver">
                        User has <span className="text-white font-medium">valid business justification</span> based on recent ticket #IAM-492.
                    </p>
                </div>
                <div className="flex items-start space-x-3">
                    <div className="mt-0.5 p-1 rounded-full bg-emerald-green/10 text-emerald-green">
                        <Check size={12} />
                    </div>
                    <p className="text-sm text-silver">
                        Request is within <span className="text-white font-medium">normal business hours</span> (10:42 AM local).
                    </p>
                </div>
                <div className="flex items-start space-x-3">
                    <div className="mt-0.5 p-1 rounded-full bg-amber-gold/10 text-amber-gold">
                        <AlertTriangle size={12} />
                    </div>
                    <p className="text-sm text-silver">
                        Resource is classified as <span className="text-white font-medium">Sensitive PII</span>. Requires MFA (Verified).
                    </p>
                </div>
            </CardContent>
        </Card>
    );
}
