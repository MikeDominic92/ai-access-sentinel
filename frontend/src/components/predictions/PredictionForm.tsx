'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { BrainCircuit } from 'lucide-react';

export function PredictionForm({ onPredict }: { onPredict: () => void }) {
    return (
        <Card variant="glass" className="h-full">
            <CardHeader>
                <CardTitle>Access Request Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <label className="text-sm text-silver">Department</label>
                        <select className="w-full bg-slate-gray/50 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-electric-cyan/50">
                            <option>Engineering</option>
                            <option>Sales</option>
                            <option>Finance</option>
                        </select>
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm text-silver">Job Title</label>
                        <select className="w-full bg-slate-gray/50 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-electric-cyan/50">
                            <option>Software Engineer</option>
                            <option>DevOps Engineer</option>
                            <option>Product Manager</option>
                        </select>
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm text-silver">Resource</label>
                        <select className="w-full bg-slate-gray/50 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-electric-cyan/50">
                            <option>AWS_PROD_DB</option>
                            <option>GitHub_Repo_Main</option>
                            <option>Salesforce_CRM</option>
                        </select>
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm text-silver">Action</label>
                        <select className="w-full bg-slate-gray/50 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-electric-cyan/50">
                            <option>Read</option>
                            <option>Write</option>
                            <option>Delete</option>
                            <option>Admin</option>
                        </select>
                    </div>
                </div>

                <div className="pt-4">
                    <Button className="w-full h-12 text-lg font-bold shadow-[0_0_20px_rgba(0,217,255,0.4)] hover:shadow-[0_0_30px_rgba(0,217,255,0.6)] transition-shadow" onClick={onPredict}>
                        <BrainCircuit className="mr-2" /> PREDICT ACCESS DECISION
                    </Button>
                </div>
            </CardContent>
        </Card>
    );
}
