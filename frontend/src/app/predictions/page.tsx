'use client';

import React, { useState } from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { PredictionForm } from '@/components/predictions/PredictionForm';
import { PredictionResult } from '@/components/predictions/PredictionResult';
import { SimilarUsersPanel } from '@/components/predictions/SimilarUsersPanel';
import { ReasoningPanel } from '@/components/predictions/ReasoningPanel';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { History } from 'lucide-react';

export default function PredictionsPage() {
    const [prediction, setPrediction] = useState<'approve' | 'deny' | null>(null);

    const handlePredict = () => {
        // Simulate API call
        setTimeout(() => {
            setPrediction('approve');
        }, 1000);
    };

    return (
        <DashboardLayout>
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-white">Access Prediction</h1>
                <p className="text-silver">AI-powered access request decision support</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Input */}
                <div className="lg:col-span-2">
                    <PredictionForm onPredict={handlePredict} />

                    {/* Result Area */}
                    <div className="mt-6">
                        <PredictionResult result={prediction} />
                    </div>
                </div>

                {/* Right Column: Context */}
                <div className="lg:col-span-1 flex flex-col space-y-6">
                    <SimilarUsersPanel />
                    <ReasoningPanel />
                </div>
            </div>

            {/* Recent Predictions Table */}
            <div className="mt-8">
                <Card variant="glass">
                    <CardHeader>
                        <CardTitle className="flex items-center text-lg">
                            <History size={20} className="mr-2 text-silver" /> Recent Predictions
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <table className="w-full text-left text-sm">
                            <thead className="text-silver border-b border-white/10">
                                <tr>
                                    <th className="pb-3 pl-2">Time</th>
                                    <th className="pb-3">User</th>
                                    <th className="pb-3">Resource</th>
                                    <th className="pb-3">Decision</th>
                                    <th className="pb-3">Confidence</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5">
                                <tr className="group hover:bg-white/5 transition-colors">
                                    <td className="py-3 pl-2 text-silver">10:42 AM</td>
                                    <td className="py-3 font-medium text-white">jane.doe</td>
                                    <td className="py-3 text-silver">AWS_PROD_DB</td>
                                    <td className="py-3"><span className="text-emerald-green font-bold">APPROVE</span></td>
                                    <td className="py-3 text-silver">94.2%</td>
                                </tr>
                                <tr className="group hover:bg-white/5 transition-colors">
                                    <td className="py-3 pl-2 text-silver">09:15 AM</td>
                                    <td className="py-3 font-medium text-white">bob.smith</td>
                                    <td className="py-3 text-silver">HR_Records</td>
                                    <td className="py-3"><span className="text-coral-red font-bold">DENY</span></td>
                                    <td className="py-3 text-silver">88.5%</td>
                                </tr>
                            </tbody>
                        </table>
                    </CardContent>
                </Card>
            </div>
        </DashboardLayout>
    );
}
