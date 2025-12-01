'use client';

import React from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { CheckCircle, XCircle } from 'lucide-react';
import { motion } from 'framer-motion';

interface PredictionResultProps {
    result: 'approve' | 'deny' | null;
}

export function PredictionResult({ result }: PredictionResultProps) {
    if (!result) return null;

    const isApprove = result === 'approve';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <Card variant="glass" className={`border-2 ${isApprove ? 'border-emerald-green/50 bg-emerald-green/5' : 'border-coral-red/50 bg-coral-red/5'}`}>
                <CardContent className="p-8 flex flex-col items-center text-center">
                    <motion.div
                        initial={{ scale: 0.8 }}
                        animate={{ scale: 1 }}
                        transition={{ type: "spring", stiffness: 200, damping: 10 }}
                        className={`p-4 rounded-full mb-4 ${isApprove ? 'bg-emerald-green/20 text-emerald-green' : 'bg-coral-red/20 text-coral-red'}`}
                    >
                        {isApprove ? <CheckCircle size={64} /> : <XCircle size={64} />}
                    </motion.div>

                    <h2 className="text-3xl font-bold text-white mb-2">
                        {isApprove ? 'ACCESS APPROVED' : 'ACCESS DENIED'}
                    </h2>
                    <p className="text-silver mb-6">
                        Confidence Score: <span className="text-white font-bold">94.2%</span>
                    </p>

                    {/* Probability Bar */}
                    <div className="w-full max-w-md h-4 bg-slate-gray rounded-full overflow-hidden flex">
                        <div className="h-full bg-emerald-green transition-all duration-1000" style={{ width: isApprove ? '94%' : '12%' }}></div>
                        <div className="h-full bg-coral-red transition-all duration-1000" style={{ width: isApprove ? '6%' : '88%' }}></div>
                    </div>
                    <div className="flex justify-between w-full max-w-md mt-2 text-xs text-silver">
                        <span>Approve Probability</span>
                        <span>Deny Probability</span>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    );
}
