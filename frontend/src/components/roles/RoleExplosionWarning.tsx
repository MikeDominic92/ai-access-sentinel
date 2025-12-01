'use client';

import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

export function RoleExplosionWarning() {
    return (
        <div className="bg-amber-gold/10 border border-amber-gold/20 rounded-lg p-4 flex items-start justify-between animate-in slide-in-from-top-2">
            <div className="flex items-start space-x-3">
                <AlertTriangle className="text-amber-gold mt-0.5" size={20} />
                <div>
                    <h4 className="font-bold text-white text-sm">Role Explosion Detected</h4>
                    <p className="text-sm text-silver mt-1">
                        We've detected 15 new micro-roles created in the last 7 days. This may indicate permission fragmentation.
                        <button className="text-electric-cyan hover:underline ml-2">Review Analysis</button>
                    </p>
                </div>
            </div>
            <button className="text-silver hover:text-white">
                <X size={16} />
            </button>
        </div>
    );
}
