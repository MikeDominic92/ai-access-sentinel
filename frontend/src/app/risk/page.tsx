'use client';

import React from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { UserProfileCard } from '@/components/risk/UserProfileCard';
import { RiskFactorBreakdown } from '@/components/risk/RiskFactorBreakdown';
import { RiskTimeline } from '@/components/risk/RiskTimeline';
import { AIRecommendations } from '@/components/risk/AIRecommendations';
import { Card, CardContent } from '@/components/ui/Card';
import { Search } from 'lucide-react';

export default function RiskScoringPage() {
    const mockUser = {
        name: 'Jane Doe',
        role: 'Senior DevOps Engineer',
        department: 'Engineering',
        location: 'San Francisco, US',
        email: 'jane.doe@company.com',
        riskScore: 73,
        riskLevel: 'HIGH' as const,
    };

    return (
        <DashboardLayout>
            {/* Search/Filter Bar */}
            <div className="flex items-center space-x-4 mb-6">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-silver h-4 w-4" />
                    <input
                        type="text"
                        placeholder="Search user by name, ID, or email..."
                        className="w-full pl-10 pr-4 py-2 bg-slate-gray/50 border border-white/10 rounded-lg text-white focus:outline-none focus:border-electric-cyan/50"
                    />
                </div>
                <div className="flex space-x-2">
                    <select className="bg-slate-gray/50 border border-white/10 rounded-lg px-4 py-2 text-white text-sm focus:outline-none">
                        <option>All Departments</option>
                        <option>Engineering</option>
                        <option>Sales</option>
                    </select>
                    <select className="bg-slate-gray/50 border border-white/10 rounded-lg px-4 py-2 text-white text-sm focus:outline-none">
                        <option>Risk Level: All</option>
                        <option>Critical</option>
                        <option>High</option>
                    </select>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: User Profile */}
                <div className="lg:col-span-1">
                    <UserProfileCard user={mockUser} />
                </div>

                {/* Right Column: Analytics & Actions */}
                <div className="lg:col-span-2 flex flex-col space-y-6">
                    {/* Top Row: Factors & Timeline */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[350px]">
                        <RiskFactorBreakdown />
                        <RiskTimeline />
                    </div>

                    {/* Bottom Row: Recommendations */}
                    <div className="flex-1">
                        <AIRecommendations />
                    </div>
                </div>
            </div>
        </DashboardLayout>
    );
}
