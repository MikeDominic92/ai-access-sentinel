import React from 'react';
import { cn } from '@/utils/cn';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
    variant?: 'default' | 'outline' | 'success' | 'warning' | 'danger' | 'info';
    size?: 'sm' | 'md';
}

export function Badge({
    className,
    variant = 'default',
    size = 'md',
    children,
    ...props
}: BadgeProps) {
    const variants = {
        default: 'bg-slate-gray text-white border-white/10',
        outline: 'border border-white/20 text-silver',
        success: 'bg-emerald-green/10 text-emerald-green border-emerald-green/20',
        warning: 'bg-amber-gold/10 text-amber-gold border-amber-gold/20',
        danger: 'bg-coral-red/10 text-coral-red border-coral-red/20',
        info: 'bg-electric-cyan/10 text-electric-cyan border-electric-cyan/20',
    };

    const sizes = {
        sm: 'text-[10px] px-2 py-0.5',
        md: 'text-xs px-2.5 py-0.5',
    };

    return (
        <span
            className={cn(
                'inline-flex items-center rounded-full border font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
                variants[variant],
                sizes[size],
                className
            )}
            {...props}
        >
            {children}
        </span>
    );
}
