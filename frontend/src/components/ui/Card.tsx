import React from 'react';
import { cn } from '@/utils/cn';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'glass' | 'glass-hover';
    noPadding?: boolean;
}

export function Card({
    className,
    variant = 'glass',
    noPadding = false,
    children,
    ...props
}: CardProps) {
    const variants = {
        default: 'bg-slate-gray border border-white/10',
        glass: 'glass-panel',
        'glass-hover': 'glass-card',
    };

    return (
        <div
            className={cn(
                'rounded-xl overflow-hidden',
                variants[variant],
                !noPadding && 'p-6',
                className
            )}
            {...props}
        >
            {children}
        </div>
    );
}

export function CardHeader({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
    return (
        <div className={cn('flex flex-col space-y-1.5 p-6', className)} {...props}>
            {children}
        </div>
    );
}

export function CardTitle({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
    return (
        <h3 className={cn('font-bold leading-none tracking-tight text-lg', className)} {...props}>
            {children}
        </h3>
    );
}

export function CardContent({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
    return (
        <div className={cn('p-6 pt-0', className)} {...props}>
            {children}
        </div>
    );
}
