import React from 'react';
import { cn } from '@/utils/cn';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg' | 'icon';
    isLoading?: boolean;
}

export function Button({
    className,
    variant = 'primary',
    size = 'md',
    isLoading = false,
    children,
    disabled,
    ...props
}: ButtonProps) {
    const variants = {
        primary: 'bg-electric-cyan text-deep-navy hover:bg-electric-cyan/90 shadow-[0_0_15px_rgba(0,217,255,0.3)]',
        secondary: 'bg-slate-gray text-white hover:bg-slate-gray/80 border border-white/10',
        outline: 'border border-electric-cyan text-electric-cyan hover:bg-electric-cyan/10',
        ghost: 'hover:bg-white/10 text-silver hover:text-white',
        danger: 'bg-coral-red text-white hover:bg-coral-red/90 shadow-[0_0_15px_rgba(255,71,87,0.3)]',
    };

    const sizes = {
        sm: 'h-8 px-3 text-xs',
        md: 'h-10 px-4 py-2',
        lg: 'h-12 px-8 text-lg',
        icon: 'h-10 w-10',
    };

    return (
        <button
            className={cn(
                'inline-flex items-center justify-center rounded-lg font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-electric-cyan disabled:pointer-events-none disabled:opacity-50',
                variants[variant],
                sizes[size],
                className
            )}
            disabled={disabled || isLoading}
            {...props}
        >
            {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {children}
        </button>
    );
}
