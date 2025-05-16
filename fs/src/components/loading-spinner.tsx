import { cva, VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils" 

const spinnerVariants = cva(
  "border-4 rounded-full border-gray-300 border-t-gray-900 animate-spin",
  {
    variants: {
      size: {
        sm: "w-4 h-4 border-2",
        md: "w-6 h-6 border-4",
        lg: "w-8 h-8 border-4",
      },
    },
    defaultVariants: {
      size: "md",
    },
  }
)

interface LoadingSpinnerProps extends VariantProps<typeof spinnerVariants> {
  className?: string
}

export const LoadingSpinner = ({ size, className }: LoadingSpinnerProps) => {
  return (
    <div className="flex justify-center items-center">
      <div className={cn(spinnerVariants({ size }), className)} />
    </div>
  )
}
