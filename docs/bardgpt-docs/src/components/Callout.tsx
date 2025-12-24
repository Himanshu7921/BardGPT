import { Info, AlertTriangle, Lightbulb } from "lucide-react";
import { cn } from "@/lib/utils";

type CalloutType = "info" | "warning" | "tip";

interface CalloutProps {
  type?: CalloutType;
  title?: string;
  children: React.ReactNode;
}

const icons = {
  info: Info,
  warning: AlertTriangle,
  tip: Lightbulb,
};

const styles = {
  info: "callout-info",
  warning: "callout-warning",
  tip: "callout-tip",
};

const iconStyles = {
  info: "text-primary",
  warning: "text-warning",
  tip: "text-success",
};

export function Callout({ type = "info", title, children }: CalloutProps) {
  const Icon = icons[type];

  return (
    <div className={cn("callout", styles[type])}>
      <div className="flex gap-3">
        <Icon className={cn("w-5 h-5 shrink-0 mt-0.5", iconStyles[type])} />
        <div className="flex-1">
          {title && (
            <p className="font-medium mb-1">{title}</p>
          )}
          <div className="text-sm text-foreground/80">{children}</div>
        </div>
      </div>
    </div>
  );
}
