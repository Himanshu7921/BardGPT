import { Link, useLocation } from "react-router-dom";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface NavItem {
  title: string;
  href?: string;
  items?: NavItem[];
}

const sidebarNav: NavItem[] = [
  {
    title: "Getting Started",
    items: [
      { title: "Introduction", href: "/docs" },
      { title: "Quick Start", href: "/docs/quickstart" },
    ],
  },
  {
    title: "Core Concepts",
    items: [
      { title: "Architecture", href: "/docs/architecture" },
      { title: "Configuration", href: "/docs/config" },
      { title: "Training", href: "/docs/training" },
      { title: "Sampling", href: "/docs/sampling" },
    ],
  },
  {
    title: "API Reference",
    items: [
      { title: "transformer.model", href: "/docs/api/model" },
      { title: "transformer.attention", href: "/docs/api/attention" },
      { title: "transformer.block", href: "/docs/api/block" },
      { title: "transformer.embedding", href: "/docs/api/embedding" },
      { title: "transformer.utils", href: "/docs/api/utils" },
      { title: "transformer.train", href: "/docs/api/train" },
      { title: "transformer.sample", href: "/docs/api/sample" },
    ],
  },
  {
    title: "Community",
    items: [
      { title: "Contributing", href: "/contribute" },
      { title: "Code of Conduct", href: "/legal/code-of-conduct" },
      { title: "Security", href: "/legal/security" },
    ],
  },
  {
    title: "Legal",
    items: [
      { title: "License", href: "/legal/license" },
      { title: "Citation", href: "/legal/citation" },
    ],
  },
];

function NavGroup({ item }: { item: NavItem }) {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(true);
  
  const hasActiveChild = item.items?.some(
    (child) => child.href && location.pathname === child.href
  );

  return (
    <div className="mb-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full sidebar-group-title mb-2 hover:text-foreground transition-colors"
      >
        <span>{item.title}</span>
        {isOpen ? (
          <ChevronDown className="w-3 h-3" />
        ) : (
          <ChevronRight className="w-3 h-3" />
        )}
      </button>
      {isOpen && item.items && (
        <div className="space-y-0.5 animate-slide-in">
          {item.items.map((subItem) => (
            <Link
              key={subItem.href}
              to={subItem.href || "#"}
              className={cn(
                "sidebar-link",
                location.pathname === subItem.href && "active"
              )}
            >
              {subItem.title}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export function DocSidebar() {
  return (
    <aside className="w-64 shrink-0 border-r border-border bg-sidebar hidden lg:block">
      <div className="sticky top-14 h-[calc(100vh-3.5rem)] overflow-y-auto py-6 px-4">
        <nav>
          {sidebarNav.map((item) => (
            <NavGroup key={item.title} item={item} />
          ))}
        </nav>
      </div>
    </aside>
  );
}
