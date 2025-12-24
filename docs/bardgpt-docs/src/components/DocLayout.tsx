import { Header } from "./Header";
import { DocSidebar } from "./DocSidebar";
import { TableOfContents } from "./TableOfContents";

interface TOCItem {
  id: string;
  text: string;
  level: number;
}

interface DocLayoutProps {
  children: React.ReactNode;
  toc?: TOCItem[];
  showSidebar?: boolean;
  showToc?: boolean;
}

export function DocLayout({ 
  children, 
  toc = [], 
  showSidebar = true,
  showToc = true 
}: DocLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex-1 flex">
        {showSidebar && <DocSidebar />}
        <main className="flex-1 min-w-0">
          <div className="flex">
            <article className="flex-1 max-w-4xl mx-auto px-4 md:px-8 py-8">
              {children}
            </article>
            {showToc && toc.length > 0 && <TableOfContents items={toc} />}
          </div>
        </main>
      </div>
    </div>
  );
}
