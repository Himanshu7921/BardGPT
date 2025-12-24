import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";

export default function Security() {
  return (
    <DocLayout>
      <Breadcrumb items={[{ label: "Legal" }, { label: "Security" }]} />
      <div className="doc-prose">
        <h1>Security Policy</h1>
        <p className="text-lg text-muted-foreground">How to report security vulnerabilities.</p>
        <h2>Reporting a Vulnerability</h2>
        <p>If you discover a security vulnerability, please report it responsibly by emailing the maintainers directly rather than opening a public issue.</p>
        <h2>Response Timeline</h2>
        <ul><li>We will acknowledge receipt within 48 hours</li><li>We will provide an initial assessment within 7 days</li><li>We will work with you to understand and resolve the issue</li></ul>
        <h2>Scope</h2>
        <p>This security policy applies to the BardGPT repository and its official distributions.</p>
      </div>
    </DocLayout>
  );
}
