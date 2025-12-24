import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";

export default function Citation() {
  return (
    <DocLayout>
      <Breadcrumb items={[{ label: "Legal" }, { label: "Citation" }]} />
      <div className="doc-prose">
        <h1>Citation</h1>
        <p className="text-lg text-muted-foreground">How to cite BardGPT in your work.</p>
        <h2>BibTeX</h2>
        <CodeBlock language="bibtex" code={`@software{bardgpt2024,
  author = {Himanshu7921},
  title = {BardGPT: A Decoder-Only Transformer Implementation},
  year = {2025},
  url = {https://github.com/Himanshu7921/BardGPT},
  license = {MIT}
}`} />
        <h2>APA Format</h2>
        <p>Himanshu7921. (2025). BardGPT: A Decoder-Only Transformer Implementation [Computer software]. GitHub. https://github.com/Himanshu7921/BardGPT</p>
      </div>
    </DocLayout>
  );
}
