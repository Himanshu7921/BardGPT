import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";

export default function License() {
  return (
    <DocLayout>
      <Breadcrumb items={[{ label: "Legal", href: "/legal/license" }, { label: "License" }]} />
      <div className="doc-prose">
        <h1>MIT License</h1>
        <p className="text-lg text-muted-foreground mb-8">BardGPT is open source software licensed under the MIT License.</p>
        <div className="bg-card border border-border rounded-lg p-6 font-mono text-sm leading-relaxed">
          <p className="mb-4">MIT License</p>
          <p className="mb-4">Copyright (c) 2024 Himanshu7921</p>
          <p className="mb-4">Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:</p>
          <p className="mb-4">The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.</p>
          <p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</p>
        </div>
      </div>
    </DocLayout>
  );
}
