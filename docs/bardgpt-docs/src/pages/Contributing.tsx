import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";

export default function Contributing() {
  return (
    <DocLayout>
      <Breadcrumb items={[{ label: "Contribute" }]} />
      <div className="doc-prose">
        <h1>Contributing to BardGPT</h1>
        <p className="text-lg text-muted-foreground">Guidelines for contributing to the BardGPT project.</p>
        <h2>Getting Started</h2>
        <ol><li>Fork the repository on GitHub</li><li>Clone your fork locally</li><li>Create a new branch for your feature or fix</li><li>Make your changes with clear commit messages</li><li>Push to your fork and submit a pull request</li></ol>
        <h2>Code Standards</h2>
        <ul><li>Follow PEP 8 style guidelines for Python code</li><li>Add docstrings to all functions and classes</li><li>Include type hints where possible</li><li>Write tests for new functionality</li></ul>
        <h2>Pull Request Process</h2>
        <ol><li>Ensure your code passes all existing tests</li><li>Update documentation for any changed functionality</li><li>Add a clear description of changes in your PR</li><li>Link any related issues</li></ol>
        <h2>Reporting Issues</h2>
        <p>When reporting issues, please include:</p>
        <ul><li>A clear, descriptive title</li><li>Steps to reproduce the issue</li><li>Expected vs actual behavior</li><li>Your environment (Python version, OS, etc.)</li></ul>
        <p>Visit the <a href="https://github.com/Himanshu7921/BardGPT" target="_blank" rel="noopener noreferrer">GitHub repository</a> to get started.</p>
      </div>
    </DocLayout>
  );
}
