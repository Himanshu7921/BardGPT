# Contributing to BardGPT

Thank you for your interest in contributing to BardGPT.  
This project welcomes high-quality contributions from researchers, practitioners, and developers interested in Transformer architectures and language modeling.

The guidelines below describe the process for contributing code, documentation, and ideas.

---

## 1. Code of Conduct

Participation in this project requires adherence to the  
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Please ensure respectful, collaborative behavior at all times.

---

## 2. How to Contribute

### 2.1 Reporting Issues

Before opening an issue:

1. Check existing issues to avoid duplicates.
2. Use the official issue template.
3. Provide:
   - clear description
   - steps to reproduce (if applicable)
   - environment details
   - logs, stack traces, or screenshots

---

### 2.2 Submitting Pull Requests

Pull requests (PRs) must follow these rules:

1. **Fork the repository** and create a new branch:
```
git checkout -b feature/my-improvement
```


2. Ensure your code adheres to:
- clear structure
- readable naming
- modular design
- minimal side effects
- PEP8 standards (for Python)
- consistent style with the project

3. Verify your changes do not break existing functionality.

4. Update or add documentation if your changes introduce new concepts or APIs.

5. Commit messages must be clean, concise, and descriptive.  
Example:
feat: add rotary positional embeddings to attention layer
fix: correct dataset path resolution in train.py
docs: update sampling instructions for root-level execution


6. Submit the PR with a clear description:
- What was changed
- Why it was changed
- How it was implemented
- Any benchmarks or evaluations (if applicable)

7. Be responsive to review comments.

---

## 3. Development Setup

Install dependencies:
```
pip install -r requirements.txt
```

Run training (from repository root):
```
python transformer/train.py
```

Run sampling (from repository root):
```
python transformer/sample.py
```


---

## 4. Style Guidelines

### Python Code

- Prefer simple, explicit implementations over overly abstracted designs.
- Use type hints when reasonable.
- Follow PEP8 formatting.
- Document classes and methods using clear docstrings.
- Avoid unused imports, dead code, or inconsistent naming.
- Keep functions short and well-scoped.

### Git Guidelines

- Create topic branches for work.
- Squash commits before merging if needed.
- Write commit messages with one of these prefixes:
  - `feat:` (new features)
  - `fix:` (bug fixes)
  - `docs:` (documentation)
  - `refactor:` (internal improvements)
  - `chore:` (maintenance)

---

## 5. Adding New Features

If introducing a significant new feature (e.g., FlashAttention, LoRA, new tokenizers):

1. Open an issue first to discuss design.
2. Provide a minimal prototype or pseudo-code.
3. Ensure backward compatibility where possible.
4. Include benchmarking results if affecting performance.

---

## 6. Tests and Validation

Before finalizing contributions:

- Run small-scale training to verify model convergence.
- If modifying core components (attention, embedding, scheduler), add tests or reproducible scripts.
- Validate that sampling still produces coherent Shakespeare-like text.

---

## 7. Documentation Updates

If your contribution affects:
- configuration,
- training,
- sampling,
- checkpoints,
- internal architecture,

please update the README accordingly.

---

## 8. Community and Support

For questions, discussions, or research dialogue, open a GitHub Discussion thread or issue.

---


Thank you for contributing to BardGPT.
Your contributions help advance open research on transparent, reproducible Transformer architectures.