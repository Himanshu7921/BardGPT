# **BardGPT Documentation Website**

This folder contains the **complete documentation website** for the BardGPT project.
It is a production-grade, fully structured, SEO-optimized documentation system designed to mirror the quality and clarity of modern AI/ML frameworks such as **PyTorch**, **OpenAI Docs**, and **HuggingFace**.

The documentation site includes:

* A polished Home page
* Detailed technical docs
* Full API references (functions, classes, modules)
* Architecture explanations
* Training and sampling guides
* Legal & governance pages
* Contribution guidelines
* Direct references to the BardGPT GitHub repository

The site is meant to serve both **researchers** and **developers** working with the BardGPT codebase.

---

## **Directory Purpose**

This directory is **isolated from the core BardGPT source code**, ensuring documentation evolves cleanly alongside the Transformer implementation.

---

## **What This Documentation Website Includes**

The site covers **100%** of the BardGPT repository:

### **1. Overview & Architecture**

* Introduction
* Model philosophy
* Architecture diagram
* Embedding → Decoder → Projection pipeline
* Weight tying explanation
* Positional embeddings
* Causal self-attention flow
* Shapes & tensor-level data flow

---

### **2. Comprehensive API Reference**

Every Python file inside `transformer/` is documented as a full API module:

#### `transformer.attention`

* `MultiHeadAttention`
* Attention scoring
* Masking
* Shape rules
* Dropout handling

#### `transformer.block`

* FeedForward
* LayerNorm
* Residual connections (Pre-LN)
* DecoderBlock
* Decoder

#### `transformer.embedding`

* Token embeddings
* Fixed positional embeddings
* Learned positional embeddings
* EmbeddingBlock

#### `transformer.model`

* `DecoderOnlyTransformerModel`
* Embeddings → Decoder → Projection
* Weight tying mechanism

#### `transformer.utils`

* `TransformerConfig`
* DataLoader
* causal_mask
* generate
* checkpoint loader/saver
* top-k / top-p filters
* model summary printer

#### `transformer.train`

* Training loop
* Warmup + cosine LR scheduler
* Early stopping
* Gradient clipping
* TensorBoard logging

#### `transformer.sample`

* Inference pipeline
* Temperature
* top-k / top-p sampling
* Sliding window decoding

---

### **3. Dataset Documentation**

* Tiny Shakespeare dataset description
* Sliding window batching
* Vocabulary extraction
* Train/val split strategy

---

### **4. Training Guide**

* How to train
* How LR scheduling works
* Checkpointing strategy
* Best practices for reproducibility

---

### **5. Sampling Guide**

* Sampling parameters
* Practical examples
* Expected outputs

---

### **6. Legal & Governance Pages**

Automatically extracted and formatted from:

* LICENSE (MIT)
* CODE_OF_CONDUCT.md
* CONTRIBUTING.md
* SECURITY.md
* CITATION.cff
* ISSUE_TEMPLATE

---

## **Website Architecture**

The documentation website uses a **modern docs interface**:

### **Header Navigation**

* Home
* Docs
* API Reference
* Model Architecture
* Training
* Sampling
* Contribute
* License
* GitHub

### **Sidebar Structure**

* Introduction
* Architecture
* Configuration
* Training
* Sampling
* API Reference

  * attention
  * block
  * embedding
  * model
  * utils
  * train
  * sample

Each page includes:

* A right-side Table of Contents
* Breadcrumbs
* Clean typography
* Developer-friendly code blocks

---

## **Related Links**

* **Main Repository:** [https://github.com/Himanshu7921/BardGPT](https://github.com/Himanshu7921/BardGPT)
* **Issues:** See repository Issue Template
* **Contributions:** Refer to CONTRIBUTING.md
* **Security Policy:** SECURITY.md
