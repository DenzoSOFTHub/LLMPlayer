# LLMPlayer Fine-Tuning Pipeline

Pure Java LoRA fine-tuning pipeline integrated in LLMPlayer. From raw data to a fine-tuned GGUF model in a single command, zero external dependencies.

## Overview

The pipeline takes raw data (source code, documents, or structured data), generates a training dataset using a generator LLM, fine-tunes a target model with LoRA adapters, and exports the result as a ready-to-use GGUF file.

```
Raw Data → Parse & Chunk → Generate Q&A → Train LoRA → Merge → Export GGUF
             (Stage 2)      (Stage 3)     (Stage 4)  (Stage 5)  (Stage 6)
```

The entire process is checkpointed — it can be suspended at any point (Ctrl+C) and resumed from where it left off.

---

## Quick Start

```bash
# Fine-tune on Java source code
./run.sh --fine-tune \
  --source ./src/main/java \
  --target-model gguf/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf \
  --output gguf/MyProject-Coder-3B-Q4_K_M.gguf

# Fine-tune on documents
./run.sh --fine-tune \
  --documents ./docs \
  --target-model gguf/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --output gguf/MyDocs-3B-Q4_K_M.gguf

# Fine-tune on database export
./run.sh --fine-tune \
  --data export.csv --schema schema.sql \
  --target-model gguf/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf \
  --output gguf/MyData-3B-Q4_K_M.gguf
```

When `--generator-model` is not specified, the target model is used as its own generator (faster, simpler, single model load). For higher quality dataset generation, specify a larger model:

```bash
./run.sh --fine-tune \
  --source ./src/main/java \
  --generator-model gguf/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --target-model gguf/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf \
  --output gguf/MyProject-Coder-3B-Q4_K_M.gguf
```

---

## Three Scenarios

### Scenario 1: Source Code

**Goal:** The fine-tuned model learns to understand, explain, navigate, and modify your codebase.

**Input:** A directory of Java source files (or any programming language — the parser extracts text).

**How it works:**
1. The source parser scans `.java` files and splits them into logical units: classes, methods, and cross-references
2. Each chunk includes the code plus contextual metadata (package, class name, imports)
3. The generator LLM produces Q&A pairs covering:
   - **Explanation:** "What does this class/method do?"
   - **Parameters:** "What are the inputs and outputs of method X?"
   - **Flow:** "How does X call Y? What's the execution path?"
   - **Bugs:** "Are there potential issues in this code?"
   - **Improvement:** "How could this method be refactored?"

**Recommended models:**

| Role | Model | Why |
|------|-------|-----|
| Generator | Qwen2.5-Coder-7B-Instruct | Code-specialized, understands Java structures |
| Target | Qwen2.5-Coder-3B-Instruct | Code-specialized, QLoRA fits in 6 GB VRAM |

**Example generated Q&A pair:**
```
Q: Explain the purpose of the TensorFactory.create() method and how it selects GPU vs CPU tensors.
A: TensorFactory.create() is a factory method that creates the appropriate FloatTensor subclass
   based on the GGMLType. It first checks if a GpuBufferManager is registered — if so, it
   attempts to create a GPU-accelerated tensor variant (e.g., Q4_KGpuTensor). If GPU creation
   fails or no manager is registered, it falls back to the CPU variant (e.g., Q4_KFloatTensor).
   This allows transparent GPU/CPU selection during model loading.
```

### Scenario 2: Documents

**Goal:** The fine-tuned model becomes a domain expert on your document corpus — answers questions, summarizes, and reasons about the content.

**Input:** A directory containing documents in supported formats:
- **Text:** `.txt`, `.md`, `.csv`, `.tsv`
- **Office:** `.docx`, `.xlsx` (ZIP-based XML, parsed without external libraries)
- **PDF:** `.pdf` (basic text extraction from content streams)

**How it works:**
1. Each document is parsed to extract plain text with structural metadata (title, sections, pages)
2. Text is chunked by sections/paragraphs with configurable overlap (default: 100 tokens) to preserve context across boundaries
3. The generator LLM produces Q&A pairs covering:
   - **Factual:** "What is the deadline specified in section 4.2?"
   - **Summary:** "Summarize the key points of chapter 3"
   - **Inferential:** "What are the implications of clause 7?"
   - **Comparative:** "Compare the specifications of product A and B"
   - **Procedural:** "What are the steps to complete process X?"

**Recommended models:**

| Role | Model | Why |
|------|-------|-----|
| Generator | Qwen2.5-7B-Instruct | Excellent comprehension, multilingual |
| Target | Qwen2.5-3B-Instruct | Good balance of quality and size, multilingual |

### Scenario 3: Structured Data

**Goal:** The fine-tuned model understands your database schema and data, answers analytical questions, and optionally generates SQL queries.

**Input:** Exported data (CSV or JSON) plus an optional schema definition (SQL DDL).

**How it works:**
1. The schema is parsed to understand table structures, column types, and relationships
2. Data is sampled and aggregated to create statistical summaries (row counts, value distributions, min/max/avg for numeric columns)
3. Chunks combine schema + data samples + statistics
4. The generator LLM produces Q&A pairs at three levels:
   - **Schema:** "What tables are in the database? What are the relationships?"
   - **Analytical:** "How many customers are in Milan? What's the revenue trend?"
   - **SQL generation:** "Show me the top 10 customers by revenue" → SQL query

**Recommended models:**

| Role | Model | Why |
|------|-------|-----|
| Generator | Qwen2.5-Coder-7B-Instruct | Strong at structured reasoning and SQL |
| Target | Qwen2.5-Coder-3B-Instruct | Understands SQL and data patterns |

---

## Pipeline Stages

### Stage 1: Analyze Target

Quick-parses the target GGUF to extract:
- Architecture → determines chat template format for training data
- Tokenizer vocabulary → accurate token counting for chunk sizing
- Context length → maximum training sequence length
- Embedding size → complexity calibration (smaller models get simpler Q&A)

No model weights are loaded — only metadata and tokenizer.

### Stage 2: Parse & Chunk

Parses input data and splits into chunks that fit within the target model's context budget:

```
context_budget = context_length - template_overhead
max_chunk_tokens = context_budget × 0.45
max_response_tokens = context_budget × 0.45
```

Token counting uses the target model's tokenizer for accuracy.

### Stage 3: Generate Dataset

Loads the generator model and produces Q&A pairs for each chunk:

1. Builds a meta-prompt appropriate to the data type (code/document/data)
2. Adjusts response complexity based on target model capacity
3. Generates N pairs per chunk (default: 5)
4. Formats each pair in the target model's chat template
5. Validates that each training example fits in the target context
6. Saves to JSONL with progressive checkpointing

This is typically the longest stage. Progress is checkpointed after each chunk — interrupting and resuming skips already-generated pairs.

**Optimization:** When no separate generator model is specified, the target model generates its own training data (no model swap overhead). This is faster but may produce lower-quality pairs for smaller models.

### Stage 4: Train LoRA

Loads the target model and trains LoRA adapters:

1. Injects LoRA matrices (A: rank×dim, B: dim×rank) at each target layer
2. For each training example:
   - Forward pass through full model with LoRA path
   - Cross-entropy loss against target tokens
   - Backward pass to compute gradients for LoRA parameters only
   - AdamW optimizer step
3. Checkpoints after each epoch

**Training modes:**
- **QLoRA (default when target is quantized):** Base weights stay quantized in memory. LoRA adapters are F32. Lowest memory usage.
- **Full-precision LoRA (when target is F16/F32):** Higher quality gradients, more memory.

The system auto-selects based on the target model's quantization format.

### Stage 5: Merge

Combines trained LoRA weights into the base model:

1. For each layer with LoRA adapters:
   - Dequantize base weight tensor W → F32
   - Compute LoRA contribution: delta = B × A (scaled by alpha/rank)
   - Merge: W' = W + delta
   - Re-quantize W' to target format
2. Non-LoRA tensors (embeddings, norms, output head) pass through unchanged

### Stage 6: Export GGUF

Writes the final GGUF file:

1. GGUF header with magic number, version, tensor/metadata counts
2. All metadata from original target (architecture, tokenizer, hyperparameters)
3. Updated model name in metadata
4. Quantized tensor data aligned to 32 bytes
5. Integrity verification

---

## CLI Reference

### Required flags

| Flag | Description |
|------|-------------|
| `--fine-tune` | Activate the fine-tuning pipeline |
| `--target-model <path>` | GGUF of the model to fine-tune |
| `--output <path>` | Output path for the fine-tuned GGUF |

Plus one of:

| Flag | Description |
|------|-------------|
| `--source <path>` | Directory of source code files (scenario 1) |
| `--documents <path>` | Directory of documents (scenario 2) |
| `--data <path>` | CSV/JSON data file (scenario 3) |

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--generator-model <path>` | same as target | GGUF of the model that generates Q&A pairs |
| `--type <type>` | auto-detect | `code`, `document`, `structured` |
| `--schema <path>` | — | SQL DDL file for structured data scenario |
| `--quantization <fmt>` | same as target | Output format: Q4_K_M, Q5_K_M, Q8_0, F16 |
| `--lora-rank <n>` | 16 | LoRA rank (higher = more capacity, more memory) |
| `--lora-alpha <n>` | 32 | LoRA scaling factor |
| `--epochs <n>` | 3 | Number of training epochs |
| `--learning-rate <f>` | 2e-4 | AdamW learning rate |
| `--pairs-per-chunk <n>` | 5 | Q&A pairs to generate per chunk |
| `--chunk-overlap <n>` | 100 | Token overlap between adjacent chunks |
| `--work-dir <path>` | `work/` | Working directory for checkpoints |
| `--dataset-only` | false | Run only stages 1-3 (generate dataset, skip training) |
| `--train-dataset <path>` | — | Skip stages 1-3, train from existing JSONL dataset |

### Suspend and resume

The pipeline checkpoints progress after each significant unit of work:
- Stage 3: after each chunk (Q&A generation)
- Stage 4: after each epoch (training)

To suspend, press **Ctrl+C**. The current state is saved to `work/pipeline.json`.

To resume, run the exact same command again:
```bash
./run.sh --fine-tune --source ./src --target-model ... --output ...
# Output: "Resuming from stage 3, chunk 85/142..."
```

The pipeline detects the checkpoint and resumes from the last completed step.

To force a fresh start, delete the work directory or use a different `--work-dir`.

---

## Estimated Times

On Intel Core Ultra 7 155H + NVIDIA RTX 4050 (6 GB VRAM):

### Dataset generation (Stage 3)

| Generator model | tok/s | Time per chunk | 100 chunks | 500 chunks |
|----------------|-------|----------------|------------|------------|
| 3B Q4_K_M (GPU) | ~5.5 | ~3 min | ~5h | ~25h |
| 7B Q4_K_M (partial GPU) | ~2.5 | ~7 min | ~12h | ~58h |

### Training (Stage 4)

| Configuration | Time per example | 1000 ex × 3 epochs | 5000 ex × 3 epochs |
|--------------|-----------------|---------------------|---------------------|
| QLoRA 3B (GPU) | ~3-4 sec | ~3h | ~15h |
| QLoRA 3B (CPU) | ~8-10 sec | ~8h | ~42h |

### Typical end-to-end times

| Scenario | Input size | Dataset pairs | Training | Total |
|----------|-----------|---------------|----------|-------|
| Java project (100 classes) | 120 chunks | ~600 pairs | QLoRA 3B GPU | ~16h |
| Documents (200 pages) | 200 chunks | ~1000 pairs | QLoRA 3B GPU | ~26h |
| Database (10 tables) | 70 chunks | ~350 pairs | QLoRA 3B GPU | ~10h |

---

## Memory Requirements

### RTX 4050 (6 GB VRAM)

| Configuration | Fits in VRAM? | Notes |
|---------------|---------------|-------|
| QLoRA 3B (Q4_K_M base) | Yes (~4 GB) | Recommended default |
| QLoRA 7B (Q4_K_M base) | No | Use CPU or --no-gpu |
| Full LoRA 3B (F16 base) | No | Use CPU, needs ~16 GB RAM |

### RAM requirements (CPU training)

| Configuration | RAM needed |
|---------------|-----------|
| QLoRA 3B | ~6 GB |
| QLoRA 7B | ~12 GB |
| Full LoRA 3B (F16) | ~16 GB |

---

## Advanced Usage

### Generate dataset only (for inspection/editing)

```bash
./run.sh --fine-tune --source ./src \
  --target-model gguf/model.gguf \
  --output gguf/out.gguf \
  --dataset-only
# Output: work/dataset.jsonl — review and edit manually
```

### Train from existing dataset

```bash
./run.sh --fine-tune \
  --train-dataset work/dataset.jsonl \
  --target-model gguf/model.gguf \
  --output gguf/out.gguf
# Skips stages 1-3, goes directly to training
```

### Custom LoRA configuration

```bash
./run.sh --fine-tune --source ./src \
  --target-model gguf/model.gguf \
  --output gguf/out.gguf \
  --lora-rank 32 \
  --lora-alpha 64 \
  --epochs 5 \
  --learning-rate 1e-4
```

### Different output quantization

```bash
# Train on Q4_K_M model, output as Q5_K_M (higher quality)
./run.sh --fine-tune --source ./src \
  --target-model gguf/model-Q4_K_M.gguf \
  --output gguf/out-Q5_K_M.gguf \
  --quantization Q5_K_M
```
