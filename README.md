<div align="center">
<br>

<h1>PhysMaster</h1>

<p><strong>Solve physics problems with LLM-driven Monte Carlo Tree Search</strong></p>

<p>
<a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>&nbsp;
<a href="#setup"><img src="https://img.shields.io/badge/API-OpenAI%20Compatible-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI Compatible"></a>&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" alt="MIT License"></a>
</p>

<p><a href="README_CN.md">中文文档</a></p>

<br>
</div>

PhysMaster decomposes a physics problem into subtasks, explores multiple solution strategies **in parallel** through an MCTS search tree, evaluates and refines them with a Critic, and distills reusable knowledge at every level &mdash; from a single node all the way up to a cross-task wisdom store.

---

## Table of Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Key Concepts](#key-concepts)
  - [MCTS Search](#-mcts-search)
  - [Memory Hierarchy (HCC)](#-memory-hierarchy-hcc)
  - [Prior Knowledge (RAG)](#-prior-knowledge-rag)
  - [Skills](#-skills)
  - [Workflow Templates](#-workflow-templates)
- [Project Structure](#project-structure)
- [Feishu Bot](#-feishu-bot)
- [License](#license)

---

## Architecture

Six specialized agents collaborate inside an MCTS loop:

```
 ╭──────────╮
 │  Query   │
 ╰────┬─────╯
      ▼
 ╭──────────╮     ╭────────────────────────────────────────────────╮
 │ Clarifier│────▶│              MCTS  Search  Loop                │
 ╰──────────╯     │                                                │
                  │   ┌────────────┐      ┌──────────────────┐     │
                  │   │ Supervisor │─────▶│  Theoretician(s) │     │
                  │   │  (dispatch)│      │  (solve, x N)    │     │
                  │   └─────▲──────┘      └────────┬─────────┘     │
                  │         │                      ▼               │
                  │   ┌─────┴──────┐      ┌──────────────────┐     │
                  │   │  Promoter  │◀─────│     Critic       │     │
                  │   │ (L1 → L2)  │      │   (evaluate)     │     │
                  │   └────────────┘      └──────────────────┘     │
                  │                                                │
                  │   select ── expand ── evaluate ── promote      │
                  │           backpropagate ── prune               │
                  ╰────────────────────┬───────────────────────────╯
                                       ▼
                                ╭────────────╮     ╭────────╮
                                │ Summarizer │────▶│ Report │
                                ╰────────────╯     ╰────────╯
```

| Agent | What it does |
|:------|:-------------|
| **Clarifier** | Parses the raw problem into a structured contract with subtasks |
| **Supervisor** | Reads the tree context (HCC), picks the next subtask, decides draft vs. revise |
| **Theoretician** | Solves a subtask &mdash; can call Python, skills, web search, and the prior knowledge base |
| **Critic** | Scores the solution (0&ndash;1) and returns a verdict: `complete` / `to_revise` / `to_redraft` |
| **Promoter** | Compresses raw experience (L1) into reusable knowledge (L2) on the tree |
| **Summarizer** | Extracts the best trajectory from the tree and writes a Markdown report |

> The loop stops when all subtasks are completed along some path, or the round budget runs out.

---

## Setup

> Requires **Python 3.10+**

```bash
git clone https://github.com/AdrianMiao27/PHY_Master.git
cd PHY_Master
pip install -r requirements.txt
```

Then fill in `config.yaml` with your LLM endpoint (any OpenAI-compatible API):

```yaml
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  model: "gpt-4o"
```

---

## Quick Start

**1. Write a problem** &mdash; create a plain text file (LaTeX is fine):

```
instructions/my_problem.txt
```

**2. Point the config at it:**

```yaml
pipeline:
  query_file: "instructions/my_problem.txt"
```

**3. Run:**

```bash
python run.py                    # default config.yaml
python run.py -c custom.yaml     # custom config
```

**4. Check results** in `outputs/<task_name>/`:

```
outputs/<task_name>/
 ├─ contract.json            Structured problem decomposition
 ├─ node_1/                  Theoretician output for MCTS node 1
 ├─ node_2/                  ...
 ├─ summary.md               Final solution report
 └─ visualization.html       Interactive MCTS tree (open in browser)
```

> **Minimal mode** &mdash; to run without any external knowledge, set `skills.enabled: false` and all `landau.*_enabled: false` in the config.

---

## Configuration Reference

All behavior lives in `config.yaml`. Only `llm` and `pipeline.query_file` are required; everything else has defaults.

<details>
<summary><b>Full config with comments</b> (click to expand)</summary>

```yaml
# ── LLM ──────────────────────────────────────────────
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  model: "gpt-4o"

# ── Pipeline ─────────────────────────────────────────
pipeline:
  query_file: "instructions/test.txt"
  output_path: "outputs"
  max_rounds: 10              # MCTS round budget
  parallel_processes: 2       # concurrent Theoretician workers

# ── MCTS ─────────────────────────────────────────────
mcts:
  draft_expansion: 2          # children per draft expansion
  revise_expansion: 1         # children per revise expansion
  exploration_constant: 1.414 # UCB1 exploration weight (sqrt 2)
  active_beam_width: 0        # 0 = no pruning; N = keep top-N per depth

# ── Clarifier ────────────────────────────────────────
clarifier:
  max_key_concpets: 5

# ── Skills ───────────────────────────────────────────
skills:
  enabled: true
  roots:
    - "LANDAU/skills"

# ── LANDAU knowledge modules ─────────────────────────
landau:
  library_enabled: true       # MCP web search & parse
  library: "LANDAU/library"
  workflow_enabled: true       # problem-solving templates
  workflow: "LANDAU/workflow"
  prior_enabled: true          # FAISS RAG knowledge base
  prior: "LANDAU/prior"
  wisdom_save_enabled: false   # persist L3 wisdom after each task

# ── Visualization ────────────────────────────────────
visualization:
  enabled: true
```

</details>

| Key | Description | Default |
|:----|:------------|:--------|
| `pipeline.max_rounds` | Total MCTS iterations before forced stop | `10` |
| `pipeline.parallel_processes` | Theoretician subprocesses | `2` |
| `mcts.draft_expansion` | Child nodes per draft round | `2` |
| `mcts.revise_expansion` | Child nodes per revise round | `2` |
| `mcts.exploration_constant` | UCB1 exploration term | `1.414` |
| `mcts.active_beam_width` | Beam pruning width (0 = off) | `0` |

---

## Key Concepts

### 🌳 MCTS Search

PhysMaster does **not** solve linearly. It maintains a tree of solution attempts and navigates it like a game:

| Step | What happens |
|:-----|:-------------|
| **Select** | UCB1 picks the most promising leaf, balancing reward vs. exploration |
| **Expand** | N Theoretician workers spawn in parallel and produce child nodes |
| **Evaluate** | Critic scores each child on a 0&ndash;1 scale |
| **Promote** | Promoter compresses the raw trace into a concise knowledge string |
| **Backpropagate** | Reward flows upward; high-reward nodes (&gt;0.8) reinforce ancestors |
| **Prune** | If beam width is set, low-reward nodes beyond the budget are closed |

The search terminates when a complete path is found or `max_rounds` is hit. The best root-to-leaf path is extracted for the summary.

---

### 🧠 Memory Hierarchy (HCC)

Three levels, from ephemeral to persistent:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  L3  Wisdom          across tasks    → FAISS prior index        │
 │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
 │  L2  Knowledge       across tree     → node.knowledge           │
 │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
 │  L1  Experience      single node     → node.experience          │
 └──────────────────────────────────────────────────────────────────┘
```

- **L1 &mdash; Experience** &nbsp; Raw Theoretician output: full reasoning, tool calls, code. Stored per node.
- **L2 &mdash; Knowledge** &nbsp; Compressed by the Promoter after evaluation. Propagated through the tree context so descendant nodes see what ancestors learned. When a node scores &gt;0.8, its L2 is tagged onto ancestors during backpropagation (*cognitive reinforcement*).
- **L3 &mdash; Wisdom** &nbsp; Distilled after the entire task completes. Written back to the FAISS prior index so **future tasks** can retrieve it.

---

### 📚 Prior Knowledge (RAG)

`LANDAU/prior/` provides a full retrieval-augmented generation pipeline:

| Stage | File | What it does |
|:------|:-----|:-------------|
| **Ingest** | `prior_store.py` | PDF (via MinerU) / Markdown / Text &rarr; parent chunks (4k chars) &rarr; child chunks (1.2k chars) &rarr; `bge-small-en-v1.5` embeddings &rarr; FAISS `IndexFlatIP` |
| **Retrieve** | `prior_retrieve.py` | Dense search + BM25, fused with Reciprocal Rank Fusion, then a weighted reranker. Optional HyDE for better recall |
| **Wisdom** | `wisdom_store.py` | Post-task LLM distillation &rarr; new chunk appended to the same index |

```bash
# place source files in LANDAU/prior/source/, then:
python LANDAU/prior/prior_store.py                             # ingest all
python LANDAU/prior/prior_store.py --target path/to/file.pdf   # single file
python LANDAU/prior/prior_store.py --reset                     # full rebuild
```

Enable in config:

```yaml
landau:
  prior_enabled: true
  wisdom_save_enabled: true   # optional: persist L3 wisdom
```

---

### 🔧 Skills

Domain knowledge packages in `LANDAU/skills/`. Each skill is a folder with a `SKILL.md` (YAML frontmatter + methodology body). The Theoretician sees a brief of all installed skills and can load any on demand.

<details>
<summary><b>12 built-in skills</b></summary>

| Skill | Coverage |
|:------|:---------|
| Classical Electrodynamics | Maxwell's equations, radiation, waveguides |
| Quantum Mechanics | Schrodinger equation, scattering, angular momentum |
| Thermodynamics & Statistical Mechanics | Partition functions, phase transitions, ensembles |
| Conservation Laws | Noether's theorem, conserved currents |
| Perturbation Expansion | Regular/singular perturbation, asymptotic series |
| Variational Methods | Euler-Lagrange, Rayleigh-Ritz |
| Dimensional Analysis | Pi theorem, natural units, scaling laws |
| Symmetry Analysis | Group theory, Lie algebras, representations |
| Fourier & Spectral Analysis | Fourier/Laplace transforms, spectral methods |
| Numerical ODE/PDE | Runge-Kutta, finite difference/element |
| Statistical Error Analysis | Error propagation, fitting, Monte Carlo |
| LaMET Asymptotic Expansion | Large-momentum effective theory |

</details>

---

### 📋 Workflow Templates

YAML files in `LANDAU/workflow/` that define structured solving strategies. The Clarifier matches a template to the query by keyword overlap with its Goal field and uses it to produce a better subtask decomposition.

```yaml
landau:
  workflow_enabled: true
  workflow: "LANDAU/workflow"
```

---

## Project Structure

```
PHY_Master/
│
├── run.py                       Entry point
├── config.yaml                  Configuration
├── requirements.txt
│
├── core/                        Core pipeline
│   ├── clarifier.py               Query → structured contract
│   ├── supervisor.py              MCTS orchestrator
│   ├── mcts.py                    MCTSNode / MCTSTree
│   ├── theoretician.py            Solver agent (subprocess)
│   ├── summarizer.py              Trajectory → markdown report
│   └── visualization.py           Tree → interactive HTML
│
├── LANDAU/                      Knowledge modules
│   ├── skills/                    12 built-in physics skills
│   ├── workflow/                  Problem-solving YAML templates
│   ├── library/                   Web search & parse (MCP)
│   └── prior/                     FAISS RAG knowledge base
│       ├── prior_store.py           Ingestion pipeline
│       ├── prior_retrieve.py        Hybrid retriever
│       └── wisdom_store.py          L3 wisdom persistence
│
├── utils/                       Utilities
│   ├── llm_client.py              OpenAI-compatible API wrapper
│   ├── python_utils.py            Subprocess code execution
│   ├── skill_loader.py            SKILL.md discovery & loading
│   └── tool_schemas.py            Tool definitions
│
├── prompts/                     14 prompt templates (7 agents)
├── instructions/                Query files
├── feishu/                      Feishu bot integration
└── outputs/                     Generated at runtime
```

---

## 🤖 Feishu Bot

PhysMaster can run as a **Feishu (Lark) chatbot**. Send a physics problem in chat &rarr; bot replies "solving..." &rarr; pipeline runs in a background thread &rarr; summary is pushed back when done.

See **[feishu/README.md](feishu/README.md)** for setup instructions.

---

## License

[MIT](LICENSE)
