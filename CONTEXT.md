# Research and Web Intelligence Agent — Project Context (Wrapper)

> Important: Do not alter the original CONTEXT.md text; it appears verbatim below between ORIGINAL_CONTEXT_START and ORIGINAL_CONTEXT_END markers. All additions in Appendices come after the END marker and do not modify the original. [web:711]

---

## Original CONTEXT.md (verbatim; do not edit)

<!-- ORIGINAL_CONTEXT_START -->

# Research and Web Intelligence Agent — Context

## Project title
### Axiom
Research and Web Intelligence Agent — Input‑Enriched, Evidence‑Grounded, LangGraph‑Orchestrated with DSPy Optimization, Tavily/Firecrawl Tools, SGLang Inference, and LangSmith Tracing.

## Executive summary 
This project delivers a production‑ready research agent that decomposes a query into parallel tasks, gathers evidence via agent‑grade search and crawl APIs, reasons primarily over snippets, and returns a structured brief with citations and confidence, following a design similar to proven deep‑research systems. The system uses LangGraph for stateful orchestration, DSPy to optimize input‑enrichment prompts (multi‑query and HyDE), Tavily for search, Firecrawl for structured page extraction, an OpenAI‑compatible runtime via SGLang for fast local inference, and LangSmith for rigorous tracing.

## Why this project
- Research and web intelligence is a top enterprise agentic use case with measurable gains when agents plan, search, extract, and synthesize with strict structure and observability.  
- A snippet‑first strategy reduces cost/latency while preserving quality; full‑page fetch is escalated only when signal is insufficient.  
Input‑enrichment before retrieval materially lifts recall and precision in open‑web settings and can be optimized further with DSPy’s programmatic methods.

## Primary goals
- Plan and execute multi‑step web research with explicit schemas at each stage to ensure reproducibility, quality control, and downstream API consumption.  
- Prioritize snippet‑level reasoning and structured crawling to keep context concise, citing all claims to authoritative sources discovered by agent tools.  
- Instrument every step with tracing to debug failures, measure token usage/latency, and enforce budgets for production viability.

## Non-goals
- Building a general chat assistant or unbounded web surfer; the agent is purpose‑built for high‑precision research briefs with auditable evidence and strict output schemas.  
- Developing a bespoke inference framework; the system uses an OpenAI‑compatible runtime (SGLang) during development and stays portable to any compatible provider.

## Domain scope 
- Focus on real‑time web research across public sources using agent‑grade search and structured crawling with configurable domain filters, time ranges, and depth.  
- Avoid niche proprietary APIs unless they provide clear, additive signal and are accessible with documented terms and rate limits.

## System architecture overview
- Orchestrator: LangGraph runs a planner → parallel task runners → observer/validator graph with explicit state and edges, enabling durable, reliable workflows.  
- Input‑enrichment: DSPy optimizes the prompts for multi‑query expansion and HyDE to boost retrieval recall before expensive LLM synthesis.  
- Search: Tavily provides agent‑grade search with parameters for depth, time windows, domains, and raw content for snippet‑first reasoning.  
- Crawl: Firecrawl fetches and converts pages into LLM‑ready formats (Markdown/HTML/JSON), including async jobs and scrape controls for performance.  
- Inference: SGLang offers an OpenAI‑compatible local endpoint for fast iteration and easy portability to other compatible providers.  
- Observability: LangSmith attaches traces to each node/tool call with token and latency metrics, including distributed tracing integration for LangGraph.

## Core workflow
- Planner node: decomposes the initial question into tasks with required outputs, suggested tools, and target domains/timeframes.  
- Task runner nodes (parallel): execute search with Tavily, pull snippets, optionally escalate to Firecrawl for full content, and synthesize per‑task findings with citations.  
- Observer/validator node: aggregates, deduplicates, validates schema and citation presence, computes confidence, and prepares the final structured brief.

## Why LangGraph
- LangGraph provides explicit graphs, state, and transitions that prevent prompt‑only brittleness, enable parallelism, and support human‑in‑the‑loop validation when needed.  
- The framework’s patterns for multi‑agent research and node‑level control align with industry case studies and best practices for deep research systems.

## Why DSPy for input enrichment
- DSPy allows programming LLM steps with optimizers to systematically improve query diversification (multi‑query), hypothetical document generation (HyDE), and compression prompts.  
- Optimizers such as MIPROv2 and related workflows help find higher‑recall, lower‑cost prompt strategies over a small labeled evaluation set.

## Why Tavily and Firecrawl
- Tavily is purpose‑built for agent workflows with time‑range, domain filters, and raw content retrieval for robust snippet‑first reasoning and fewer dead ends.  
- Firecrawl provides structured crawling and scraping with async controls, returning consistent, LLM‑ready content conducive to downstream compression and synthesis.

## Why SGLang
- SGLang exposes an OpenAI‑compatible server suitable for local and staging inference with low latency, straightforward client integration, and easy provider swaps later.  
- Keeping inference OpenAI‑compatible avoids lock‑in and allows switching to other compatible servers without changing node logic.

## Why LangSmith 
- LangSmith enables systematic tracing of inputs, outputs, tool calls, metrics, and errors per node for faster debugging, cost/latency control, and regression analysis.  
- Distributed tracing support with LangGraph lets traces mirror the actual graph execution for end‑to‑end visibility.

## Data flow (end‑to‑end)
- Receive query → normalize and classify intent → multi‑query expansion → HyDE hypothetical doc → search with Tavily → snippet reasoning; if insufficient, Firecrawl crawl → contextual compression → per‑task synthesis → validation/aggregation → final structured brief with citations and confidence.

## Typed state (recommended)  
- normalized_query: string.  
- sub_queries: string[].  
- hyde_doc: string.  
- tasks: {id, question, target_domains?, time_range?, require_full_content?: bool}[].  
- evidence_snippets: {task_id, url, snippet, title?, published_at?}[].  
- crawled_docs: {task_id, url, content_md, meta}.  
- compressed_context: {task_id, url, extracted_points[]}[].  
- per_task_findings: {task_id, summary, citations[], confidence}.  
- final_brief: {topic, questions_answered[], key_findings[], evidence[], citations[], remaining_gaps[], confidence}.

## Input enrichment modules
- Multi‑query expansion: generate diverse paraphrases/decompositions, deduplicate, and route to search; optimize prompt with DSPy for recall@k on the eval set.  
- HyDE: create a short hypothetical answer document to seed embeddings/search for semantically richer matches; optimize instruction length/style via DSPy.  
- Contextual compression: filter irrelevant snippets or crawled content to minimize context tokens while preserving salient facts.

## Tooling interfaces

### Tavily Search (as a tool) 
- Inputs: {query, time_range?, domains?, include_raw_content?: bool, max_results?}.  
- Outputs: [{title, url, snippet, raw_content?, published_at?, score?}].

### Firecrawl Crawl (as a tool)
- Inputs: {url, include_selectors?, max_depth?, obey_robots?: bool, render_js?: bool, async?: bool}.  
- Outputs: {content_md, content_html?, metadata, status, job_id?}, with async job polling when requested.

## MCP adapter usage
- Load search and crawl tools via MCP adapters to standardize schemas and decouple tool implementations from orchestration logic; optionally expose this agent as an MCP endpoint with typed I/O.  
- MCP standardization eases adding or swapping tools without refactoring graph nodes, improving maintainability.

## Inference configuration
- Default dev/staging: SGLang at base_url http://localhost:<port>/v1 with a configured OSS model; use a standard OpenAI client pointed to this base URL.  
- Portability: Keep the client configuration provider‑agnostic to switch to any OpenAI‑compatible endpoint later without code changes.

## Observability and tracing
- Attach tracing to each node and tool call with LangSmith to capture input/output payloads, token usage, and latency, including graph‑aware distributed traces.  
- Use traces to enforce latency budgets, find bottlenecks, and verify that snippet‑first reasoning is reducing cost compared to full‑page flows.

## Evaluation and CI
- Build a 30–50 prompt evaluation set spanning exploratory, comparative, and timeline research tasks; measure recall@k for retrieval, citation completeness, faithfulness, and p95 latency per stage.  
- Run a baseline vs. DSPy‑optimized A/B to quantify improvements in recall and reductions in redundant fetches, logging results for reproducibility.

## Security and compliance
- Enforce domain and time filters at the planner and search stages to avoid untrusted or irrelevant sources by default; use allowlists for sensitive deployments.  
- Respect robots.txt and site policies when crawling, and prefer snippet‑first logic to reduce unnecessary content ingestion.  
- Ensure secrets for search/crawl APIs and inference providers are stored in environment variables and never logged in traces.

## Performance targets
- Prefer snippet‑first reasoning to keep context windows small and reduce token spend; escalate to crawling only if snippet signal is insufficient per task.  
- Set p95 latency budgets per node and overall, and log token usage per stage to keep costs predictable at scale.

## Acceptance criteria
- Quality: The agent returns a structured brief with complete citations for all key findings on ≥90% of evaluation prompts, with measured gains over baseline due to input‑enrichment.  
- Performance: p95 end‑to‑end latency meets target budgets, with snippet‑first mode significantly cheaper than default full‑page fetch baselines.  
- Operability: Traces show per‑node inputs, outputs, tool calls, and metrics, with reproducible runs across providers via OpenAI‑compatible clients.

## Repository layout (suggested)
- app/graph/nodes/{planner.py, task_runner.py, observer.py, enrichment.py} for LangGraph nodes and transitions.  
- app/tools/{tavily_tool.py, firecrawl_tool.py, mcp_loader.py} for tool wrappers and MCP adapter plumbing.  
- app/dspy/{multi_query_predictor.py, hyde_predictor.py, compile_optimizers.py} for DSPy modules and compile scripts.  
- app/tracing/{langsmith.py} for tracing initialization and run helpers integrated with LangGraph.  
- app/config/{providers.py, settings.py} for SGLang/OpenAI client setup and environment toggles.  
- eval/{dataset.jsonl, runner.py, metrics.py} for evaluation prompts, runners, and metric calculators.  
- README.md with quickstart, environment, and A/B results, and CONTEXT.md for architectural clarity.

## Environment and configuration
- Required env: TAVILY_API_KEY, FIRECRAWL_API_KEY, OPENAI_API_KEY or SGLANG_BASE_URL/MODEL for OpenAI‑compatible runtime, plus tracing keys if applicable.  
- Provider toggle: Use a single switch to move between local SGLang and any compatible managed endpoint without changing graph code.

## Planner and tasks
- The planner emits tasks with explicit required outputs and suggested tools to enable parallelism and reduce wasted calls, guided by learned patterns from production‑grade research systems.  
- Tasks should declare domain filters and time windows to improve result quality and avoid stale evidence.

## Snippet‑first strategy
- Begin with Tavily results and reason over snippets, which tend to be high‑signal and concise, and only crawl granular pages when snippets lack sufficient coverage.  
- This strategy allows better cost control and faster first answers while still supporting deeper dives when necessary.

## Contextual compression
- After search and/or crawl, apply contextual compression to filter irrelevant content and reduce prompt size, preserving only salient points tied to the tasks.  
- Compression improves latency and faithfulness by keeping the synthesis grounded in relevant evidence.

## Synthesis and validation 
- Per‑task synthesis produces a short JSON object with findings and citations, which the observer aggregates and validates for completeness and consistency.  
- The final brief follows a strict schema with topic, questions_answered, key_findings, evidence, citations, remaining_gaps, and confidence for robust downstream use.

## DSPy optimization loop
- Create small labeled evals for enrichment nodes and run DSPy optimizers to compile improved predictors that maximize retrieval recall and reduce redundant queries.  
- Persist compiled predictors and load them at runtime for deterministic behavior and simpler rollback if regressions occur.

## Tracing and budgets
- Attach trace context across nodes and tool calls and log tokens and latency, then set and enforce budgets per node and end‑to‑end, with alerts on breaches.  
- Use traces to A/B compare prompt variants, providers, and tool configurations to continuously improve cost/quality trade‑offs.

## Failure handling
- Implement retries with exponential backoff on tool calls, degrade gracefully when search/crawl limits are hit, and mark partial results with explicit remaining_gaps.  
- If citations are missing or low‑confidence thresholds are not met, fail closed with a helpful message rather than fabricate content.

## Testing strategy
- Unit test nodes for schema compliance and error paths, integration test planner→tasks→observer for end‑to‑end behavior, and regression test on the eval set with fixed seeds and provider configs.  
- Track metrics over time to catch drift when upstream tools or providers update behavior or ranking.

## Deployment posture 
- Primary target is local/staging environments with SGLang for development speed and observability, keeping a straightforward path to any OpenAI‑compatible runtime later.  
- Keep secrets in environment variables and avoid embedding keys in code or traces; rotate keys periodically and restrict scopes where supported.

## Operating guidelines 
- Prefer conservative tool limits and depth initially, then tune with measurements from traces and evaluation outcomes to reach the desired balance.  
- Maintain allowlists for domains where feasible to improve precision and avoid untrusted sources for critical topics.

## References and implementation guides
- Exa deep‑research case study for multi‑agent planning, snippet‑first reasoning, and structured outputs.  
- LangGraph research agent patterns and overviews to shape node design and execution reliability.  
- Tavily integration docs and Python wrapper for agent search tooling.  
- Firecrawl crawl API docs and integration tutorials for structured extraction.  
- DSPy repository for programmatic LLM optimization workflows.  
- SGLang quick start for OpenAI‑compatible local inference configuration.  
- LangSmith tracing docs for per‑node and distributed traces with LangGraph.

<!-- ORIGINAL_CONTEXT_END -->

---

## Appendix A — Tooling references (added; original unchanged)

- Orchestration: LangGraph research/agent patterns for planner→tasks→observer flows. [web:531]  
- Input enrichment: DSPy framework and optimizers for multi‑query/HyDE tuning. [web:351]  
- Search: Tavily API for agent‑grade search with snippets and filters. [web:681][web:684]  
- Crawl: Firecrawl crawl/scrape endpoints for LLM‑ready content. [web:693][web:695]  
- Inference: SGLang as an OpenAI‑compatible local runtime for fast iteration. [web:588]  
- Observability: LangSmith tracing and distributed tracing with LangGraph. [web:694][web:440]  
- Markdown integrity: follow basic Markdown syntax to ensure consistent rendering. [web:711]  

## Appendix B — Minimal runbook (added; original unchanged)

- Prerequisites: Python 3.11+, SGLang running locally, Tavily/Firecrawl API keys in .env. [web:588][web:684][web:693]  
- Install: pip install -r requirements.txt and export environment variables for tools and tracing. [web:711]  
- Run (dev): python -m app.graph.graph and verify traces appear in the tracing UI. [web:694]  
- Evaluate: python -m eval.runner and python -m eval.metrics on eval/dataset.jsonl for baseline and A/B after DSPy compile. [web:351]  

## Appendix C — I/O contracts (added; original unchanged)

- Tavily input/output shapes as listed in the original section should be enforced at tool boundaries to keep nodes deterministic. [web:681]  
- Firecrawl input/output shapes as listed in the original section should be enforced, using async jobs for high‑latency pages. [web:693]  
- Final brief JSON schema from the original section must be validated before emitting responses. [web:711]  

## Appendix D — Quality gates (added; original unchanged)

- Retrieval quality: require recall@k improvement from enrichment vs single‑query baseline before merging prompt changes. [web:351]  
- Output faithfulness: require complete citations for key findings and automated faithfulness checks on sampled outputs. [web:694]  
- Performance: enforce per‑node and end‑to‑end p95 latency budgets with alerts on breach via traces. [web:694]  

## Appendix E — Security posture (added; original unchanged)

- Respect robots.txt and use domain allowlists where appropriate to avoid untrusted content ingestion. [web:693]  
- Keep secrets in environment variables and never log raw keys or access tokens in traces or console. [web:694]  
- Prefer snippet‑first and minimal content fetch to reduce unnecessary data handling on third‑party sites. [web:681]  

## Appendix F — Change control (added; original unchanged)

- Record any prompt/route/provider changes in docs/decisions/ with date, rationale, and trace links for auditability. [web:694]  
- Version evaluation datasets and compiled DSPy artifacts to reproduce results and roll back safely. [web:351]  


## Appendix G — Investment Banking Analytics Transformation (Oct 2024)

### Project Evolution
The core architecture described above has been specialized for Investment Banking Analytics while maintaining all original technical capabilities. The transformation leverages the same DSPy + LangGraph + multi‑tool foundation for high‑value financial analysis use cases.

### Investment Banking Specialization
- **Domain Focus**: M&A due diligence, company valuation, market intelligence, risk assessment
- **Analysis Types**: Due diligence workflows, DCF valuations, competitive analysis, regulatory compliance checks
- **Output Format**: Investment banking‑grade reports with structured financial metrics, risk assessments, and confidence scoring
- **Quality Standards**: Conservative analysis with multiple validation layers suitable for financial decision‑making

### Multi‑AI Provider Architecture Enhancement
- **Provider Abstraction**: `axiom/ai_client_integrations/base_ai_provider.py` — unified interface for OpenAI, Claude, SGLang, Hugging Face, Gemini
- **Layer Configuration**: `axiom/config/ai_layer_config.py` — user‑configurable AI provider assignments per analysis type
- **Dynamic Detection**: System auto‑detects available providers based on configured credentials
- **Optimal Routing**: Due diligence → Claude (reasoning), Valuation → OpenAI (structured), Market Intelligence → Claude (synthesis)

### Investment Banking Workflow Adaptations
- **Financial Planner**: Decomposes investment queries into structured analysis tasks (financial health, market position, regulatory compliance)
- **Parallel Analysis Engines**: Execute financial research across multiple data sources simultaneously
- **Investment Validator**: Aggregates findings with conservative confidence thresholds suitable for financial decisions
- **Audit Trail**: Enhanced LangSmith tracing for regulatory compliance and decision documentation

### Environment Configuration Updates
- **Multi‑Provider Support**: Optional configuration for OpenAI, Claude, SGLang, Hugging Face, Gemini
- **Financial Data Sources**: Integration points for financial APIs (Alpha Vantage, Financial Modeling Prep, Polygon)
- **Investment Banking Parameters**: Due diligence confidence thresholds, valuation model types, risk assessment settings
- **Compliance Configuration**: Regulatory compliance checks, audit trail requirements, conservative temperature settings

### Repository Structure Evolution
```
axiom/
├── ai_client_integrations/  # Multi‑AI provider abstraction layer
├── config/                  # Enhanced with AI layer configuration and financial parameters  
├── graph/                   # Investment banking workflow orchestration
├── tools/                   # Financial data integration (Tavily, Firecrawl, MCP)
├── dspy_modules/           # Financial query optimization and analysis enhancement
├── tracing/                # Audit trails and compliance tracking
└── eval/                   # Investment decision accuracy and performance metrics
```

### Backwards Compatibility
- Original research agent functionality preserved as foundation
- Same core architecture (LangGraph + DSPy + tools + tracing)
- OpenAI‑compatible inference maintains provider portability
- All original evaluation and optimization workflows remain functional

### Usage Examples
```bash
# M&A Analysis
python -m axiom.main "Analyze Microsoft acquisition of OpenAI for strategic value"

# Due Diligence  
python -m axiom.main "Comprehensive due diligence analysis of NVIDIA financial health"

# Market Intelligence
python -m axiom.main "Investment banking analysis of AI infrastructure market trends"
```

### Development Status
- **macOS Development**: Fully functional with core components and multi‑provider support
- **NVIDIA Production**: SGLang local inference ready for high‑performance deployment
- **Multi‑AI Flexibility**: Users can configure optimal AI providers for each analysis layer
- **Professional Grade**: Investment banking‑focused prompts, conservative settings, audit compliance
