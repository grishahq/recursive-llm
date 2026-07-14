# Recursive Language Models (RLM)

Python implementation of Recursive Language Models for processing unbounded context lengths.

**Based on the [RLM paper](https://arxiv.org/abs/2512.24601) by Alex L. Zhang, Tim Kraska, and Omar Khattab** | [Official implementation](https://github.com/alexzhang13/rlm)


## What is RLM?

RLM enables language models to process extremely long contexts (100k+ tokens) by:
- Storing context as a Python variable instead of in the prompt
- Allowing the LM to recursively explore and partition the context
- Avoiding "context rot" (performance degradation with long context)

Instead of this:
```python
llm.complete(prompt="Summarize this", context=huge_document)  # Context rot!
```

RLM does this:
```python
rlm = RLM(model="gpt-5-mini")
result = rlm.complete(
    query="Summarize this",
    context=huge_document  # Stored as variable, not in prompt
)
```

The LM can then peek, search, and recursively process the context adaptively.

## Installation

**Note:** This package is not yet published to PyPI. Install from source:

```bash
# Clone the repository
git clone https://github.com/grishahq/recursive-llm.git
cd recursive-llm

# Install in editable mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

**Future:** Once published to PyPI, you'll be able to install with `pip install recursive-llm`

## Requirements

- Python 3.9 or higher
- An API key for your chosen LLM provider (OpenAI, Anthropic, etc.)
- Or a local model setup (Ollama, llama.cpp, etc.)

## Quick Start

```python
from rlm import RLM

def main():
    # Initialize with any LLM
    rlm = RLM(model="gpt-5-mini")

    # Process long context
    result = rlm.complete(
        query="What are the main themes in this document?",
        context=long_document,
    )
    print(result)

if __name__ == "__main__":
    main()
```

RLM uses a spawned worker process for isolated REPL execution. Executable Python scripts must use
the standard `if __name__ == "__main__":` entry-point guard, as shown above. This is required by
Python multiprocessing on spawn-based platforms; all repository examples follow this pattern.

### Usage and Cost Statistics

`RLM.stats` aggregates model calls across the complete recursion tree. Token usage comes from
provider responses, while cost is calculated on a best-effort basis using LiteLLM's model pricing
metadata.

```python
rlm = RLM(
    model="gpt-5-mini",
    recursive_model="deepseek/deepseek-v4-flash",
)
result = rlm.complete(query="Summarize this", context=document)

print(rlm.stats)
# {
#     "llm_calls": 11,
#     "root_calls": 3,
#     "recursive_calls": 8,
#     "leaf_calls": 4,
#     "prompt_tokens": 12500,
#     "completion_tokens": 3200,
#     "cached_tokens": 6000,
#     "estimated_cost_usd": 0.0047,
#     "by_model": {
#         "gpt-5-mini": {"calls": 3, ...},
#         "deepseek/deepseek-v4-flash": {"calls": 8, ...},
#     },
# }
```

Statistics are reset when a new root completion starts, so they describe the latest recursion tree.
`estimated_cost_usd` is `None` when LiteLLM has no pricing metadata for any completed call. Compare
`priced_calls` with `llm_calls` before treating the estimate as the full run cost.

### Live Model Comparison

The comparison script uses the same model for both root and recursive calls. It runs one small
recursive smoke test by default and reports latency, calls, tokens, and estimated cost:

```bash
python benchmarks/compare_same_model.py gpt-5-mini
python benchmarks/compare_same_model.py deepseek/deepseek-v4-flash
```

Use `--full` for the slower two-task suite. Live benchmarks make paid API calls and require the
corresponding provider keys.

## API Keys Setup

Copy the example environment file and add keys only for the providers you use:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=...
MOONSHOT_API_KEY=...
```

Use the LiteLLM provider prefix for non-OpenAI models, for example
`deepseek/deepseek-v4-flash` or `moonshot/kimi-k2.6`. This lets a hybrid RLM select the correct API
key for each model automatically.

Or pass directly in code:
```python
rlm = RLM(model="gpt-5-mini", api_key="sk-...")
```

## Supported Models

Works with 100+ LLM providers via LiteLLM:

```python
# OpenAI
rlm = RLM(model="gpt-5")
rlm = RLM(model="gpt-5-mini")

# Anthropic
rlm = RLM(model="claude-sonnet-4")
rlm = RLM(model="claude-sonnet-4-20250514")

# Ollama (local)
rlm = RLM(model="ollama/llama3.2")
rlm = RLM(model="ollama/mistral")

# llama.cpp (local)
rlm = RLM(
    model="openai/local",
    api_base="http://localhost:8000/v1"
)

# Azure OpenAI
rlm = RLM(model="azure/gpt-4-deployment")

# And many more via LiteLLM...
```

## Advanced Usage

### Two Models (Optimize Cost)

Use a cheaper model for recursive calls:

```python
rlm = RLM(
    model="gpt-5",              # Root LM (main decisions)
    recursive_model="gpt-5-mini"  # Recursive calls (cheaper)
)
```

### Async API

For better performance with parallel recursive calls:

```python
import asyncio

async def main():
    rlm = RLM(model="gpt-5-mini")
    result = await rlm.acomplete(query, context)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration

```python
rlm = RLM(
    model="gpt-5-mini",
    max_depth=2,                 # One child RLM level, then a plain-LM fallback
    max_iterations=20,           # Maximum REPL iterations per RLM
    repl_timeout=5,              # Hard timeout for each local Python step
    max_output_chars=2000,       # Observation truncation limit
    max_concurrent_subcalls=4,   # Bound batch concurrency
    # Optional LiteLLM params: temperature, timeout, etc.
)
```

`max_depth` is an explicit constructor option so that runs remain reproducible. It is not read from
an environment variable by the library. Applications may map their own configuration or environment
variables to this argument.

| `max_depth` | Behavior |
| ---: | --- |
| `0` | Root RLM and REPL only; no LM subcalls |
| `1` (default) | Root RLM may call a plain LM |
| `2` | Root RLM may create one child RLM; the child falls back to a plain LM |
| `n` | Adds one child RLM level for every increment above `1` |

This follows the paper's capability-based depth convention. The root RLM itself is depth `0` and is
still valid when `max_depth=0`.

### REPL Subcall API

The model can use these functions from its persistent REPL:

```python
# One plain LM call, without another REPL loop
llm_query("Extract the date", context[1000:2000])

# Child RLM when depth permits; otherwise one plain-LM boundary call
rlm_query("Analyze this section", context[2000:8000])

# Ordered parallel calls, limited by max_concurrent_subcalls
results = llm_query_batched(queries, chunks)
```

`recursive_llm` remains as a backward-compatible alias for `rlm_query`. A step can finish directly
through `FINAL(...)`, `FINAL_VAR(...)`, or the mutable `answer` object:

```python
answer["content"] = result
answer["ready"] = True
```

## How It Works

1. **Context is stored as a variable** in a Python REPL environment
2. **Root LM gets only the query** plus instructions
3. **LM can explore context** using Python code:
   ```python
   # Peek at context
   context[:1000]

   # Search with regex
   re.findall(r'pattern', context)

   # Recursive processing with a plain-LM boundary fallback
   rlm_query("extract dates", context[1000:2000])
   ```
4. **Returns the final answer** via a standalone `FINAL("answer")`, `FINAL_VAR(name)`, or `answer`
   publication

REPL variables persist between iterations. Each local step executes in an isolated subprocess, its
final expression is evaluated exactly once, print output is isolated per step, and non-terminating
local code is terminated by `repl_timeout`. Time spent waiting for model subcalls is not charged to
the local Python timeout. Imports are limited to the already exposed `re`, `json`, `math`,
`datetime`, and `collections` helpers; arbitrary modules remain blocked.

## Examples

See the `examples/` directory for complete working examples:
- `basic_usage.py` - Simple complete with OpenAI
- `ollama_local.py` - Using Ollama locally
- `two_models.py` - Cost optimization with two models
- `long_document.py` - Processing 50k+ token documents
- `data_extraction.py` - Extract structured data from text
- `multi_file.py` - Process multiple documents
- `custom_config.py` - Advanced configuration

Run an example:
```bash
# Set your API key first
export OPENAI_API_KEY="sk-..."

# Run example
python examples/basic_usage.py
```

## Performance

### Paper Results

On OOLONG benchmark (132k tokens):
- GPT-5: baseline
- RLM(GPT-5-Mini): **33% better than GPT-5** at similar cost

### Reproducible Project Benchmark

`benchmarks/compare_same_model.py` contains deterministic structured contexts with exact expected
answers. Use `--full` to run both tasks. Model outputs remain stochastic, so compare several runs
before drawing quality conclusions; the script reports accuracy, latency, calls, tokens, and
best-effort cost for every run.

## Development

```bash
# Clone repository
git clone https://github.com/grishahq/recursive-llm.git
cd recursive-llm

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests, branch coverage, and the enforced coverage gate
pytest

# Type checking
mypy src/rlm

# Linting
ruff check src tests benchmarks examples

# Format code
black src/rlm tests examples
```

## Architecture

```
RLM
├── Core (async completion logic)
├── REPL Executor (restricted subprocess, persistent state, hard step timeout)
├── Prompt Builder (system prompts)
└── Parser (extract FINAL() answers)
```

Built on top of LiteLLM for universal LLM support.

## Limitations

- Python REPL steps are sequential; explicit batched LM/RLM subcalls can run concurrently
- No prefix caching (future enhancement)
- Recursion depth is limited (configurable via `max_depth`)
- No streaming support yet

## Troubleshooting

### "Max iterations exceeded"
- Increase `max_iterations` parameter
- Simplify your query
- Check if the model is getting stuck in a loop

### "API key not found"
- Copy `.env.example` to `.env` and set the appropriate provider variable:
  - `OPENAI_API_KEY` for OpenAI
  - `DEEPSEEK_API_KEY` for DeepSeek
  - `MOONSHOT_API_KEY` for Kimi
- Or pass `api_key` parameter to RLM constructor

### "Model not found"
- Check model name format for your provider
- See LiteLLM docs: https://docs.litellm.ai/docs/providers

### Using Ollama
- Make sure Ollama is running: `ollama serve`
- Pull a model first: `ollama pull llama3.2`
- Use model format: `ollama/model-name`

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest tests/`)
5. Follow code style (use `black` and `ruff`)
6. Submit a pull request

## Citation

This implementation is based on the RLM paper by Alex L. Zhang, Tim Kraska, and Omar Khattab.

**To cite this implementation:**
```bibtex
@software{rlm_python,
  title = {recursive-llm: Python Implementation of Recursive Language Models},
  author = {Gvadzabia, Grisha},
  year = {2025},
  url = {https://github.com/grishahq/recursive-llm}
}
```

**To cite the original paper:**
```bibtex
@misc{zhang2025rlm,
  title = {Recursive Language Models},
  author = {Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  year = {2025},
  url = {https://arxiv.org/abs/2512.24601},
  eprint = {2512.24601},
  archivePrefix = {arXiv}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Based on the Recursive Language Models paper by Alex L. Zhang, Tim Kraska, and Omar Khattab.

Built using:
- LiteLLM for universal LLM API support
- RestrictedPython for restricted code execution

## Links

- **Paper**: https://alexzhang13.github.io/blog/2025/rlm/
- **arXiv**: https://arxiv.org/abs/2512.24601
- **Official implementation**: https://github.com/alexzhang13/rlm
- **LiteLLM Docs**: https://docs.litellm.ai/
- **Issues**: https://github.com/grishahq/recursive-llm/issues
