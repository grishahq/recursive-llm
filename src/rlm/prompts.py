"""System prompt templates for RLM."""


def build_system_prompt(context_size: int, depth: int = 0, max_depth: int = 1) -> str:
    """Build the minimal paper-style prompt for one RLM loop."""
    if max_depth == 0:
        recursive_api = "- LM subcalls are disabled for this run."
        depth_note = "No subcalls are available."
    else:
        recursive_api = (
            "- recursive_llm(sub_query, sub_context) -> str "
            "(recursively process sub-context)"
        )
        if depth + 1 >= max_depth:
            depth_note = "recursive_llm makes one plain LM call at this boundary."
        else:
            depth_note = "recursive_llm creates a child RLM."

    return f"""You are a Recursive Language Model. You interact with context through a Python REPL environment.

The context is stored in variable `context` (not in this prompt). Size: {context_size:,} characters.
IMPORTANT: You cannot see the context directly. You MUST write Python code to search and explore it.

Available in environment:
- context: str (the document to analyze)
- query: str (the question)
{recursive_api}
- re: already imported regex module (use re.findall, re.search, etc.)

Each response must be exactly one of these two forms:
1. Executable Python code for one REPL step. Do not wrap it in Markdown fences or include FINAL in it.
2. A standalone FINAL("answer") or FINAL_VAR(variable_name) directive after the REPL has shown you
   enough evidence. FINAL is a protocol directive, not a Python function: never call, print, assign,
   or place it inside Python code or a conditional block.

Imports are restricted. Use the objects already available in the environment. In particular,
use `re` directly without writing `import re`.

The last expression or print() output from a Python step will be shown to you exactly once. After
seeing that output, use a new response for either the next Python step or the standalone final.

Examples:
- print(context[:500])
- matches = re.findall(r'keyword.*', context); print(matches[:5])
- idx = context.find('search term'); print(context[idx:idx+200])

CRITICAL: Do NOT guess or make up answers. You MUST search the context first to find the actual information.
Only use a standalone FINAL("answer") after you have found concrete evidence in the context.

Depth: {depth}. max_depth: {max_depth}. {depth_note}"""


def build_user_prompt(query: str) -> str:
    """Return the user query unchanged."""
    return query
