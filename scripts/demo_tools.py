"""
demo_tools.py - Lightweight tool-calling system for AnyProjector Demo.

LLM outputs JSON like: {"tool": "calculator", "expression": "15 * 7"}
This module detects, parses, and executes such tool calls.
"""

import json
import re
import ast
import math
import operator
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Any


# ── Safe Math Evaluator ──────────────────────────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval_node(node):
    """Recursively evaluate an AST node with only arithmetic ops."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _SAFE_OPS[type(node.op)](left, right)
    else:
        raise ValueError(f"Unsupported operation: {ast.dump(node)}")


def safe_math_eval(expression: str) -> str:
    """Safely evaluate a math expression (no code injection)."""
    # Clean up common Vietnamese/natural language patterns
    expr = expression.strip()
    expr = expr.replace("x", "*").replace("X", "*")
    expr = expr.replace(",", "")  # Remove thousand separators

    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree.body)
        # Format nicely
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


# ── Tool Definitions ─────────────────────────────────────────────────

@dataclass
class Tool:
    """A callable tool that the LLM can invoke via JSON."""
    name: str
    description: str
    parameters: dict[str, str]  # param_name -> description
    handler: Callable[[dict], str]


def _handle_calculator(args: dict) -> str:
    expr = args.get("expression", "")
    if not expr:
        return "Error: missing 'expression' parameter"
    return safe_math_eval(expr)


def _handle_get_time(args: dict) -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


def _handle_translate(args: dict) -> str:
    text = args.get("text", "")
    target = args.get("to", "en")
    if not text:
        return "Error: missing 'text' parameter"
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="auto", target=target).translate(text)
        return result or "Translation failed"
    except ImportError:
        return f"[deep_translator not installed] Would translate: '{text}' -> {target}"
    except Exception as e:
        return f"Translation error: {e}"


def _handle_search(args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "Error: missing 'query' parameter"
    # Simulated search with common knowledge
    return f"[Search result for '{query}': No live search available in demo mode. Try asking the LLM directly.]"


# ── Tool Registry ────────────────────────────────────────────────────

BUILTIN_TOOLS: list[Tool] = [
    Tool(
        name="calculator",
        description="Calculate a math expression. Use for arithmetic operations.",
        parameters={"expression": "Math expression to evaluate (e.g. '15 * 7 + 3')"},
        handler=_handle_calculator,
    ),
    Tool(
        name="get_time",
        description="Get the current date and time.",
        parameters={},
        handler=_handle_get_time,
    ),
    Tool(
        name="translate",
        description="Translate text to another language.",
        parameters={
            "text": "Text to translate",
            "to": "Target language code (e.g. 'en', 'vi', 'ja', 'zh')",
        },
        handler=_handle_translate,
    ),
    Tool(
        name="search",
        description="Search for information on the web.",
        parameters={"query": "Search query string"},
        handler=_handle_search,
    ),
]


class ToolRegistry:
    """Manages tool registration, prompt injection, and execution."""

    def __init__(self, tools: list[Tool] | None = None):
        self.tools: dict[str, Tool] = {}
        for tool in (tools or BUILTIN_TOOLS):
            self.tools[tool.name] = tool

    def get_tool_prompt(self, enabled_tools: list[str] | None = None) -> str:
        """Generate system prompt section describing available tools."""
        active = {k: v for k, v in self.tools.items()
                  if enabled_tools is None or k in enabled_tools}
        if not active:
            return ""

        lines = [
            "You have access to the following tools. To use a tool, respond with a JSON object:",
            '{"tool": "<tool_name>", ...parameters}',
            "",
            "Available tools:",
        ]
        for name, tool in active.items():
            params_str = ", ".join(
                f'"{k}": "{v}"' for k, v in tool.parameters.items()
            ) if tool.parameters else "no parameters needed"
            lines.append(f"- {name}: {tool.description}")
            lines.append(f"  Parameters: {{{params_str}}}")

        lines.append("")
        lines.append("If no tool is needed, respond normally with text.")
        return "\n".join(lines)

    def detect_and_execute(self, llm_output: str) -> tuple[dict | None, str | None]:
        """Detect JSON tool call in LLM output and execute it.

        Returns:
            (tool_call_dict, result_string) or (None, None) if no tool call found.
        """
        # Try to find JSON in the output
        json_match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\}', llm_output)
        if not json_match:
            return None, None

        try:
            call = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None, None

        tool_name = call.get("tool")
        if not tool_name or tool_name not in self.tools:
            return call, f"Unknown tool: {tool_name}"

        tool = self.tools[tool_name]
        try:
            result = tool.handler(call)
            return call, result
        except Exception as e:
            return call, f"Tool error: {e}"

    def list_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self.tools.keys())
