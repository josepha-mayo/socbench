"""Tree-sitter syntax validation (optional).

The audit pipeline uses tree-sitter grammars when available. If a grammar is not
installed or fails to load, that language is skipped rather than failing the
whole audit.
"""

from __future__ import annotations

import logging
from typing import Callable

LOGGER = logging.getLogger("socbench.audit")

# Map common fastText language codes to the grammar module names used by
# py-tree-sitter. Users can install the grammars they need; missing ones are
# logged and skipped.
LANGUAGE_TO_GRAMMAR: dict[str, str | list[str]] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java": "tree_sitter_java",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "cpp": "tree_sitter_cpp",
    "c": "tree_sitter_c",
    "c++": ["tree_sitter_cpp", "tree_sitter_c"],
    "php": "tree_sitter_php",
    "ruby": "tree_sitter_ruby",
    "swift": "tree_sitter_swift",
    "kotlin": ["tree_sitter_kotlin", "tree_sitter_java"],
    "scala": ["tree_sitter_scala", "tree_sitter_java"],
    "csharp": "tree_sitter_c_sharp",
}


def _load_grammar(module_name: str) -> object | None:
    try:
        mod = __import__(module_name, fromlist=["LANGUAGE"])
        return getattr(mod, "LANGUAGE", None)
    except Exception:
        return None


def _get_parser_for_language(language: str) -> tuple[Callable[[str], bool], str] | None:
    """Return a parse checker and the module it came from, or None."""
    candidates = LANGUAGE_TO_GRAMMAR.get(language.lower())
    if not candidates:
        return None
    if isinstance(candidates, str):
        candidates = [candidates]

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        return None

    for module_name in candidates:
        grammar = _load_grammar(module_name)
        if grammar is None:
            continue
        try:
            lang = Language(grammar)
            parser = Parser(lang)

            def check(text: str, parser=parser) -> bool:
                try:
                    tree = parser.parse(text.encode("utf-8"))
                    # Reject if the root node is an ERROR.
                    return not (tree.root_node.type == "ERROR")
                except Exception:
                    return False

            return check, module_name
        except Exception:
            continue
    return None


def is_code_like(text: str) -> bool:
    """Heuristic to decide whether a row is worth running through a code parser."""
    lower = text.lower()
    code_markers = ("def ", "class ", "import ", "function ", "const ", "let ", "var ",
                    "package ", "public class", "#include", "using namespace", "int main",
                    "struct ", "impl ", "fn ")
    return any(m in lower for m in code_markers)


def validate_code_syntax(text: str, language: str) -> bool:
    """Return True if the text parses in ``language`` or if no parser is available.

    Non-code text and languages without an installed grammar pass through so the
    audit can continue without hard dependencies on every grammar.
    """
    if not is_code_like(text):
        return True

    checker = _get_parser_for_language(language)
    if checker is None:
        LOGGER.debug("No tree-sitter grammar available for %s", language)
        return True

    parse, module_name = checker
    return parse(text)
