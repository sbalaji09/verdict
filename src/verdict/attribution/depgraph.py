"""A deliberately lightweight static import graph, used only to *enrich*
an already-proven attribution with a "why" — never to establish one.

This does not resolve imports the way a real toolchain would: no
sys.path/tsconfig `paths` awareness, no handling of dynamic imports,
no distinguishing a package from a module it re-exports through. A full
resolver is a project unto itself and isn't what's load-bearing here — the
bisection result is the proof regardless of whether this graph finds a
matching edge. Missing an edge just means a thinner sentence, not a wrong
attribution.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

class _PyModule(NamedTuple):
    file: str          # repo-relative, posix-style
    is_package: bool    # True if this was an __init__.py


def _iter_files(worktree: Path, suffixes: tuple[str, ...]) -> list[Path]:
    out = []
    for path in worktree.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(worktree).parts):
            continue
        out.append(path)
    return out


def _module_path(worktree: Path, file: Path) -> _PyModule:
    rel = file.relative_to(worktree)
    parts = list(rel.parts)
    is_package = parts[-1] == "__init__.py"
    if is_package:
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return _PyModule(file=rel.as_posix(), is_package=is_package)


def _build_python_graph(worktree: Path) -> dict[str, set[str]]:
    files = _iter_files(worktree, (".py",))
    module_map: dict[str, _PyModule] = {}
    file_module: dict[Path, str] = {}
    for f in files:
        mod = _module_path(worktree, f)
        # mod.file already has __init__.py stripped for packages, so this
        # dotted form is package-shaped without extra handling needed.
        dotted = mod.file.removesuffix(".py").replace("/", ".")
        module_map[dotted] = mod
        file_module[f] = dotted

    graph: dict[str, set[str]] = {}

    def _package_of(dotted: str, is_package: bool) -> str:
        if is_package:
            return dotted
        return dotted.rsplit(".", 1)[0] if "." in dotted else ""

    def _ascend(package: str, steps: int) -> str:
        for _ in range(steps):
            package = package.rsplit(".", 1)[0] if "." in package else ""
        return package

    for f in files:
        dotted = file_module[f]
        rel = f.relative_to(worktree).as_posix()
        edges: set[str] = set()
        try:
            tree = ast.parse(f.read_text(errors="ignore"))
        except SyntaxError:
            graph[rel] = edges
            continue

        is_package = module_map[dotted].is_package
        current_package = _package_of(dotted, is_package)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in module_map:
                        edges.add(module_map[alias.name].file)
            elif isinstance(node, ast.ImportFrom):
                target_package = _ascend(current_package, max(node.level - 1, 0)) if node.level else ""
                if node.module:
                    prefix = f"{target_package}.{node.module}" if target_package else node.module
                else:
                    prefix = target_package
                if prefix in module_map:
                    edges.add(module_map[prefix].file)
                    continue
                for alias in node.names:
                    candidate = f"{prefix}.{alias.name}" if prefix else alias.name
                    if candidate in module_map:
                        edges.add(module_map[candidate].file)

        graph[rel] = edges

    return graph


_JS_IMPORT_RES = (
    re.compile(r"""from\s+['"]([^'"]+)['"]"""),
    re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""^\s*import\s+['"]([^'"]+)['"]""", re.MULTILINE),
)
_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx")


def _resolve_js_specifier(worktree: Path, from_file: Path, specifier: str) -> str | None:
    if not specifier.startswith("."):
        return None  # bare package import — external, not in this repo
    target = (from_file.parent / specifier).resolve()
    candidates = [target] if target.suffix in _JS_EXTENSIONS else []
    candidates += [target.with_suffix(target.suffix + ext) for ext in _JS_EXTENSIONS]
    candidates += [target / f"index{ext}" for ext in _JS_EXTENSIONS]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return candidate.relative_to(worktree).as_posix()
            except ValueError:
                return None
    return None


def _build_js_graph(worktree: Path) -> dict[str, set[str]]:
    files = _iter_files(worktree, _JS_EXTENSIONS)
    graph: dict[str, set[str]] = {}
    for f in files:
        rel = f.relative_to(worktree).as_posix()
        edges: set[str] = set()
        text = f.read_text(errors="ignore")
        specifiers: set[str] = set()
        for pattern in _JS_IMPORT_RES:
            specifiers.update(pattern.findall(text))
        for spec in specifiers:
            resolved = _resolve_js_specifier(worktree, f, spec)
            if resolved:
                edges.add(resolved)
        graph[rel] = edges
    return graph


def build_dependency_graph(worktree: Path) -> dict[str, set[str]]:
    """file (repo-relative, posix) -> set of files it imports (same form)."""
    graph = _build_python_graph(worktree)
    for rel, edges in _build_js_graph(worktree).items():
        graph.setdefault(rel, set()).update(edges)
    return graph


def find_dependency_depth(
    graph: dict[str, set[str]], from_file: str, to_file: str, max_depth: int = 4
) -> int | None:
    """BFS from `from_file` for `to_file`, up to `max_depth` hops. Returns
    the hop count if found, else None. `from_file` is typically the file
    the failing test/error lives in; `to_file` is the bisection culprit.
    """
    if from_file == to_file:
        return 0
    frontier = {from_file}
    seen = {from_file}
    for depth in range(1, max_depth + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in graph.get(node, ()):
                if neighbor == to_file:
                    return depth
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            return None
        frontier = next_frontier
    return None
