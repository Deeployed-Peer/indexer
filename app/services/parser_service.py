"""Language-aware parsing for repository indexing."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass(frozen=True)
class ParsedRelationship:
    source: str
    target: str
    relationship_type: str
    attributes: Dict[str, object] = field(default_factory=dict)
    is_external: bool = False


@dataclass(frozen=True)
class ParsedElement:
    qualified_name: str
    symbol: str
    element_type: str
    signature: Optional[str]
    docstring: Optional[str]
    summary: str
    span: Tuple[int, int]
    ast_repr: Dict[str, object]
    tokens: List[str]
    is_external: bool = False


@dataclass(frozen=True)
class ParsedFile:
    path: str
    language: str
    digest: str
    summary: str
    keyword_index: List[str]
    elements: List[ParsedElement]
    relationships: List[ParsedRelationship]


class ParserService:
    """Facade returning parsed representations for source files."""

    def parse_files(self, repo_root: Path, files: Sequence[Path]) -> List[ParsedFile]:
        results: List[ParsedFile] = []
        for path in files:
            if path.suffix == ".py" or path.suffix == ".pyi":
                results.append(self._parse_python_file(repo_root, path))
            else:
                results.append(self._parse_generic_file(repo_root, path))
        return results

    def _parse_python_file(self, repo_root: Path, path: Path) -> ParsedFile:
        source = path.read_text(encoding="utf-8", errors="ignore")
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        rel_path = path.relative_to(repo_root)
        module_name = ".".join(rel_path.with_suffix("").parts)
        module_qname = module_name or path.name
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._parse_generic_file(repo_root, path)

        analyzer = _PythonAnalyzer(
            module_qname=module_qname,
            rel_path=str(rel_path),
            source=source,
        )
        analyzer.visit(tree)
        module_doc = ast.get_docstring(tree)
        module_summary = module_doc.splitlines()[0] if module_doc else f"Module {module_qname}"
        module_element = ParsedElement(
            qualified_name=module_qname,
            symbol=module_qname.split(".")[-1],
            element_type="module",
            signature=None,
            docstring=module_doc,
            summary=module_summary,
            span=(1, len(source.splitlines()) or 1),
            ast_repr={"type": "Module"},
            tokens=_extract_tokens(source),
        )

        elements = [module_element] + analyzer.elements
        relationships = analyzer.relationships

        return ParsedFile(
            path=str(rel_path),
            language="python",
            digest=digest,
            summary=module_summary,
            keyword_index=_extract_tokens(source),
            elements=elements,
            relationships=relationships,
        )

    def _parse_generic_file(self, repo_root: Path, path: Path) -> ParsedFile:
        source = path.read_text(encoding="utf-8", errors="ignore")
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        rel_path = path.relative_to(repo_root)
        tokens = _extract_tokens(source)
        summary = f"File {rel_path}"
        module_name = ".".join(rel_path.parts)
        element = ParsedElement(
            qualified_name=module_name,
            symbol=rel_path.name,
            element_type="file",
            signature=None,
            docstring=None,
            summary=summary,
            span=(1, len(source.splitlines()) or 1),
            ast_repr={"type": "File"},
            tokens=tokens,
        )
        return ParsedFile(
            path=str(rel_path),
            language="text",
            digest=digest,
            summary=summary,
            keyword_index=tokens,
            elements=[element],
            relationships=[],
        )


class _PythonAnalyzer(ast.NodeVisitor):
    """Collects structural information from a Python module."""

    def __init__(self, module_qname: str, rel_path: str, source: str) -> None:
        self.module_qname = module_qname
        self.rel_path = rel_path
        self.source = source
        self.elements: List[ParsedElement] = []
        self.relationships: List[ParsedRelationship] = []
        self.scope: List[Tuple[str, str]] = []
        self.import_aliases: Dict[str, str] = {}
        self.defined_names: Dict[str, List[str]] = {}

    # Helpers -----------------------------------------------------------

    def _current_scope(self) -> str:
        if not self.scope:
            return self.module_qname
        path = "/".join(name for name, _ in self.scope)
        return f"{self.module_qname}::{path}"

    def _qualify(self, name: str) -> str:
        prefix = self._current_scope()
        if prefix == self.module_qname:
            return f"{self.module_qname}::{name}"
        return f"{prefix}/{name}"

    def _add_element(
        self,
        symbol: str,
        element_type: str,
        node: ast.AST,
        signature: Optional[str],
        docstring: Optional[str],
        summary: str,
        tokens: List[str],
    ) -> None:
        span = (
            getattr(node, "lineno", 1),
            getattr(node, "end_lineno", getattr(node, "lineno", 1)),
        )
        qualified_name = self._qualify(symbol)
        element = ParsedElement(
            qualified_name=qualified_name,
            symbol=symbol,
            element_type=element_type,
            signature=signature,
            docstring=docstring,
            summary=summary,
            span=span,
            ast_repr={"type": type(node).__name__, "dump": ast.dump(node, indent=2)},
            tokens=tokens,
        )
        self.elements.append(element)
        bucket = self.defined_names.setdefault(symbol, [])
        bucket.append(qualified_name)
        parent_scope = self._current_scope()
        self.relationships.append(
            ParsedRelationship(
                source=parent_scope,
                target=qualified_name,
                relationship_type="contains" if parent_scope != self.module_qname else "defines",
                attributes={"path": self.rel_path, "line": span[0]},
            )
        )

    def _import_target(self, name: str) -> str:
        parts = name.split(".")
        alias = parts[0]
        resolved = self.import_aliases.get(alias)
        if resolved:
            remainder = ".".join(parts[1:])
            return f"{resolved}.{remainder}".rstrip(".")
        return name

    def _record_call(self, source: str, node: ast.AST) -> None:
        target = self._stringify_call_target(node)
        if not target:
            return
        normalized = self._normalize_target(source, target)
        self.relationships.append(
            ParsedRelationship(
                source=source,
                target=normalized,
                relationship_type="calls",
                attributes={"line": getattr(node, "lineno", None)},
            )
        )

    def _record_data_flow(self, symbol: str, value_node: ast.AST, line: int) -> None:
        targets = {name for name in self._collect_names(value_node)}
        for target in targets:
            normalized = self._normalize_target(self._qualify(symbol), target)
            self.relationships.append(
                ParsedRelationship(
                    source=self._qualify(symbol),
                    target=normalized,
                    relationship_type="data_flow",
                    attributes={"line": line},
                )
            )

    def _collect_names(self, node: ast.AST) -> Iterable[str]:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name):
                yield inner.id
            elif isinstance(inner, ast.Attribute):
                parts: List[str] = []
                current: Optional[ast.AST] = inner
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    yield ".".join(reversed(parts))

    def _stringify_call_target(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            current: Optional[ast.AST] = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _normalize_target(self, source: str, target: str) -> str:
        resolved = self._resolve_defined_name(target, source_scope=source)
        if resolved:
            return resolved
        if target.startswith("self.") or target.startswith("cls."):
            attr = target.split(".", 1)[1]
            class_scope = self._enclosing_class_scope()
            if class_scope:
                candidate = self._resolve_defined_name(attr, class_scope)
                if candidate:
                    return candidate
                return f"{class_scope}/{attr}"
        return self._import_target(target)

    def _resolve_defined_name(self, name: str, source_scope: Optional[str] = None) -> Optional[str]:
        candidates = self.defined_names.get(name)
        if not candidates:
            return None
        if source_scope:
            for candidate in reversed(candidates):
                if candidate.startswith(source_scope):
                    return candidate
        for candidate in reversed(candidates):
            if candidate.startswith(self.module_qname):
                return candidate
        return candidates[-1]

    def _enclosing_class_scope(self) -> Optional[str]:
        if not self.scope:
            return None
        for index in range(len(self.scope) - 1, -1, -1):
            name, scope_type = self.scope[index]
            if scope_type == "class":
                path = "/".join(entry[0] for entry in self.scope[: index + 1])
                return f"{self.module_qname}::{path}"
        return None

    def _function_signature(self, node: ast.AST) -> Optional[str]:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        params: List[str] = []
        for arg in node.args.posonlyargs:
            params.append(arg.arg)
        for arg in node.args.args:
            params.append(arg.arg)
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        for key, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if default is not None:
                params.append(f"{key.arg}=")
            else:
                params.append(key.arg)
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")
        return f"{node.name}({', '.join(params)})"

    # Visitor overrides -------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            target = alias.name
            asname = alias.asname or target.split(".")[0]
            self.import_aliases[asname] = target
            self.relationships.append(
                ParsedRelationship(
                    source=self.module_qname,
                    target=target,
                    relationship_type="imports",
                    attributes={"line": getattr(node, "lineno", None)},
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            full = f"{module}.{alias.name}" if module else alias.name
            asname = alias.asname or alias.name
            self.import_aliases[asname] = full
            self.relationships.append(
                ParsedRelationship(
                    source=self.module_qname,
                    target=full,
                    relationship_type="imports",
                    attributes={"line": getattr(node, "lineno", None)},
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        docstring = ast.get_docstring(node)
        summary = docstring.splitlines()[0] if docstring else f"Class {node.name}"
        tokens = _extract_tokens(summary + (" " + docstring if docstring else ""))
        signature = f"class {node.name}"
        self._add_element(node.name, "class", node, signature, docstring, summary, tokens)

        qualified = self._qualify(node.name)
        for base in node.bases:
            target = self._stringify_call_target(base)
            if target:
                normalized = self._import_target(target)
                self.relationships.append(
                    ParsedRelationship(
                        source=qualified,
                        target=normalized,
                        relationship_type="inherits",
                        attributes={"line": getattr(base, "lineno", getattr(node, "lineno", None))},
                    )
                )

        self.scope.append((node.name, "class"))
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function_like(node)

    def _handle_function_like(self, node: ast.AST) -> None:
        func_node = node  # alias for typing
        docstring = ast.get_docstring(func_node)
        summary = docstring.splitlines()[0] if docstring else f"Function {getattr(func_node, 'name', '<lambda>')}"
        tokens = _extract_tokens(summary + (" " + docstring if docstring else ""))
        signature = self._function_signature(func_node)
        name = getattr(func_node, "name", "lambda")
        self._add_element(name, "function", func_node, signature, docstring, summary, tokens)

        qualified = self._qualify(name)
        for call in [n for n in ast.walk(func_node) if isinstance(n, ast.Call)]:
            self._record_call(qualified, call.func)

        self.scope.append((name, "function"))
        self.generic_visit(func_node)
        self.scope.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = [target for target in node.targets if isinstance(target, ast.Name)]
        if not targets:
            self.generic_visit(node)
            return
        value_line = getattr(node, "lineno", None) or 0
        for target in targets:
            summary = f"Variable {target.id}"
            tokens = _extract_tokens(summary)
            self._add_element(target.id, "variable", target, target.id, None, summary, tokens)
            self._record_data_flow(target.id, node.value, value_line)
        self.generic_visit(node)


def _extract_tokens(text: str) -> List[str]:
    tokens = _TOKEN_PATTERN.findall(text)
    return list(dict.fromkeys(token.lower() for token in tokens))
