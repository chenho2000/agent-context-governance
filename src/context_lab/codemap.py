"""tokenmax 启发的 Python AST 持久化符号地图；不是 MCP server。"""

import ast
from pathlib import Path
from .types import digest


class CodeMap:
    def __init__(self, store, root):
        self.store, self.root = store, Path(root).resolve()
        self.namespace = "codemap:" + str(self.root)

    def build(self):
        files = {}
        for path in sorted(self.root.rglob("*.py")):
            if any(
                p.startswith(".") or p in {"__pycache__", "node_modules", "runs"}
                for p in path.relative_to(self.root).parts
            ):
                continue
            if not path.resolve().is_relative_to(self.root):
                continue
            text = path.read_text(encoding="utf-8")
            symbols = []
            try:
                tree = ast.parse(text)
            except SyntaxError as error:
                files[str(path.relative_to(self.root))] = {"version": digest(text), "error": str(error)}
                continue

            def walk(node, prefix=""):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = prefix + child.name
                        symbols.append(
                            {
                                "name": name,
                                "kind": type(child).__name__,
                                "start": min([child.lineno] + [d.lineno for d in child.decorator_list]),
                                "end": child.end_lineno,
                                "docstring": ast.get_docstring(child) or "",
                            }
                        )
                        walk(child, name + ".")
                    else:
                        walk(child, prefix)

            walk(tree)
            files[str(path.relative_to(self.root))] = {"version": digest(text), "symbols": symbols}
        self.store.set_block(self.namespace, "files", files)
        return files

    def get_symbol(self, file, symbol):
        path = (self.root / file).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("path outside code root")
        files = self.store.blocks(self.namespace).get("files", {})
        data = files.get(file)
        if not data:
            raise ValueError("file is not indexed; rebuild first")
        text = path.read_text(encoding="utf-8")
        if digest(text) != data["version"]:
            raise ValueError("stale codemap; rebuild before reading")
        matches = [s for s in data.get("symbols", []) if s["name"] == symbol]
        if len(matches) != 1:
            raise ValueError("symbol absent or ambiguous")
        item = matches[0]
        return {
            **item,
            "path": file,
            "version": data["version"],
            "text": "".join(text.splitlines(keepends=True)[item["start"] - 1 : item["end"]]),
        }

    def search(self, query, *, embedder=None, limit=5):
        """Search symbol metadata; retrieve exact versioned source only after selecting a hit."""
        from .retrieval import bm25

        if not query.strip() or limit < 1:
            raise ValueError("query and positive limit required")
        files = self.store.blocks(self.namespace).get("files", {})
        candidates = [
            {**symbol, "path": path, "version": item["version"]}
            for path, item in files.items()
            for symbol in item.get("symbols", [])
        ]
        texts = [c["name"] + " " + c["path"] + " " + c.get("docstring", "") for c in candidates]
        scores = bm25(query, texts)
        if embedder and texts:
            vectors = embedder.encode([query] + texts)
            semantic = [sum(a * b for a, b in zip(vectors[0], v)) for v in vectors[1:]]
            fused = [0.0] * len(texts)
            for ranking in (scores, semantic):
                for rank, i in enumerate(sorted(range(len(texts)), key=lambda i: -ranking[i])):
                    if ranking[i] > 0:
                        fused[i] += 1 / (61 + rank)
            scores = fused
        return [
            {**candidates[i], "score": scores[i]}
            for i in sorted(range(len(candidates)), key=lambda i: -scores[i])[:limit]
            if scores[i] > 0
        ]
