import ast, pathlib, sys

root = pathlib.Path(sys.argv[1])
for p in sorted(root.rglob("*.py")):
    if "__pycache__" in str(p):
        continue
    rel = p.relative_to(root.parent)
    src = p.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"## {rel}  [SyntaxError {e}]")
        continue
    print(f"\n## {rel}")
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        print("  > " + mod_doc.strip().splitlines()[0][:150])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            d = (ast.get_docstring(node) or "").strip().splitlines()
            print(f"  def {node.name}({args})" + (f"  # {d[0][:90]}" if d else ""))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            print(f"  class {node.name}({bases})")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = ", ".join(a.arg for a in sub.args.args)
                    d = (ast.get_docstring(sub) or "").strip().splitlines()
                    print(f"      .{sub.name}({args})" + (f"  # {d[0][:80]}" if d else ""))
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    print(f"      .{sub.target.id}: {ast.unparse(sub.annotation)[:60]}")
