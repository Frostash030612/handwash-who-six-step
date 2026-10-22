"""Static integrity gate for the English source release."""
from pathlib import Path
import ast
import re
import subprocess

root = Path(__file__).resolve().parents[1]
tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
for name in filter(None, tracked):
    path = root / name
    if not path.exists():
        raise SystemExit(f"Missing tracked file: {name}")
    if path.suffix.lower() in {".md", ".py", ".json", ".csv", ".yaml", ".yml", ".toml", ".html", ".js", ".txt", ".typed"}:
        text = path.read_text(encoding="utf-8-sig")
        if re.search("[\u3400-\u9fff]", text):
            raise SystemExit(f"Non-English text in {name}")
        if path.suffix == ".py":
            ast.parse(text, filename=name)
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", text):
                if link.startswith(("https:", "http:", "mailto:", "#")):
                    continue
                if not (path.parent / link.split("#")[0]).exists():
                    raise SystemExit(f"Broken link in {name}: {link}")
    if path.suffix.lower() in {".pt", ".pth", ".safetensors", ".mov", ".mp4", ".zip"}:
        raise SystemExit(f"Unexpected binary asset: {name}")
print("English source integrity passed")
