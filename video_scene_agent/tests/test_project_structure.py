from pathlib import Path


def test_python_files_stay_under_500_lines():
    repo_root = Path(__file__).resolve().parents[2]
    excluded_parts = {
        ".git",
        ".venv",
        "__pycache__",
        "artifacts",
        "build",
        "dist",
    }
    oversized = []
    for path in repo_root.rglob("*.py"):
        if excluded_parts & set(path.parts):
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > 500:
            oversized.append(f"{path.relative_to(repo_root)}:{line_count}")
    assert oversized == []
