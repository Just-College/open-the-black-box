from __future__ import annotations

import fnmatch
import hashlib
import os
import sys

import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RELEASE_NAME = "open-the-black-box"
OUTPUT_DIR = REPO_ROOT / "release"

VERSION = "v2.7"

# fmt: off
INCLUDE_DIRS = [
    "model",
    "flipflop_demo,py",
    "paper.pdf",
    "plot_fixed_points_transitions.py",
    "plot_model_behavior.py",
    "plot_state_space.py",
    "README.md",
    "requirements.txt",
    "run_flipflop_demo.py",
]

EXCLUDE_DIRS = {
    "__pycache__",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
}

EXCLUDE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "*~",
}
# fmt: on


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name in EXCLUDE_FILES:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_PATTERNS)


def iter_release_files() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE_DIRS:
        path = REPO_ROOT / item
        if not path.exists():
            print(f"Warning: missing include path, skipped: {item}")
            continue
        if path.is_file():
            if not should_exclude(path):
                files.append(path)
            continue
        for root, dirnames, filenames in os.walk(path):
            root_path = Path(root)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not should_exclude(root_path / dirname)
            ]
            for filename in filenames:
                file_path = root_path / filename
                if not should_exclude(file_path):
                    files.append(file_path)
    return sorted(set(files), key=lambda p: p.relative_to(REPO_ROOT).as_posix())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(files: list[Path], zip_name: str) -> str:
    lines = [
        f"release: {zip_name}",
        f"created_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"file_count: {len(files)}",
        "",
        "files:",
    ]
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines.append(f"- {rel}  sha256={file_sha256(path)}")
    return "\n".join(lines) + "\n"


def print_release_tree(files: list[Path]) -> None:
    tree: dict[str, dict] = {}
    file_sizes: dict[str, int] = {}

    for path in files:
        rel_parts = path.relative_to(REPO_ROOT).parts
        node = tree
        for part in rel_parts[:-1]:
            node = node.setdefault(part, {})
        rel_key = "/".join(rel_parts)
        node[rel_parts[-1]] = None
        file_sizes[rel_key] = path.stat().st_size

    def subtree_stats(node: dict, rel_prefix: tuple[str, ...]) -> tuple[int, int]:
        file_count = 0
        total_size = 0
        for name, child in node.items():
            if child is None:
                rel_key = "/".join((*rel_prefix, name))
                file_count += 1
                total_size += file_sizes[rel_key]
            else:
                child_count, child_size = subtree_stats(child, (*rel_prefix, name))
                file_count += child_count
                total_size += child_size
        return file_count, total_size

    def walk(node: dict, prefix: str, rel_prefix: tuple[str, ...]) -> None:
        entries = sorted(
            node.items(), key=lambda item: (item[1] is None, item[0].lower())
        )
        for index, (name, child) in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            if child is None:
                rel_key = "/".join((*rel_prefix, name))
                print(f"{prefix}{connector}{name} ({format_size(file_sizes[rel_key])})")
            else:
                child_count, child_size = subtree_stats(child, (*rel_prefix, name))
                print(
                    f"{prefix}{connector}{name}/ "
                    f"({child_count} files, {format_size(child_size)})"
                )
                walk(child, prefix + next_prefix, (*rel_prefix, name))

    total_size = sum(path.stat().st_size for path in files)
    print("\nRelease contents:")
    print(f". ({len(files)} files, {format_size(total_size)})")
    walk(tree, "", ())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = OUTPUT_DIR / f"{RELEASE_NAME}_{VERSION}.zip"
    files = iter_release_files()
    # manifest = build_manifest(files, zip_path.name)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(REPO_ROOT).as_posix()
            zf.write(path, arcname)
        # zf.writestr("RELEASE_MANIFEST.txt", manifest)

    print(f"Created draft release: {zip_path}")
    print(f"Included files: {len(files)}")
    print_release_tree(files)


if __name__ == "__main__":
    main()
