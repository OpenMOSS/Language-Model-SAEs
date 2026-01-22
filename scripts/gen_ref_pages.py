"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files


def is_valid_python_package_path(path: Path, src: Path) -> bool:
    """Check if all parent directories have __init__.py files."""
    relative_path = path.relative_to(src)
    current = src

    # Check each directory level (excluding the file itself)
    for part in relative_path.parts[:-1]:
        current = current / part
        # If this is a directory and doesn't have __init__.py, it's not a valid package
        if current.is_dir() and not (current / "__init__.py").exists():
            return False

    return True


nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"

for path in [root / "src/lm_saes/__init__.py"]:
    # Skip files that are not in valid Python packages
    if not is_valid_python_package_path(path, src):
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
