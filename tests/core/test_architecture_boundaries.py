from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[2] / "src" / "dominican_eaters"

FORBIDDEN_PREFIXES = {
    "data": (
        "dominican_eaters.cli",
        "dominican_eaters.collection",
        "dominican_eaters.evaluation",
        "dominican_eaters.speech",
    ),
    "collection": (
        "dominican_eaters.cli",
        "dominican_eaters.evaluation",
        "dominican_eaters.speech",
    ),
    "speech": (
        "dominican_eaters.cli",
        "dominican_eaters.collection",
        "dominican_eaters.evaluation",
    ),
    "evaluation": ("dominican_eaters.cli", "dominican_eaters.collection"),
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            modules.add(node.module)
    return modules


def test_package_dependency_boundaries_are_acyclic() -> None:
    violations: list[str] = []
    for area, forbidden in FORBIDDEN_PREFIXES.items():
        for path in (PACKAGE / area).rglob("*.py"):
            for imported in imported_modules(path):
                if imported.startswith(forbidden):
                    violations.append(f"{path.relative_to(PACKAGE)} imports {imported}")

    assert violations == []
