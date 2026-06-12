#!/usr/bin/env python
"""Audit SpatioEv development notebooks without executing heavy workflows.

The check validates notebook JSON, parses Python code cells, records optional
third-party imports that are not available in the current environment, and
verifies that public ``spatioev`` references still resolve through the package
lazy API maps.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import nbformat
    from nbformat.validator import validate
except ImportError as exc:  # pragma: no cover - exercised only without dev deps.
    raise SystemExit(
        "Notebook compatibility audit requires nbformat. "
        "Install development dependencies with `pip install -e '.[dev]'`."
    ) from exc


DEFAULT_NOTEBOOKS = [
    "notebooks/00_dev_seg_qc_testing.ipynb",
    "notebooks/01_dev_clustering_based_phenotyping_test.ipynb",
    "notebooks/01_dev_scimap_phenotype_workflow.ipynb",
    "notebooks/02_dev_SVM_phenotype_probility.ipynb",
    "notebooks/03_dev_general_density.ipynb",
    "notebooks/03_dev_local_density_KNN.ipynb",
    "notebooks/03_dev_local_density_radius.ipynb",
    "notebooks/04_dev_spatial_stats_exp3.ipynb",
    "notebooks/04_dev_spatial_stats.ipynb",
    "notebooks/04_global_organization_PDAC_IgG4AIP.ipynb",
    "notebooks/05_dev_spatial_niche_boundaries.ipynb",
    "notebooks/06_dev_graph_pseudotime_v2_combined_exp_2_3_4_5.ipynb",
    "notebooks/06_dev_graph_pseudotime_v2_exp_2.ipynb",
    "notebooks/06_dev_graph_pseudotime_v2_exp_3.ipynb",
    "notebooks/06_dev_graph_pseudotime_v2_exp_4.ipynb",
    "notebooks/06_dev_graph_pseudotime_v2_exp_5.ipynb",
    "notebooks/07_xenium_00_data_audit_and_spatialdata.ipynb",
    "notebooks/07_xenium_01_cell_annotation.ipynb",
    "notebooks/07_xenium_02_epithelial_niche_features.ipynb",
    "notebooks/07_xenium_03_pooled_pseudotime.ipynb",
    "notebooks/08_RA_OA_ECM_cell_00_prepare_links.ipynb",
    "notebooks/08_RA_OA_ECM_cell_05_chp_density_micro_holes_col6_dark_zone_segmentation.ipynb",
    "notebooks/08_trajectory_microenvironment_interactions.ipynb",
    "notebooks/09_RA_OA_ECM_cell_spatioev_module_paper_applications.ipynb",
    "notebooks/09_xenium_banksy_pseudotime_integration.ipynb",
    "notebooks/10_xenium_spatialcellchat_pseudotime_integration.ipynb",
]

PUBLIC_MODULES = {
    "spatioev",
    "spatioev.hl",
    "spatioev.io",
    "spatioev.pl",
    "spatioev.pp",
    "spatioev.tl",
    "spatioev.xe",
}

LOCAL_OR_STDLIB = {
    "annotations",
    "argparse",
    "ast",
    "collections",
    "csv",
    "dataclasses",
    "datetime",
    "functools",
    "glob",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "random",
    "re",
    "shutil",
    "statistics",
    "subprocess",
    "sys",
    "textwrap",
    "typing",
    "warnings",
}


@dataclass
class NotebookAudit:
    path: Path
    cells: int = 0
    code_cells: int = 0
    references: set[str] = field(default_factory=set)
    missing_references: list[str] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    missing_imports: set[str] = field(default_factory=set)
    syntax_errors: list[str] = field(default_factory=list)
    validation_error: str | None = None

    @property
    def passed(self) -> bool:
        return not self.validation_error and not self.syntax_errors and not self.missing_references


def public_exports() -> dict[str, set[str]]:
    exports: dict[str, set[str]] = {}
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        exports[module_name] = set(getattr(module, "__all__", dir(module)))
    return exports


def clean_code_cell(source: str) -> str:
    stripped = source.lstrip()
    if stripped.startswith("%%"):
        return ""
    lines = []
    for line in source.splitlines():
        clean = line.lstrip()
        if clean.startswith(("%", "!", "?")):
            continue
        if clean.endswith("?") and not clean.startswith(("'", '"', "#")):
            continue
        lines.append(line)
    return "\n".join(lines)


def dotted_name(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return None


def imported_top_level_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def spatioev_aliases(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "spatioev" or alias.name.startswith("spatioev."):
                    alias_name = alias.asname or alias.name.split(".")[-1]
                    aliases[alias_name] = tuple(alias.name.split(".")[1:])
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "spatioev" or node.module.startswith("spatioev."):
                module_parts = tuple(node.module.split(".")[1:])
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    alias_name = alias.asname or alias.name
                    aliases[alias_name] = module_parts + (alias.name,)
    return aliases


def check_spatioev_references(tree: ast.AST, exports: dict[str, set[str]]) -> tuple[set[str], list[str]]:
    aliases = spatioev_aliases(tree)
    references: set[str] = set()
    missing: list[str] = []

    for alias, path in aliases.items():
        if not path:
            continue
        module_name = "spatioev." + ".".join(path[:-1]) if len(path) > 1 else "spatioev"
        attr = path[-1]
        if module_name in exports:
            references.add(f"{module_name}.{attr}")
            if attr not in exports[module_name]:
                missing.append(f"{module_name}.{attr}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = dotted_name(node)
        if not chain or chain[0] not in aliases:
            continue

        imported_path = aliases[chain[0]]
        full = imported_path + chain[1:]
        if not full:
            continue

        module_name = "spatioev"
        attr_index = 0
        if len(full) >= 2 and f"spatioev.{full[0]}" in exports:
            module_name = f"spatioev.{full[0]}"
            attr_index = 1

        if attr_index >= len(full):
            continue
        attr = full[attr_index]
        reference = f"{module_name}.{attr}"
        references.add(reference)
        if module_name in exports and attr not in exports[module_name]:
            missing.append(reference)

    return references, sorted(set(missing))


def missing_optional_imports(imports: set[str]) -> set[str]:
    missing = set()
    for name in sorted(imports):
        if name in LOCAL_OR_STDLIB or name == "spatioev":
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.add(name)
    return missing


def audit_notebook(path: Path, exports: dict[str, set[str]]) -> NotebookAudit:
    audit = NotebookAudit(path=path)
    try:
        notebook = nbformat.read(path, as_version=4)
        validate(notebook)
    except Exception as exc:  # noqa: BLE001 - report validation failures verbatim.
        audit.validation_error = f"{type(exc).__name__}: {exc}"
        return audit

    audit.cells = len(notebook.cells)
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        audit.code_cells += 1
        source = clean_code_cell(cell.source)
        if not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            audit.syntax_errors.append(f"cell {index}: {exc}")
            continue
        audit.imports.update(imported_top_level_names(tree))
        refs, missing = check_spatioev_references(tree, exports)
        audit.references.update(refs)
        audit.missing_references.extend(missing)

    audit.missing_imports = missing_optional_imports(audit.imports)
    audit.missing_references = sorted(set(audit.missing_references))
    return audit


def markdown_report(audits: list[NotebookAudit]) -> str:
    passed = sum(audit.passed for audit in audits)
    syntax_errors = sum(len(audit.syntax_errors) for audit in audits)
    missing_refs = sum(len(audit.missing_references) for audit in audits)

    lines = [
        "# Notebook Compatibility Audit",
        "",
        "This audit validates notebook structure and checks that notebook references to",
        "the public `spatioev` API still resolve. It does not execute heavyweight",
        "image, Xenium, or trajectory workflows.",
        "",
        f"- Notebooks scanned: {len(audits)}",
        f"- Compatibility checks passed: {passed}/{len(audits)}",
        f"- Python syntax errors: {syntax_errors}",
        f"- Missing `spatioev` public API references: {missing_refs}",
        "",
        "| Notebook | Code Cells | SpatioEv Refs | Missing Optional Imports | Status |",
        "| --- | ---: | ---: | --- | --- |",
    ]

    for audit in audits:
        missing = ", ".join(sorted(audit.missing_imports)) or "-"
        status = "pass" if audit.passed else "review"
        lines.append(
            f"| `{audit.path}` | {audit.code_cells} | {len(audit.references)} | {missing} | {status} |"
        )

    reviewed_missing = sorted({item for audit in audits for item in audit.missing_imports})
    if reviewed_missing:
        lines.extend(
            [
                "",
                "## Environment Notes",
                "",
                "The following imports were referenced by one or more notebooks but were not",
                "installed in the current lightweight test environment:",
                "",
                ", ".join(f"`{item}`" for item in reviewed_missing) + ".",
                "",
                "Install the relevant optional analysis stack before full notebook execution.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook paths to audit.")
    parser.add_argument("--report", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()

    notebook_paths = [Path(path) for path in (args.notebooks or DEFAULT_NOTEBOOKS)]
    exports = public_exports()
    audits = [audit_notebook(path, exports) for path in notebook_paths]

    report = markdown_report(audits)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report)

    failures = [audit for audit in audits if not audit.passed]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
