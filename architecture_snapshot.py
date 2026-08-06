#!/usr/bin/env python3
"""
Generate an architecture snapshot for PySeshat.

Outputs
-------
docs/architecture/
    package_tree.txt
    public_api.md
    statistics.md
    class_diagram.puml

The script never imports project modules.
It uses only the Python AST.
"""

from __future__ import annotations

import ast
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[0]

SRC = ROOT / "src"

TESTS = ROOT / "tests"

OUTPUT = ROOT / "docs" / "architecture"

OUTPUT.mkdir(parents=True, exist_ok=True)


###########################################################################
# Helpers
###########################################################################


def package_name(path: Path):

    rel = path.relative_to(SRC)

    return ".".join(rel.with_suffix("").parts)


def tree(directory: Path, prefix=""):

    entries = sorted(directory.iterdir())

    for i, entry in enumerate(entries):

        connector = "└── " if i == len(entries) - 1 else "├── "

        yield prefix + connector + entry.name

        if entry.is_dir():

            extension = "    " if i == len(entries) - 1 else "│   "

            yield from tree(entry, prefix + extension)


###########################################################################
# AST parsing
###########################################################################


class ModuleInfo:

    def __init__(self):

        self.classes = []

        self.functions = []

        self.dataclasses = []

        self.enums = []

        self.abstract = []


modules = {}

stats = defaultdict(int)


for file in SRC.rglob("*.py"):

    stats["modules"] += 1

    module = ModuleInfo()

    source = file.read_text(encoding="utf8")

    tree_ast = ast.parse(source)

    decorators = {}

    for node in tree_ast.body:

        if isinstance(node, ast.ClassDef):

            stats["classes"] += 1

            module.classes.append(node.name)

            base_names = []

            for base in node.bases:

                if isinstance(base, ast.Name):

                    base_names.append(base.id)

                elif isinstance(base, ast.Attribute):

                    base_names.append(base.attr)

            if "Enum" in base_names:

                module.enums.append(node.name)

            if "ABC" in base_names:

                module.abstract.append(node.name)

            for dec in node.decorator_list:

                if isinstance(dec, ast.Name):

                    if dec.id == "dataclass":

                        module.dataclasses.append(node.name)

        elif isinstance(node, ast.FunctionDef):

            stats["functions"] += 1

            module.functions.append(node.name)

    modules[package_name(file)] = module


###########################################################################
# Package tree
###########################################################################

with open(OUTPUT / "package_tree.txt", "w", encoding="utf8") as f:

    f.write("PySeshat\n")

    for line in tree(SRC):

        f.write(line + "\n")


###########################################################################
# API inventory
###########################################################################

with open(OUTPUT / "public_api.md", "w", encoding="utf8") as f:

    f.write("# Public API\n\n")

    for module in sorted(modules):

        info = modules[module]

        f.write(f"## {module}\n\n")

        if info.dataclasses:

            f.write("### Dataclasses\n")

            for cls in info.dataclasses:

                f.write(f"- {cls}\n")

            f.write("\n")

        if info.abstract:

            f.write("### Abstract Base Classes\n")

            for cls in info.abstract:

                f.write(f"- {cls}\n")

            f.write("\n")

        if info.enums:

            f.write("### Enums\n")

            for cls in info.enums:

                f.write(f"- {cls}\n")

            f.write("\n")

        if info.classes:

            f.write("### Classes\n")

            for cls in info.classes:

                f.write(f"- {cls}\n")

            f.write("\n")

        public = [

            fn

            for fn in info.functions

            if not fn.startswith("_")

        ]

        if public:

            f.write("### Functions\n")

            for fn in public:

                f.write(f"- {fn}()\n")

            f.write("\n")


###########################################################################
# Statistics
###########################################################################

stats["tests"] = len(list(TESTS.rglob("test_*.py")))

stats["dataclasses"] = sum(

    len(m.dataclasses)

    for m in modules.values()

)

stats["enums"] = sum(

    len(m.enums)

    for m in modules.values()

)

stats["abstract_classes"] = sum(

    len(m.abstract)

    for m in modules.values()

)

with open(OUTPUT / "statistics.md", "w", encoding="utf8") as f:

    f.write("# Project Statistics\n\n")

    for key in sorted(stats):

        f.write(f"- **{key.replace('_',' ').title()}**: {stats[key]}\n")


###########################################################################
# PlantUML
###########################################################################

with open(OUTPUT / "class_diagram.puml", "w", encoding="utf8") as f:

    f.write("@startuml\n\n")

    for module in sorted(modules):

        info = modules[module]

        package = module.replace(".", "_")

        f.write(f'package "{module}" {{\n')

        for cls in info.classes:

            stereotype = ""

            if cls in info.dataclasses:

                stereotype = " <<dataclass>>"

            elif cls in info.enums:

                stereotype = " <<enum>>"

            elif cls in info.abstract:

                stereotype = " <<abstract>>"

            f.write(f"class {cls}{stereotype}\n")

        f.write("}\n\n")

    f.write("@enduml\n")


print()

print("Architecture snapshot written to")

print()

print(OUTPUT)