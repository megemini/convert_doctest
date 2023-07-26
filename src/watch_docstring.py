from __future__ import annotations

import ast
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

XDOCTEST_TEST_DIR = Path("./xdoctest_test")


@dataclass
class Location:
    lineno: int
    col_offset: int


def crop_string(text: str, start: Location, end: Location) -> str:
    lines_length = [len(line) + 1 for line in text.splitlines()]
    start_index = sum(lines_length[: start.lineno - 1]) + start.col_offset
    end_index = sum(lines_length[: end.lineno - 1]) + end.col_offset
    return text[start_index:end_index]


class DocstringType(Enum):
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function/Method"

    @staticmethod
    def from_node_type(node_type: type[ast.AST]) -> DocstringType:
        if node_type == ast.Module:
            return DocstringType.MODULE
        elif node_type == ast.ClassDef:
            return DocstringType.CLASS
        elif node_type == ast.FunctionDef:
            return DocstringType.FUNCTION
        else:
            raise ValueError(f"Unknown node type {node_type}")


@dataclass
class Docstring:
    type: DocstringType
    name: str
    source: str
    start: Location
    end: Location
    raw_value: str
    value: str


def extract_docstrings(source: str) -> Iterable[Docstring]:
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            docstring_value = ast.get_docstring(node)
            docstring_type = DocstringType.from_node_type(type(node))
            docstring_name = node.name if not isinstance(node, ast.Module) else "<Mod>"
            if docstring_value is None:
                continue
            if (
                node.body
                and isinstance(
                    node.body[0], ast.Expr
                )  # docstring is the first statement
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring_node = node.body[0].value
                start = Location(docstring_node.lineno, docstring_node.col_offset)
                end = Location(docstring_node.end_lineno, docstring_node.end_col_offset)  # type: ignore
                docstring_raw_value = crop_string(source, start, end)
                yield Docstring(
                    type=docstring_type,
                    name=docstring_name,
                    source=source,
                    start=start,
                    end=end,
                    raw_value=docstring_raw_value,
                    value=docstring_value,
                )


def create_xdoctest_test_dir():
    XDOCTEST_TEST_DIR.mkdir(exist_ok=True)
    gitignore_path = XDOCTEST_TEST_DIR / ".gitignore"
    gitignore_path.write_text("*\n")


def copy_docstring_to_xdoctest_test_dir(path: Path):
    template = """
def test_{name}():
    {docstring}
"""
    content = path.read_text()
    for docstring in extract_docstrings(content):
        test_path = XDOCTEST_TEST_DIR / f"docstring_{docstring.name}.py"
        test_path.write_text(
            template.format(name=docstring.name, docstring=docstring.raw_value)
        )


class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == "created":
            print(f"[Create] {event.src_path}")
            if event.src_path.endswith(".py"):
                copy_docstring_to_xdoctest_test_dir(Path(event.src_path))
        elif event.event_type == "modified":
            print(f"[Modify] {event.src_path}")
            if event.src_path.endswith(".py"):
                copy_docstring_to_xdoctest_test_dir(Path(event.src_path))
        else:
            print(f"Unsupport event: {event.event_type}")


def main():
    create_xdoctest_test_dir()

    path = sys.argv[1]
    observer = Observer()
    event_handler = Handler()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
