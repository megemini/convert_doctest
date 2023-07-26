from __future__ import annotations

import argparse
import ast
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Hashable, Iterable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

XDOCTEST_TEST_DIR = Path("./xdoctest_test")


@dataclass(frozen=True)
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


def hash_hex(value: Hashable) -> str:
    return hex(hash(value) & 0xFFFFFFFF)[2:]


def make_relative_path(path: Path) -> Path:
    return path.relative_to(Path(".").absolute())


@dataclass(frozen=True)
class Docstring:
    type: DocstringType
    name: str
    source: str
    path: Path
    start: Location
    end: Location
    raw_value: str
    value: str


def extract_docstrings(source: str, source_path: Path) -> Iterable[Docstring]:
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
                    path=source_path,
                    start=start,
                    end=end,
                    raw_value=docstring_raw_value,
                    value=docstring_value,
                )


def clean_dir(path: Path):
    for child in path.iterdir():
        if child.is_dir():
            clean_dir(child)
            child.rmdir()
        else:
            child.unlink()


def create_xdoctest_test_dir():
    print("[Create] Create xdoctest_test dir")
    XDOCTEST_TEST_DIR.mkdir(exist_ok=True)
    gitignore_path = XDOCTEST_TEST_DIR / ".gitignore"
    gitignore_path.write_text("*\n")
    fake_module_init_path = XDOCTEST_TEST_DIR / "__init__.py"
    fake_module_init_path.write_text("\n")


def clean_xdoctest_test_dir():
    print("[Clean] Clean xdoctest_test dir")

    clean_dir(XDOCTEST_TEST_DIR)


def copy_docstring_to_xdoctest_test_dir(path: Path):
    template = """
def test_{name}():
    {docstring}
    code_src = "{path}"
    code_src_with_lineno_and_offset = "{path}:{start.lineno}:{start.col_offset}"
"""
    content = path.read_text()
    for docstring in extract_docstrings(content, source_path=path):
        docstring_hash = hash_hex(docstring)
        source_path_hash = hash_hex(str(docstring.path))
        test_path = (
            XDOCTEST_TEST_DIR
            / f"{source_path_hash}"
            / f"docstring_{docstring.name}_{docstring_hash}.py"
        )
        test_path.parent.mkdir(exist_ok=True)
        test_path_fake_init = test_path.parent / "__init__.py"
        test_path_fake_init.write_text("\n")
        test_path.write_text(
            template.format(
                name=docstring.name,
                docstring=docstring.raw_value,
                path=docstring.path,
                start=docstring.start,
            )
        )


def copy_docstring_recursive(path: Path):
    print(f"[FirstCopy] Copy docstring from {path}")
    if path.is_dir():
        for child in path.iterdir():
            copy_docstring_recursive(child)
    else:
        if path.suffix == ".py":
            copy_docstring_to_xdoctest_test_dir(path)


class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == "created":
            if event.src_path.endswith(".py"):
                print(f"[Create] {event.src_path}")
                src_path = make_relative_path(Path(event.src_path))
                copy_docstring_to_xdoctest_test_dir(src_path)
        elif event.event_type == "modified":
            if event.src_path.endswith(".py"):
                print(f"[Modify] {event.src_path}")
                src_path = make_relative_path(Path(event.src_path))
                source_path_hash = hash_hex(str(src_path))
                source_test_dir = XDOCTEST_TEST_DIR / source_path_hash
                if source_test_dir.exists():
                    clean_dir(source_test_dir)
                    source_test_dir.rmdir()
                copy_docstring_to_xdoctest_test_dir(src_path)
        elif event.event_type == "deleted":
            if event.src_path.endswith(".py"):
                print(f"[Delete] {event.src_path}")
                src_path = make_relative_path(Path(event.src_path))
                source_path_hash = hash_hex(str(src_path))
                source_test_dir = XDOCTEST_TEST_DIR / source_path_hash
                if source_test_dir.exists():
                    clean_dir(source_test_dir)
                    source_test_dir.rmdir()
        else:
            print(f"Unsupport event: {event.event_type}")


def main():
    create_xdoctest_test_dir()

    parser = argparse.ArgumentParser("watch_docstring")
    parser.add_argument("watch_path", nargs="+")
    args = parser.parse_args()
    watch_paths = [Path(path) for path in args.watch_path]
    observers = []
    for path in watch_paths:
        copy_docstring_recursive(path)
        observer = Observer()
        event_handler = Handler()
        observer.schedule(event_handler, path, recursive=True)
        observers.append(observer)
        observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Stop] Stop watch")
    finally:
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join()
        clean_xdoctest_test_dir()


if __name__ == "__main__":
    main()
