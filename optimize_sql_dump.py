#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQL Dump Optimizer (MySQL / Postgres)
- Detects compression by magic number (gzip, bzip2, xz, zip, none).
- Extracts CREATE TABLE and INSERT INTO ... VALUES(...) for a specified table (or all).
- Merges inserts into larger batches (default 1000 tuples) to speed up loading.
- Simple, cautious value parser (handles parentheses and quotes).
- Class structure: BaseDBHandler, MySQLHandler, PostgresHandler.
Note: The parser is best-effort -- it is not a full SQL analyzer.
It works well on typical dumps from mysqldump/pg_dump but may have issues with non-standard extensions.
"""

import argparse
import bz2
import configparser
import gzip
import io
import locale
import lzma
import os
import re
import sys
import warnings
import zipfile
from abc import ABC, abstractmethod

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    import mysql.connector
except ImportError:
    mysql = None

try:
    from typing import Callable
    # Set user locale from the operating system
    locale.setlocale(locale.LC_ALL, "")
except (locale.Error, IndexError):
    pass  # Keep default locale if setting fails
import gettext

# -----------------------------------------
global tl
# -----------------------------------------

# ---------- Localization setup ----------
APP_NAME = "optimize_sql_dump"
LOCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locale")
try:
    # Find and load the translation file
    translation = gettext.translation(APP_NAME, localedir=LOCALE_DIR, fallback=True)
    tl: Callable[[str], str] = translation.gettext
except FileNotFoundError:
    # Fallback to default gettext if no .mo file is found
    tl = gettext.gettext

# ---------- compression detection by magic number ----------
MAGIC_TYPES = [
    (b"\x1f\x8b\x08", "gzip"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"PK\x03\x04", "zip"),
]


def detect_compression(path):
    with open(path, "rb") as f:
        head = f.read(10)
    for sig, name in MAGIC_TYPES:
        if head.startswith(sig):
            return name
    return "none"


def open_maybe_compressed(path, mode="rt"):
    c = detect_compression(path)
    text_mode = "b" not in mode
    if c == "gzip":
        f = gzip.open(path, mode)
    elif c == "bzip2":
        f = bz2.open(path, mode)
    elif c == "xz":
        f = lzma.open(path, mode)
    elif c == "zip":
        z = zipfile.ZipFile(path, "r")
        names = z.namelist()
        if not names:
            raise ValueError(tl("Empty zip file"))
        if len(names) > 1:
            warnings.warn(
                tl(
                    "ZIP archive contains multiple files, using only the first one: {name}"
                ).format(name=names[0])
            )
        b = z.open(names[0], "r")
        return (
            io.TextIOWrapper(b, encoding="utf-8", errors="replace") if text_mode else b
        )
    else:
        f = (
            open(path, mode, encoding="utf-8", errors="replace")
            if text_mode
            else open(path, mode)
        )
    return f


class TypeValidator(ABC):
    """Base class for data type validation."""

    @abstractmethod
    def parse_column_definition(self, line: str) -> tuple[str, str] | None:
        """Parses a line from CREATE TABLE and returns (column_name, column_type) or None."""
        pass

    @abstractmethod
    def validate(self, value: str, column_type: str) -> str:
        """Validates a value (as a string) against an SQL column type."""
        pass


class MySQLTypeValidator(TypeValidator):
    """Type validator for MySQL."""

    def parse_column_definition(self, line: str) -> tuple[str, str] | None:
        parts = line.split(maxsplit=2)
        if len(parts) >= 2:
            return parts[0].strip('`"'), parts[1].upper()
        return None

    def validate(self, value: str, column_type: str) -> str:
        if column_type.startswith("DATETIME") or column_type.startswith("TIMESTAMP"):
            if value.strip("'\"") == "":
                return "NULL"
        return value


class PostgresTypeValidator(TypeValidator):
    """Stub for PostgreSQL type validation."""

    def parse_column_definition(self, line: str) -> tuple[str, str] | None:
        parts = line.split(maxsplit=2)
        if len(parts) >= 2:
            return parts[0].strip('"'), parts[1].upper()
        return None

    def validate(self, value: str, column_type: str) -> str:
        return value  # no validation for now


class DatabaseHandler(ABC):
    """Base class for handling database-specific operations."""

    def __init__(self):
        self.create_re = re.compile(r"^(CREATE\s+TABLE\b).*", re.IGNORECASE)
        self.insert_re = re.compile(
            r"^(INSERT\s+INTO\s+)(?P<table>[^\s(]+)", re.IGNORECASE
        )
        self.copy_re = re.compile(r"^(COPY\s+)(?P<table>[^\s(]+)", re.IGNORECASE)
        self.insert_template = "INSERT INTO {table} {cols} VALUES\n{values};\n"
        self.validator: TypeValidator | None = None

    @abstractmethod
    def normalize_table_name(self, raw_name: str) -> str:
        pass

    @abstractmethod
    def detect_db_type(path):
        """Try to detect if the dump is from MySQL or PostgreSQL."""
        with open_maybe_compressed(path, "rt") as f:
            head = f.read(2000)
        if "ENGINE=" in head or "AUTO_INCREMENT" in head:
            return "mysql"
        if "COPY " in head or "WITH OIDS" in head:
            return "postgres"
        return "mysql"  # fallback

    def extract_columns_from_create(self, create_stmt: str) -> str:
        return self._extract_columns_from_create_base(
            create_stmt,
            body_regex=r"\((.*)\)\s*(ENGINE|TYPE|AS|COMMENT|;)",
            fallback_regex=r"CREATE\s+TABLE[^\(]*\((.*)\)\s*;",
            ignore_regex=r"PRIMARY\s+KEY|KEY\s+|UNIQUE\s+|CONSTRAINT\s+",
            quote_char="`",
            process_colname_func=lambda c: c.strip('`"'),
        )

    def _extract_columns_from_create_base(
        self,
        create_stmt: str,
        body_regex: str,
        fallback_regex: str,
        ignore_regex: str,
        quote_char: str,
        process_colname_func,
    ) -> str:
        """Generic extractor that pulls column names from a CREATE TABLE statement.

        Returns a parenthesized column list like "(col1, col2)". The
        process_colname_func is applied to raw column tokens to normalize names.
        """
        if not create_stmt:
            return ""
        m = re.search(body_regex, create_stmt, re.S | re.I)
        if not m:
            m = re.search(fallback_regex, create_stmt, re.S | re.I)
        if not m:
            return ""  # Return empty string if no column definitions are found
        cols_blob = m.group(1)
        cols = []
        for line in cols_blob.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            if re.match(ignore_regex, line, re.I):
                continue
            # First token is column name
            parts = line.split()
            if not parts:
                continue
            raw_col = parts[0]
            col = process_colname_func(raw_col)
            if quote_char:
                cols.append(f"{quote_char}{col}{quote_char}")
            else:
                cols.append(col)
        if not cols:
            return ""
        return f"({', '.join(cols)})"


class MySQLHandler(DatabaseHandler):
    """Handler for MySQL database specifics."""

    def __init__(self):
        super().__init__()
        self.validator = MySQLTypeValidator()

    def normalize_table_name(self, raw_name: str) -> str:
        name = raw_name.strip().strip('`"')
        if "." in name:
            name = name.split(".")[-1]
        return name

    def get_truncate_statement(self, tname: str) -> str:
        return f"TRUNCATE TABLE `{tname}`;\n"

    def get_load_statement(self, tname: str, tsv_path: str, cols_str: str) -> str:
        safe_path = tsv_path.replace("'", "''")
        return (
            f"LOAD DATA LOCAL INFILE '{safe_path}'\n"
            f"INTO TABLE `{tname}`\n"
            f"FIELDS TERMINATED BY '\\t' ENCLOSED BY '' ESCAPED BY '\\\\'\n"
            f"LINES TERMINATED BY '\\n'\n"
            f"{cols_str};\n"
        )

    def extract_columns_with_types_from_create(
        self, create_stmt: str
    ) -> list[tuple[str, str]]:
        m = re.search(
            r"\((.*)\)\s*(ENGINE|TYPE|AS|COMMENT|;)", create_stmt, re.S | re.I
        )
        if not m:
            m = re.search(r"CREATE\s+TABLE[^\(]*\((.*)\)\s*;", create_stmt, re.S | re.I)
        if not m or not self.validator:
            return []
        cols_blob = m.group(1)
        cols_with_types = []
        for line in cols_blob.splitlines():
            line = line.strip().rstrip(",")
            if not line or re.match(
                r"PRIMARY\s+KEY|KEY\s+|UNIQUE\s+|CONSTRAINT\s+", line, re.I
            ):
                continue
            parsed = self.validator.parse_column_definition(line)
            if parsed:
                cols_with_types.append(parsed)
        return cols_with_types

    def extract_full_column_definitions(self, create_stmt: str) -> dict[str, str]:
        m = re.search(
            r"\((.*)\)\s*(ENGINE|TYPE|AS|COMMENT|;)", create_stmt, re.S | re.I
        )
        if not m:
            m = re.search(r"CREATE\s+TABLE[^\(]*\((.*)\)\s*;", create_stmt, re.S | re.I)
        if not m:
            return {}
        cols_blob = m.group(1)
        definitions = {}
        for line in cols_blob.splitlines():
            line = line.strip().rstrip(",")
            if not line or re.match(
                r"PRIMARY\s+KEY|KEY\s+|UNIQUE\s+|CONSTRAINT\s+", line, re.I
            ):
                continue
            col_name = line.split()[0].strip('`"')
            definitions[col_name] = line
        return definitions

    def extract_primary_key(self, create_stmt: str) -> list[str]:
        m = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", create_stmt, re.I)
        if not m:
            return []
        pk_blob = m.group(1)
        pk_cols = [
            re.sub(r"\s*\(\d+\)", "", c.strip()).strip('`"') for c in pk_blob.split(",")
        ]
        return pk_cols

    def extract_columns_from_create(self, create_stmt: str) -> str:
        return self._extract_columns_from_create_base(
            create_stmt,
            body_regex=r"\((.*)\)\s*(ENGINE|TYPE|AS|COMMENT|;)",
            fallback_regex=r"CREATE\s+TABLE[^\(]*\((.*)\)\s*;",
            ignore_regex=r"PRIMARY\s+KEY|KEY\s+|UNIQUE\s+|CONSTRAINT\s+",
            quote_char="`",
            process_colname_func=lambda c: c.strip('`"'),
        )

    @staticmethod
    def detect_db_type(path):
        """Return 'mysql' for this handler. Uses global detection as fallback."""
        try:
            return detect_db_type(path)
        except Exception:
            return "mysql"


class PostgresHandler(DatabaseHandler):
    """Handler for PostgreSQL database specifics."""

    def __init__(self):
        super().__init__()
        self.validator = PostgresTypeValidator()

    @staticmethod
    def detect_db_type(path):
        """Return 'postgres' for this handler. Uses global detection as fallback."""
        try:
            return detect_db_type(path)
        except Exception:
            return "postgres"

    def normalize_table_name(self, raw_name: str) -> str:
        name = raw_name.strip().strip('"')
        if "." in name:
            name = name.split(".")[-1].strip('"')
        return name

    def get_truncate_statement(self, tname: str) -> str:
        return f'TRUNCATE TABLE "{tname}" RESTART IDENTITY CASCADE;\n'

    def get_load_statement(self, tname: str, tsv_path: str, cols_str: str) -> str:
        safe_path = tsv_path.replace("'", "''")
        return f"COPY \"{tname}\" {cols_str} FROM '{safe_path}' WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\n');\n"

    def extract_columns_from_create(self, create_stmt: str) -> str:
        return self._extract_columns_from_create_base(
            create_stmt,
            body_regex=r"CREATE\s+TABLE[^\(]*\(([\s\S]*?)\)\s*(?:WITH|;)",
            ignore_regex=r"PRIMARY\s+KEY|UNIQUE|CONSTRAINT|CHECK",
            quote_char='"',
            process_colname_func=lambda c: c.strip('"'),
            fallback_regex=r"CREATE\s+TABLE[^\(]*\(([\s\S]*?)\)\s*;",
        )


class SqlMultiTupleParser:
    """
    A parser for multiple SQL value tuples, like those in an INSERT's VALUES clause.
    It iterates over a string like "(1, 'a'), (2, 'b')" and yields each full tuple
    string, e.g., "(1, 'a')" and then "(2, 'b')".
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.len = len(text)

    def __iter__(self):
        paren_level = 0
        in_single_quote = False
        in_double_quote = False
        in_backtick = False
        escaped = False
        start = self.text.find("(")
        if start == -1:
            return

        self.pos = start
        start = self.pos

        while self.pos < self.len:
            char = self.text[self.pos]

            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'" and not in_double_quote and not in_backtick:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote and not in_backtick:
                in_double_quote = not in_double_quote
            elif char == "`" and not in_single_quote and not in_double_quote:
                in_backtick = not in_backtick
            elif not (in_single_quote or in_double_quote or in_backtick):
                if char == "(":
                    paren_level += 1
                elif char == ")":
                    paren_level -= 1
                    if paren_level == 0:
                        yield self.text[start: self.pos + 1]
                        # Find the start of the next tuple
                        next_paren = self.text.find("(", self.pos + 1)
                        if next_paren == -1:
                            break
                        self.pos = next_paren
                        start = self.pos
                        continue  # Continue to next char of the new tuple

            self.pos += 1


class SqlTupleFieldParser:
    """
    A parser for a single SQL value tuple. It acts as an iterator that returns
    individual field values from a tuple string, correctly handling quotes,
    parentheses, and escape characters.
    e.g. "(1, 'a', NULL)" yields "1", "'a'", "NULL".
    """

    def __init__(self, text: str):
        # Expects a single tuple string like "(...)"
        self.text = text.strip().rstrip(";")
        if self.text.startswith("(") and self.text.endswith(")"):
            self.text = self.text[1:-1]
        self.pos = 0
        self.len = len(self.text)

    def __iter__(self):
        in_quote = None
        escaped = False
        paren_level = 0
        start = 0
        while self.pos < self.len:
            char = self.text[self.pos]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_quote:
                if char == in_quote:
                    in_quote = None
            elif char in ("'", '"', "`"):
                in_quote = char
            elif char == "(" and not in_quote:
                paren_level += 1
            elif char == ")" and not in_quote:
                paren_level -= 1
            elif char == "," and not in_quote and paren_level == 0:
                yield self.text[start: self.pos].strip()
                start = self.pos + 1
            self.pos += 1
        yield self.text[start:].strip()


def parse_sql_values_to_tsv_row(sql_tuple):
    """
    Convert an SQL tuple string like "(1, 'a', NULL, '\\n')" to a TSV row string.
    """
    fields = list(SqlTupleFieldParser(sql_tuple))
    tsv_fields = []
    for field in fields:
        if field == "NULL":
            # \n is the default representation of NULL for LOAD DATA INFILE
            tsv_fields.append("\\n")
        else:
            # Remove surrounding single quotes if present
            if field.startswith("'") and field.endswith("'"):
                inner = field[1:-1]
                # SQL string literal escapes use backslashes; we must convert them to literal sequences
                # The goal is to un-escape what SQL escapes for its string literals.
                # \\ -> \
                # \' -> '
                # \" -> "
                # \n -> newline character
                # \t -> tab character
                # etc.
                # The order of replacement is important. Replace \\ first.
                inner = inner.replace("\\\\", "\\")
                inner = inner.replace("\\'", "'")
                inner = inner.replace('\\"', '"')
                inner = (
                    inner.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\r", "\r")
                    .replace("\\b", "\b")
                    .replace("\\Z", "\x1a")
                )
                tsv_fields.append(inner)
            else:
                tsv_fields.append(field)
    # Preserve trailing newline behavior: tests expect a trailing newline in output.
    return "\t".join(tsv_fields)  # + "\n"


# ---------- SQL Parser ----------
progress = None  # global for tqdm


class StatementParser:
    def __init__(self, stream):
        self.stream = stream
        self.buf = []
        self.in_single_quote = False
        self.in_double_quote = False
        self.in_backtick = False
        self.is_escaped = False
        self.paren_level = 0
        self.in_copy_data = False
        self.copy_table_name = None

    def __iter__(self):
        while True:
            chunk = self.stream.read(4096)
            if not chunk:
                if self.buf:
                    yield "other", "".join(self.buf)
                break
            if progress:
                progress.update(len(chunk))
            if self.in_copy_data:
                yield from self._process_copy_chunk(chunk)
            else:
                yield from self._process_normal_chunk(chunk)

    def _process_copy_chunk(self, chunk):
        text_chunk = "".join(self.buf) + chunk
        self.buf = []
        parts = text_chunk.split("\n\\.", 1)
        if len(parts) > 1:
            data, rest = parts
            yield "copy_data", (self.copy_table_name, data + "\n")
            self.in_copy_data = False
            self.copy_table_name = None
            yield from self._process_normal_chunk(rest.lstrip(";\n"))
        else:
            self.buf.append(text_chunk)

    def _process_char(self, char):
        self.buf.append(char)
        if self.is_escaped:
            self.is_escaped = False
            return None
        if char == "\\":
            self.is_escaped = True
        elif char == "'" and not self.in_double_quote and not self.in_backtick:
            self.in_single_quote = not self.in_single_quote
        elif char == '"' and not self.in_single_quote and not self.in_backtick:
            self.in_double_quote = not self.in_double_quote
        elif char == "`" and not self.in_single_quote and not self.in_double_quote:
            self.in_backtick = not self.in_backtick
        elif not (self.in_single_quote or self.in_double_quote or self.in_backtick):
            if char == "(":
                self.paren_level += 1
            elif char == ")":
                if self.paren_level > 0:
                    self.paren_level -= 1
            elif char == ";" and self.paren_level == 0:
                return self._emit_statement()
        return None

    def _process_normal_chunk(self, chunk):
        for char in chunk:
            statement = self._process_char(char)
            if statement:
                yield statement

    def _emit_statement(self):
        text = "".join(self.buf).strip()
        self.buf = []
        (
            self.in_single_quote,
            self.in_double_quote,
            self.in_backtick,
            self.is_escaped,
            self.paren_level,
        ) = False, False, False, False, 0
        t = text.lstrip()[:30].upper()
        if t.startswith("CREATE TABLE") or t.startswith("CREATE TEMPORARY TABLE"):
            return "create", text
        elif t.startswith("INSERT INTO"):
            return "insert", text
        elif t.startswith("COPY ") and " FROM STDIN" in t.upper():
            m = re.search(r"COPY\s+(?P<name>[^\s\(]+)", text, re.I)
            if m:
                self.copy_table_name = m.group("name").strip().strip('"')
                self.in_copy_data = True
                return "copy_start", text
        return "other", text


def iter_statements(stream):
    """Yields statements from a SQL dump stream."""
    parser = StatementParser(stream)
    yield from parser


def extract_table_from_insert(stmt, handler: DatabaseHandler):
    m = handler.insert_re.search(stmt)
    if not m:
        m = handler.copy_re.search(stmt)
    return handler.normalize_table_name(m.group("table")) if m else None


def extract_values_from_insert(stmt):
    idx = re.search(r"\bVALUES\b", stmt, re.I)
    if not idx:
        return None, None
    start = idx.end()
    body = stmt[start:].strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    return stmt[:start], body


# ---------- main functionality ----------
def detect_db_type(path):
    """Try to detect if the dump is from MySQL or PostgreSQL."""
    with open_maybe_compressed(path, "rt") as f:
        head = f.read(2000)
    if "ENGINE=" in head or "AUTO_INCREMENT" in head:
        return "mysql"
    if "COPY " in head or "WITH OIDS" in head:
        return "postgres"
    return "mysql"  # fallback


class DumpOptimizer:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.handler = self._setup_handler()
        self.progress = self._setup_progress()
        self.split_mode = bool(self.args.get("split_dir"))
        self.load_data_mode = bool(self.args.get("load_data_dir"))
        self.insert_only_mode = bool(self.args.get("insert_only"))
        self.output_dir = (
            self.args.get("split_dir")
            or self.args.get("load_data_dir")
            or self.args.get("insert_only")
        )
        self.file_map = {}
        self.fout = None
        self.create_map = {}
        self.insert_buffers = {}
        self.total_rows = 0
        self.total_batches = 0

    def _setup_handler(self):
        db_type = self.args.get("db_type", "auto")
        if db_type == "auto":
            db_type = detect_db_type(self.args["inpath"])
            if self.args.get("verbose"):
                print(tl("[INFO] Detected DB type: {db_type}").format(db_type=db_type))
        return MySQLHandler() if db_type == "mysql" else PostgresHandler()

    def _setup_progress(self):
        global progress
        filesize = os.path.getsize(self.args["inpath"])
        if self.args.get("verbose") and tqdm:
            progress = tqdm(
                total=filesize, unit="B", unit_scale=True, desc=tl("Processing")
            )
        else:
            progress = None
        return progress

    def _get_writer(self, tname):
        if tname not in self.file_map:
            if self.load_data_mode:
                sql_fname = os.path.join(self.output_dir, f"{tname}.sql")
                tsv_fname = os.path.join(self.output_dir, f"{tname}.tsv")
                self.file_map[tname] = {
                    "sql": open(sql_fname, "w", encoding="utf-8"),
                    "tsv": open(tsv_fname, "w", encoding="utf-8"),
                    "tsv_path": os.path.abspath(tsv_fname),
                    "tsv_buffer": [],
                }
                if self.args.get("verbose"):
                    print(
                        tl(
                            "[INFO] Created files for table {tname}: {sql_fname}, {tsv_fname}"
                        ).format(tname=tname, sql_fname=sql_fname, tsv_fname=tsv_fname)
                    )
            elif self.split_mode or self.insert_only_mode:
                fname = os.path.join(self.output_dir, f"{tname}.sql")
                writer = open(fname, "w", encoding="utf-8")
                if self.insert_only_mode:
                    writer.write(self.handler.get_truncate_statement(tname))
                self.file_map[tname] = writer
                if self.args.get("verbose"):
                    print(tl("[INFO] Created file {fname}").format(fname=fname))
        if self.load_data_mode:
            return self.file_map[tname]["sql"]
        if self.split_mode or self.insert_only_mode:
            return self.file_map[tname]
        return None

    def _flush_insert_buffer(self, tname):
        buf = self.insert_buffers.get(tname)
        if not buf or not buf["tuples"]:
            return
        cols = buf.get("cols_text", "")
        writer = self._get_writer(tname) if self.split_mode else self.fout
        writer.write(
            f"INSERT INTO {self.handler.normalize_table_name(tname)} {cols} VALUES\n"
        )
        writer.write(",\n".join(f"({v})" for v in buf["tuples"]))
        writer.write(";\n")
        self.total_batches += 1
        buf["tuples"].clear()

    def _flush_tsv_buffer(self, tname, force=False):
        if tname not in self.file_map:
            return
        tsv_info = self.file_map[tname]
        tsv_buffer_size = self.args.get("tsv_buffer_size", 200)
        if tsv_info.get("tsv_buffer") and (
            force or len(tsv_info["tsv_buffer"]) >= tsv_buffer_size
        ):
            tsv_info["tsv"].write("\n".join(tsv_info["tsv_buffer"]) + "\n")
            tsv_info["tsv_buffer"].clear()

    def _handle_create(self, stmt):
        m = re.search(
            r"CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?(?P<name>[^\s\(;]+)", stmt, re.I
        )
        if not m:
            return
        tname = self.handler.normalize_table_name(m.group("name").strip())
        self.create_map[tname] = stmt
        if self.insert_only_mode:
            return
        target_table = self.args.get("target_table")
        if not target_table or tname == target_table:
            if self.split_mode or self.load_data_mode:
                writer = self._get_writer(tname)
                if self.load_data_mode and "IF NOT EXISTS" not in stmt.upper()[:100]:
                    create_stmt = re.sub(
                        r"CREATE\s+TABLE",
                        "CREATE TABLE IF NOT EXISTS",
                        stmt,
                        count=1,
                        flags=re.I,
                    )
                    writer.write(create_stmt.strip() + ";\n")
                else:
                    writer.write(stmt.strip() + "\n")
            else:
                self.fout.write(stmt.strip() + "\n")

    # def _parse_single_tuple_to_fields(self, tuple_str: str) -> list[str | None]:
    #     # This is a simplified parser for the diffing logic, not for TSV generation.
    #     return self._parse_single_tuple_to_fields_standalone(tuple_str)

    def _values_to_tsv_row(self, values: list[str | None]) -> str:
        processed_values = []
        for v in values:
            # For LOAD DATA, NULL is \n. Backslash, tab, newline must be escaped.
            processed_values.append(
                "\\n"
                if v is None
                else str(v)
                .replace("\\", "\\\\")
                .replace("\t", "\\t")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
        return "\t".join(processed_values)

    def _handle_insert(self, stmt):
        target_table = self.args.get("target_table")
        tname = extract_table_from_insert(stmt, self.handler)
        if not tname or (target_table and tname != target_table):
            return
        if self.progress:
            self.progress.set_description(
                tl("Processing table: {tname}").format(tname=tname)
            )
        prefix, values_body = extract_values_from_insert(stmt)
        if self.load_data_mode:
            if values_body:
                count = 0
                try:
                    for tuple_str in SqlMultiTupleParser(values_body):
                        self.file_map[tname]["tsv_buffer"].append(
                            parse_sql_values_to_tsv_row(tuple_str)
                        )
                        count += 1
                    self._flush_tsv_buffer(tname)
                    self.total_rows += count
                except Exception as e:
                    if self.args.get("verbose"):
                        print(
                            tl(
                                "[WARN] Failed to parse VALUES in INSERT for table {tname}: {error}"
                            ).format(tname=tname, error=e)
                        )
            return
        writer = (
            self._get_writer(tname)
            if (self.split_mode or self.insert_only_mode)
            else self.fout
        )
        if not prefix or not values_body:
            writer.write(stmt)
            return
        cols_match = re.search(
            r"INSERT\s+INTO\s+[^\(]+(\([^\)]*\))\s*VALUES", prefix, re.I | re.S
        )
        cols_text = (
            cols_match.group(1).strip()
            if cols_match
            else self.handler.extract_columns_from_create(
                self.create_map.get(tname, "")
            )
        )
        try:
            tuples = list(SqlMultiTupleParser(values_body)) if values_body else []
        except Exception as e:
            if self.args.get("verbose"):
                print(
                    tl(
                        "[WARN] Failed to parse VALUES in INSERT for table {tname}: {error}"
                    ).format(tname=tname, error=e)
                )
            tuples = []
        if not tuples:
            writer.write(stmt)
            return
        buf = self.insert_buffers.setdefault(
            tname, {"cols_text": cols_text, "tuples": []}
        )
        buf["tuples"].extend(tuples)
        self.total_rows += len(tuples)
        if len(buf["tuples"]) >= self.args.get("batch_size", 1000):
            self._flush_insert_buffer(tname)

    def _handle_copy(self, stype, stmt):
        if not self.load_data_mode:
            return
        target_table = self.args.get("target_table")
        if stype == "copy_start":
            tname = extract_table_from_insert(stmt, self.handler)
            if not tname or (target_table and tname != target_table):
                return
            self._get_writer(tname)
        elif stype == "copy_data":
            tname, data = stmt
            if tname in self.file_map:
                self.file_map[tname]["tsv"].write(data)  # data already has a newline
                self.total_rows += data.count("\n")

    def _handle_other(self, stmt):
        target_table = self.args.get("target_table")
        if not self.insert_only_mode and (not target_table or (target_table in stmt)):
            if not self.split_mode and not self.load_data_mode:
                self.fout.write(stmt)

    def run(self):
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.fout = (
                open(self.args["outpath"], "w", encoding="utf-8")
                if not self.args.get("dry_run")
                else open(os.devnull, "w")
            )
            if self.fout:
                self.fout.write(tl("-- Optimized by SqlDumpOptimizer\n"))
                self.fout.write(
                    tl("-- Source: {source}\n").format(
                        source=os.path.basename(self.args["inpath"])
                    )
                )
                self.fout.write("--\n")
        with open_maybe_compressed(self.args["inpath"], "rt") as fin:
            for stype, stmt in iter_statements(fin):
                if stype == "create":
                    self._handle_create(stmt)
                elif stype == "insert":
                    self._handle_insert(stmt)
                elif stype in ("copy_start", "copy_data"):
                    self._handle_copy(stype, stmt)
                else:
                    self._handle_other(stmt)
            for t in list(self.insert_buffers.keys()):
                self._flush_insert_buffer(t)
        self.finalize()

    def finalize(self):
        if self.split_mode or self.load_data_mode or self.insert_only_mode:
            if self.load_data_mode:
                for tname in self.file_map.keys():
                    self._flush_tsv_buffer(tname, force=True)
                if self.file_map:
                    for tname, writers in self.file_map.items():
                        cols_str = self.handler.extract_columns_from_create(
                            self.create_map.get(tname, "")
                        )
                        load_stmt = self.handler.get_load_statement(
                            tname, writers["tsv_path"], cols_str
                        )
                        writers["sql"].write(load_stmt)
                        writers["sql"].close()
                        writers["tsv"].close()
            else:
                for f in self.file_map.values():
                    f.close()
        else:
            if self.fout:
                self.fout.close()
        if self.args.get("verbose"):
            print(
                tl("[INFO] Wrote {rows} records in {batches} batches.").format(
                    rows=self.total_rows, batches=self.total_batches
                )
            )
        if (
            not self.args.get("dry_run")
            and not self.split_mode
            and not self.load_data_mode
        ):
            print(tl("Done. Saved to: {path}").format(path=self.args["outpath"]))
        elif self.split_mode:
            print(
                tl("Done. Split dump into files in directory: {path}").format(
                    path=self.args["split_dir"]
                )
            )
        elif self.load_data_mode:
            print(
                tl("Done. Generated files for import in directory: {path}").format(
                    path=self.args["load_data_dir"]
                )
            )
        elif self.insert_only_mode:
            print(
                tl("Done. Generated insert-only files in directory: {path}").format(
                    path=self.args["insert_only"]
                )
            )
        if self.progress:
            self.progress.close()


class DumpAnalyzer:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.handler = self._setup_handler()
        self.progress = self._setup_progress()
        self.stats = {}

    def _setup_handler(self):
        db_type = self.args.get("db_type", "auto")
        if db_type == "auto":
            db_type = detect_db_type(self.args["inpath"])
            if self.args.get("verbose"):
                print(tl("[INFO] Detected DB type: {db_type}").format(db_type=db_type))
        return MySQLHandler() if db_type == "mysql" else PostgresHandler()

    def _setup_progress(self):
        global progress
        filesize = os.path.getsize(self.args["inpath"])
        if self.args.get("verbose") and tqdm:
            progress = tqdm(
                total=filesize, unit="B", unit_scale=True, desc=tl("Analyzing dump")
            )
        else:
            progress = None
        return progress

    def run(self):
        with open_maybe_compressed(self.args["inpath"], "rt") as fin:
            for stype, stmt in iter_statements(fin):
                if stype == "create":
                    m = re.search(
                        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(?P<name>[^`\s\(;]+)`?",
                        stmt,
                        re.I,
                    )
                    if m:
                        tname = self.handler.normalize_table_name(m.group("name"))
                        if tname not in self.stats:
                            self.stats[tname] = {"rows": 0, "inserts": 0}
                elif stype == "insert":
                    tname = extract_table_from_insert(stmt, self.handler)
                    if tname and tname in self.stats:
                        self.stats[tname]["inserts"] += 1
                        _, values_body = extract_values_from_insert(stmt)
                        if values_body:
                            self.stats[tname]["rows"] += len(
                                list(SqlMultiTupleParser(values_body))
                            )
        if self.progress:
            self.progress.close()
        self.print_summary()

    def print_summary(self):
        print("\n" + tl("--- Dump Analysis Summary ---"))
        print(f"{'Table':<40} {'INSERT Statements':>20} {'Total Rows':>20}")
        print("-" * 82)
        total_rows = 0
        for tname, data in sorted(self.stats.items()):
            print(f"{tname:<40} {data['inserts']:>20,d} {data['rows']:>20,d}")
            total_rows += data["rows"]
        print("-" * 82)
        print(
            tl("Found {num_tables} tables with a total of {total_rows} rows.").format(
                num_tables=len(self.stats), total_rows=f"{total_rows:,d}"
            )
        )
        print("---------------------------\n")


def escape_sql_value(val, prefix_str: str = "") -> str:
    """
    Returns a correctly formatted SQL fragment for a default value (DEFAULT ...).

    - For strings ('str'), it escapes single quotes and backslashes, e.g.:
        "O'Reilly\\Test" → DEFAULT 'O''Reilly\\\\Test'
    - For numeric values or None, it returns the corresponding SQL representation.
    """
    if val is None:
        return f"{prefix_str} NULL"

    if isinstance(val, str):
        # Escape special characters for SQL
        safe_val = val.replace("'", "''").replace("\\", "\\\\")
        return f"{prefix_str} '{safe_val}'"

    # For numeric types, bool, etc.
    return f"{prefix_str} {val}"


class DatabaseDiffer:
    def __init__(self, **kwargs):
        self.args = kwargs
        if kwargs.get("db_type") not in (None, "mysql", "auto"):
            raise ValueError("DatabaseDiffer supports only MySQL.")
        self.handler = MySQLHandler()
        self.progress = self._setup_progress()
        self.connection = None
        self.cursor = None
        self.create_map = {}
        self.summary = {
            "tables_created": 0,
            "tables_altered": 0,
            "rows_inserted": 0,
            "rows_updated": 0,
            "rows_deleted": 0,
        }

    def _setup_progress(self):
        global progress
        filesize = os.path.getsize(self.args["inpath"])
        if self.args.get("verbose") and tqdm:
            progress = tqdm(
                total=filesize,
                unit="B",
                unit_scale=True,
                desc=tl("Parsing dump for diff"),
            )
        else:
            progress = None
        return progress

    def connect_db(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.args["db_host"],
                user=self.args["db_user"],
                password=self.args["db_password"],
                database=self.args["db_name"],
            )
            self.cursor = self.connection.cursor(dictionary=True)
            if self.args.get("verbose"):
                print(
                    tl(
                        "[INFO] Successfully connected to database '{db}' on {host}"
                    ).format(db=self.args["db_name"], host=self.args["db_host"])
                )
        except mysql.connector.Error as err:
            print(tl("[ERROR] Database connection failed: {error}").format(error=err))
            sys.exit(1)

    def get_db_schema(self, table_name: str) -> dict[str, dict]:
        if not self.connection:
            return {}
        query = """
            SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, EXTRA, CHARACTER_SET_NAME, COLLATION_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION;
        """
        self.cursor.execute(query, (self.args["db_name"], table_name))
        schema = {row["COLUMN_NAME"]: row for row in self.cursor.fetchall()}
        return schema

    def compare_schemas(
        self, dump_cols: dict[str, str], db_cols: dict[str, dict], table_name: str
    ) -> list[str]:
        alter_statements = []
        last_col = None
        for col_name, dump_def in dump_cols.items():
            if col_name not in db_cols:
                position = f" AFTER `{last_col}`" if last_col else " FIRST"
                alter_statements.append(
                    f"ALTER TABLE `{table_name}` ADD COLUMN {dump_def}{position};"
                )
            else:
                db_col_info = db_cols[col_name]
                db_def_parts = [f"`{col_name}`", db_col_info["COLUMN_TYPE"]]
                if db_col_info["CHARACTER_SET_NAME"]:
                    db_def_parts.append(
                        f"CHARACTER SET {db_col_info['CHARACTER_SET_NAME']}"
                    )
                if db_col_info["COLLATION_NAME"]:
                    db_def_parts.append(f"COLLATE {db_col_info['COLLATION_NAME']}")
                db_def_parts.append(
                    "NOT NULL" if db_col_info["IS_NULLABLE"] == "NO" else "NULL"
                )
                if db_col_info["COLUMN_DEFAULT"] is not None:
                    default_val = db_col_info["COLUMN_DEFAULT"]
                    db_def_parts.append(escape_sql_value(default_val, "DEFAULT"))
                elif db_col_info["IS_NULLABLE"] == "YES":
                    db_def_parts.append("DEFAULT NULL")
                if db_col_info["EXTRA"]:
                    db_def_parts.append(db_col_info["EXTRA"])
                normalized_dump_def = " ".join(dump_def.lower().split())
                # normalized_db_def = " ".join(" ".join(db_def_parts).lower().split())
                if db_col_info["COLUMN_TYPE"].lower() not in normalized_dump_def:
                    alter_statements.append(
                        f"ALTER TABLE `{table_name}` MODIFY COLUMN {dump_def};"
                    )
            last_col = col_name
        return alter_statements

    def get_db_primary_keys(self, table_name: str, pk_cols: list[str]) -> set:
        if not self.connection or not pk_cols:
            return set()
        pk_cols_str = ", ".join([f"`{c}`" for c in pk_cols])
        query = f"SELECT {pk_cols_str} FROM `{table_name}`"
        self.cursor.execute(query)
        keys = set()
        for row in self.cursor.fetchall():
            pk_tuple = tuple(str(row[c]) for c in pk_cols)
            keys.add(pk_tuple)
        if self.args.get("verbose"):
            print(
                tl("[INFO] Fetched {count} primary keys for table `{tname}`.").format(
                    count=len(keys), tname=table_name
                )
            )
        return keys

    def get_db_row_by_pk(
        self, table_name: str, pk_cols: list[str], pk_values: tuple
    ) -> dict | None:
        if not self.connection or not pk_cols or len(pk_cols) != len(pk_values):
            return None
        where_clause = " AND ".join([f"`{col}` = %s" for col in pk_cols])
        query = f"SELECT * FROM `{table_name}` WHERE {where_clause}"
        self.cursor.execute(query, pk_values)
        return self.cursor.fetchone()

    def compare_data_row(
        self, dump_row: dict, db_row: dict, table_name: str, pk_cols: list[str]
    ) -> str | None:
        updates = []
        params = []
        pk_values = []
        for col_name, dump_val in dump_row.items():
            db_val = db_row.get(col_name)
            dump_val_str = str(dump_val) if dump_val is not None else None
            db_val_str = str(db_val) if db_val is not None else None
            if dump_val_str != db_val_str:
                if col_name not in pk_cols:
                    updates.append(f"`{col_name}` = %s")
                    params.append(dump_val)
        if not updates:
            return None
        for col in pk_cols:
            pk_values.append(dump_row[col])
        params.extend(pk_values)
        where_clause = " AND ".join([f"`{col}` = %s" for col in pk_cols])
        update_stmt = (
            f"UPDATE `{table_name}` SET {', '.join(updates)} WHERE {where_clause};"
        )

        def format_value(v):
            if v is None:
                return "NULL"
            if isinstance(v, (int, float)):
                return str(v)
            return escape_sql_value(v)

        final_params = [format_value(p) for p in params]
        return update_stmt.replace("%s", "{}").format(*final_params)

    def run(self):
        self.connect_db()
        with (
            open_maybe_compressed(self.args["inpath"], "rt") as fin,
            open(self.args["outpath"], "w", encoding="utf-8") as fout,
        ):
            fout.write(
                f"-- Diff generated by SqlDumpOptimizer\n"
                f"-- Source: {os.path.basename(self.args['inpath'])}\n"
                f"-- Database: {self.args['db_name']}\n--\n"
            )
            for stype, stmt in iter_statements(fin):
                if stype == "create":
                    m = re.search(
                        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(?P<name>[^`\s\(;]+)`?",
                        stmt,
                        re.I,
                    )
                    if not m:
                        continue
                    tname = self.handler.normalize_table_name(m.group("name"))
                    self.create_map[tname] = {
                        "stmt": stmt,
                        "cols": [],
                        "pk": [],
                        "exists_in_db": True,
                    }
                    dump_cols = self.handler.extract_full_column_definitions(stmt)
                    self.create_map[tname]["cols"] = list(dump_cols.keys())
                    self.create_map[tname]["pk"] = self.handler.extract_primary_key(
                        stmt
                    )
                    if progress:
                        progress.set_description(
                            tl("Diffing schema for {tname}").format(tname=tname)
                        )
                    db_cols = self.get_db_schema(tname)
                    if not db_cols:
                        if not self.args.get("insert_only"):
                            self.summary["tables_created"] += 1
                            fout.write(
                                f"-- Table `{tname}` does not exist in the database.\n{stmt};\n"
                            )
                        self.create_map[tname]["exists_in_db"] = False
                    else:
                        if not self.args.get("insert_only"):
                            alter_statements = self.compare_schemas(
                                dump_cols, db_cols, tname
                            )
                            if alter_statements:
                                self.summary["tables_altered"] += 1
                                fout.write(f"-- Schema changes for table `{tname}`\n")
                                fout.write("\n".join(alter_statements))
                                fout.write("\n")
                elif stype == "insert" and self.args.get("diff_data"):
                    tname = extract_table_from_insert(stmt, self.handler)
                    if not tname or tname not in self.create_map:
                        continue
                    table_info = self.create_map[tname]
                    if not table_info["exists_in_db"]:
                        _, values_body = extract_values_from_insert(stmt)
                        if not values_body:
                            continue
                        for tuple_str in SqlMultiTupleParser(values_body):
                            self.summary["rows_inserted"] += 1
                            fout.write(f"INSERT INTO `{tname}` VALUES {tuple_str};\n")
                        continue
                    if not table_info.get("pk"):
                        if self.args.get("verbose"):
                            print(
                                tl(
                                    "[WARN] Skipping data diff for table `{tname}`: no primary key found."
                                ).format(tname=tname)
                            )
                        table_info["pk_checked"] = True
                        continue
                    if "db_pks" not in table_info:
                        if progress:
                            progress.set_description(
                                tl("Diffing data for {tname}").format(tname=tname)
                            )
                        table_info["db_pks"] = self.get_db_primary_keys(
                            tname, table_info["pk"]
                        )
                        table_info["dump_pks"] = set()
                    _, values_body = extract_values_from_insert(stmt)
                    if not values_body:
                        continue
                    for tuple_str in SqlMultiTupleParser(values_body):
                        dump_row_list = self._parse_single_tuple_to_fields(tuple_str)
                        dump_row_dict = dict(zip(table_info["cols"], dump_row_list))
                        pk_values = tuple(
                            str(dump_row_dict.get(c)) for c in table_info["pk"]
                        )
                        table_info["dump_pks"].add(pk_values)
                        if pk_values not in table_info["db_pks"]:
                            self.summary["rows_inserted"] += 1
                            fout.write(
                                f"INSERT INTO `{tname}` ({', '.join(f'`{c}`' for c in table_info['cols'])}) VALUES ({', '.join(self._format_sql_value(v) for v in dump_row_list)});\n"
                            )
                        elif not self.args.get("insert_only"):
                            db_row = self.get_db_row_by_pk(
                                tname, table_info["pk"], pk_values
                            )
                            if db_row:
                                update_stmt = self.compare_data_row(
                                    dump_row_dict, db_row, tname, table_info["pk"]
                                )
                                if update_stmt:
                                    self.summary["rows_updated"] += 1
                                    fout.write(f"{update_stmt}\n")
            if progress:
                progress.set_description(tl("Generating DELETE statements"))
            if self.args.get("diff_data") and not self.args.get("insert_only"):
                fout.write(
                    "\n-- Deleting rows that exist in the database but not in the dump\n"
                )
                for tname, table_info in self.create_map.items():
                    if "db_pks" in table_info and "dump_pks" in table_info:
                        pks_to_delete = table_info["db_pks"] - table_info["dump_pks"]
                        if pks_to_delete:
                            pk_cols = table_info["pk"]
                            self.summary["rows_deleted"] += len(pks_to_delete)
                            for pk_tuple in pks_to_delete:
                                where_clause = " AND ".join(
                                    f"`{col}` = {self._format_sql_value(val)}"
                                    for col, val in zip(pk_cols, pk_tuple)
                                )
                                fout.write(
                                    f"DELETE FROM `{tname}` WHERE {where_clause};\n"
                                )
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
        if self.progress:
            self.progress.close()
        self.display_info()

    def display_info(self):
        print(tl("Done. Diff saved to: {path}").format(path=self.args["outpath"]))
        print("\n" + tl("--- Diff Summary ---"))
        if not self.args.get("insert_only"):
            print(
                tl("Tables to create: {count}").format(
                    count=self.summary["tables_created"]
                )
            )
            print(
                tl("Tables to alter: {count}").format(
                    count=self.summary["tables_altered"]
                )
            )
        if self.args.get("diff_data"):
            print(
                tl("Rows to insert: {count}").format(count=self.summary["rows_inserted"])
            )
            if not self.args.get("insert_only"):
                print(
                    tl("Rows to update: {count}").format(
                        count=self.summary["rows_updated"]
                    )
                )
                print(
                    tl("Rows to delete: {count}").format(
                        count=self.summary["rows_deleted"]
                    )
                )
        print("--------------------\n")

    def _parse_single_tuple_to_fields(self, tuple_str: str) -> list[str | None]:
        parser = SqlTupleFieldParser(tuple_str)
        fields: list[str | None] = []
        for field_str in parser:
            stripped_field = field_str.strip()
            if stripped_field.upper() == "NULL":
                fields.append(None)
            elif stripped_field.startswith("'") and stripped_field.endswith("'"):
                # Handles '' as empty string, and 'it''s' as "it's"
                fields.append(
                    stripped_field[1:-1].replace("''", "'").replace('\\"', '"')
                )
            else:
                # For numbers or other unquoted values
                fields.append(stripped_field)
        return fields

    def _format_sql_value(self, v):
        if v is None:
            return "NULL"
        if isinstance(v, (int, float)):
            return str(v)
        # For strings, use a simplified escape that just wraps in quotes and escapes single quotes
        # This is for generating INSERT/UPDATE values, not DEFAULT clauses.
        safe_val = str(v).replace("'", "''")
        return f"'{safe_val}'"

def _load_config(config_file="optimize_sql_dump.ini"):
    config = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes=("#", ";"))
    config_defaults = {}
    boolean_flags = {"verbose", "dry_run", "diff_from_db", "diff_data", "info"}
    boolean_like_flags = {"split", "load_data_dir", "insert_only"}
    if os.path.exists(config_file) and os.path.getsize(config_file) > 0:
        config.read(config_file)
        _parse_config_sections(
            config, config_defaults, boolean_flags, boolean_like_flags
        )
    return config_defaults


def optimize_dump(**kwargs):
    if kwargs.get("diff_from_db"):
        differ = DatabaseDiffer(**kwargs)
        differ.run()
    elif kwargs.get("info"):
        analyzer = DumpAnalyzer(**kwargs)
        analyzer.run()
    else:
        optimizer = DumpOptimizer(**kwargs)
        optimizer.run()


def _parse_config_sections(config, config_defaults, boolean_flags, boolean_like_flags):
    general_mapping = {
        "db-type": "db_type",
        "table": "table",
        "batch-size": "batch_size",
        "verbose": "verbose",
        "dry-run": "dry_run",
        "split": "split",
        "load-data": "load_data_dir",
        "tsv-buffer-size": "tsv_buffer_size",
        "insert-only": "insert_only",
    }
    diff_mapping = {
        "diff-from-db": "diff_from_db",
        "diff-data": "diff_data",
        "db-host": "db_host",
        "db-user": "db_user",
        "db-password": "db_password",
        "db-name": "db_name",
        "verbose": "verbose",
    }

    def load_section(section_name, mapping):
        if section_name not in config:
            return
        for key, dest in mapping.items():
            if key in config[section_name]:
                if dest in boolean_flags or dest in boolean_like_flags:
                    if config[section_name][key] is None or config.getboolean(
                        section_name, key
                    ):
                        config_defaults[dest] = True
                else:
                    config_defaults[dest] = config.get(section_name, key)

    load_section("optimize", general_mapping)
    load_section("diff", diff_mapping)


def _create_arg_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # --- Input/Output Arguments ---
    p.add_argument("input", nargs="?", help=tl("Dump file (can be .gz/.bz2/.xz/.zip)"))
    p.add_argument(
        "--input", "-i", dest="input_override", help=tl("Override positional input file")
    )
    p.add_argument("output", nargs="?", help=tl("Output file (optimized)"))

    p.add_argument(
        "--db-type",
        choices=["auto", "mysql", "postgres"],
        help=tl("Database dialect: mysql/postgres or auto (default)"),
    )
    p.add_argument(
        "--table",
        "-t",
        type=str,
        help=tl("If specified, optimize only this table (name without schema)"),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        help=tl("Number of tuples in a single merged INSERT (default: 1000)"),
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help=tl("Print diagnostic information")
    )
    p.add_argument(
        "--dry-run", action="store_true", help=tl("Dry run: does not write output")
    )

    # --- Mutually Exclusive Output Modes ---
    output_mode_group = p.add_mutually_exclusive_group()
    output_mode_group.add_argument(
        "--output", "-o", dest="output_override", help=tl("Override positional output file")
    )
    output_mode_group.add_argument(
        "--split",
        nargs="?",
        const=".",
        dest="split_dir",
        help=tl(
            "Split dump into files per table. Optional dir, defaults to current."
        ),
    )
    output_mode_group.add_argument(
        "--load-data",
        nargs="?",
        const=".",
        dest="load_data_dir",
        help=tl(
            "[MySQL] Generate files for LOAD DATA. Optional dir, defaults to current."
        ),
    )
    output_mode_group.add_argument(
        "--insert-only",
        nargs="?",
        const=".",
        help=tl(
            "Generate insert-only files (TRUNCATE + INSERTs). Optional dir, defaults to current."
        ),
    )
    output_mode_group.add_argument(
        "--info",
        action="store_true",
        help=tl("Analyze dump and print summary without writing files."),
    )

    # --- Options for Specific Modes ---
    p.add_argument(
        "--tsv-buffer-size",
        type=int,
        help=tl("[--load-data] Buffer size for TSV rows (default: 200)"),
    )

    # --- Database Diffing Group ---
    diff_group = p.add_argument_group(tl("Database Diffing (Experimental)"))
    diff_group.add_argument(
        "--diff-from-db",
        action="store_true",
        help=tl(
            "Generate a diff by comparing the dump against a live database."
        ),
    )
    diff_group.add_argument(
        "--diff-data",
        action="store_true",
        help=tl(
            "Also compare table data and generate INSERT/UPDATE/DELETE statements (requires --diff-from-db)."
        ),
    )
    diff_group.add_argument("--db-host", help=tl("Database host for diffing."))
    diff_group.add_argument("--db-user", help=tl("Database user for diffing."))
    diff_group.add_argument("--db-password", help=tl("Database password for diffing."))
    diff_group.add_argument("--db-name", help=tl("Database name for diffing."))
    return p


def _validate_args(p, args):
    # Consolidate positional and named input/output arguments
    args.input = args.input_override or args.input
    args.output = args.output_override or args.output

    # Check for required input file
    if not args.input:
        p.error(
            tl("You must provide an input dump file (e.g., `script.py dump.sql`)")
        )
    if not os.path.exists(args.input):
        print(tl("File not found: {path}").format(path=args.input))
        sys.exit(2)

    # Check for at least one output mode if not using --diff-from-db
    is_output_mode_set = any(
        [args.output, args.split_dir, args.load_data_dir, args.insert_only, args.info]
    )
    if not is_output_mode_set and not args.diff_from_db:
        p.error(
            tl(
                "You must specify an output mode (e.g., `script.py in.sql out.sql` or use a flag like --split, --info)."
            )
        )

    # Append .sql to output filename if needed
    if args.output and not args.output.lower().endswith(".sql"):
        if args.verbose:
            print(
                tl(
                    "[INFO] Output filename does not end with .sql, appending it. New name: {name}",
                ).format(name=args.output + ".sql")
            )
        args.output += ".sql"

    # Validate --diff-from-db dependencies
    if args.diff_from_db:
        if not mysql:
            p.error(
                tl(
                    "The 'mysql-connector-python' library is required for --diff-from-db. Please install it."
                )
            )
        if not args.output:
            p.error(tl("--diff-from-db requires --output to be specified."))
        if not args.db_user or not args.db_name:
            p.error(tl("--diff-from-db requires --db-user and --db-name."))
        if args.db_type != "mysql" and args.db_type != "auto":
            p.error(tl("--diff-from-db currently only supports MySQL."))
        if args.diff_data and not args.diff_from_db:
            p.error(tl("--diff-data can only be used with --diff-from-db."))
        args.db_type = "mysql"


def set_parse_arguments_and_config():
    parser = argparse.ArgumentParser(
        description=tl(
            "SQL Dump Optimizer: merges INSERTs, detects compression, supports MySQL/Postgres."
        )
    )
    config_defaults = _load_config()
    parser.set_defaults(**config_defaults)
    parser = _create_arg_parser(parser)
    parser.set_defaults(
        db_type="auto",
        table=None,
        batch_size=1000,
        tsv_buffer_size=200,
        db_host="localhost",
    )
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def main():
    args = set_parse_arguments_and_config()
    kwargs = vars(args)
    kwargs["inpath"] = kwargs.pop("input")
    kwargs["outpath"] = kwargs.pop("output")
    kwargs.pop("input_override", None)
    kwargs.pop("output_override", None)
    kwargs["target_table"] = kwargs.pop("table")
    optimize_dump(**kwargs)


if __name__ == "__main__":
    main()
