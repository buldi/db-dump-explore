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
import io
import os
import sys
import gzip
import bz2
import lzma
import zipfile
import re
import warnings
from abc import ABC, abstractmethod

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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
            raise ValueError("Empty zip file")
        if len(names) > 1:
            warnings.warn(
                f"ZIP archive contains multiple files, using only the first one: {names[0]}"
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
    # Simple validation for example. Can be extended with more types.
    # MySQL 8+ is more strict about date/time format.
        if column_type.startswith("DATETIME") or column_type.startswith("TIMESTAMP"):
            # This validation is simplified. Proper handling would require date parsing.
            # For now, we check if the value is not an empty string, which can be problematic.
            if value.strip("'\"") == "":
                return "NULL"
        return value


class DatabaseHandler(ABC):
    """Base class for handling database-specific operations."""

    def __init__(self):
        self.create_re = re.compile(r"^(CREATE\s+TABLE\b).*", re.IGNORECASE)
        self.insert_re = re.compile(
            r"^(INSERT\s+INTO\s+)(?P<table>[^\s(]+)", re.IGNORECASE
        )
        self.copy_re = re.compile(
            r"^(COPY\s+)(?P<table>[^\s(]+)", re.IGNORECASE
        )
        self.insert_template = "INSERT INTO {table} {cols} VALUES\n{values};\n\n"
        self.validator: TypeValidator | None = None

    @abstractmethod
    def normalize_table_name(self, raw_name: str) -> str:
        """Normalizes a table name, removing quotes and backticks."""
        pass

    @abstractmethod
    def get_load_statement(self, tname: str, tsv_path: str, cols_str: str) -> str:
        """Returns the command to load data from a TSV file."""
        pass

    @abstractmethod
    def extract_columns_from_create(self, create_stmt: str) -> str:
        """Extracts a list of columns from a CREATE TABLE statement as a string '(col1, col2...)'."""
        pass

    def _extract_columns_from_create_base(
        self,
        create_stmt: str,
        body_regex: str,
        ignore_regex: str,
        quote_char: str,
        fallback_regex: str | None = None,
        process_colname_func=None,
    ) -> str:
        m = re.search(body_regex, create_stmt, re.S | re.I)
        if not m and fallback_regex:
            m = re.search(fallback_regex, create_stmt, re.S | re.I)
        if not m:
            return ""
        cols_blob = m.group(1)
        cols = []
        for line in cols_blob.splitlines():
            line = line.strip().rstrip(",")
            if not line or re.match(ignore_regex, line, re.I):
                continue
            colname = line.split()[0]
            if process_colname_func:
                colname = process_colname_func(colname)
            cols.append(colname)
        return f"({', '.join([f'{quote_char}{c}{quote_char}' for c in cols])})" if cols else ""


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

    def get_load_statement(self, tname: str, tsv_path: str, cols_str: str) -> str:
        return (
            f"LOAD DATA LOCAL INFILE '{tsv_path}'\n"
            f"INTO TABLE `{tname}`\n"
            f"FIELDS TERMINATED BY '\\t' ENCLOSED BY '' ESCAPED BY '\\\\'\n"
            f"LINES TERMINATED BY '\\n'\n"
            f"{cols_str};\n"
        )

    def extract_columns_with_types_from_create(self, create_stmt: str) -> list[tuple[str, str]]:
        """Extracts a list of columns (name, type) from a CREATE TABLE statement."""
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
            if not line or re.match(r"PRIMARY\s+KEY|KEY\s+|UNIQUE\s+|CONSTRAINT\s+", line, re.I):
                continue
            parsed = self.validator.parse_column_definition(line)
            if parsed:
                cols_with_types.append(parsed)
        return cols_with_types

    def extract_columns_from_create(self, create_stmt: str) -> str:
        return self._extract_columns_from_create_base(
            create_stmt,
            body_regex=r"\((.*)\)\s*(ENGINE|TYPE|AS|COMMENT|;)",
            fallback_regex=r"CREATE\s+TABLE[^\(]*\((.*)\)\s*;",
            ignore_regex=r"PRIMARY\s+KEY|KEY\s+|UNIQUE\s+|CONSTRAINT\s+",
            quote_char="`",
            process_colname_func=lambda c: c.strip("`\""),
        )


class PostgresHandler(DatabaseHandler):
    """Handler for PostgreSQL database specifics."""

    def __init__(self):
        super().__init__()
    # self.validator = PostgresTypeValidator() # To be implemented in the future

    def normalize_table_name(self, raw_name: str) -> str:
    # Postgres uses " for identifiers, but our normalizer removes them anyway
        name = raw_name.strip().strip('"')
        if "." in name:
            name = name.split(".")[-1]
        return name

    def get_load_statement(self, tname: str, tsv_path: str, cols_str: str) -> str:
    # Postgres does not use `LOCAL` in the same way, the path must be accessible to the server
    # We use E'' for strings with backslashes
        return f"COPY \"{tname}\" {cols_str} FROM '{tsv_path}' WITH (FORMAT 'csv', DELIMITER E'\\t', NULL '\\N');\n"

    def extract_columns_from_create(self, create_stmt: str) -> str:
        return self._extract_columns_from_create_base(
            create_stmt,
            body_regex=r"CREATE\s+TABLE[^\(]*\((.*)\)\s*;",
            ignore_regex=r"PRIMARY\s+KEY|UNIQUE|CONSTRAINT|CHECK",
            quote_char='"',
            process_colname_func=lambda c: c.strip('"'),
        )


class SqlTupleParser:
    """
    A parser for SQL value tuples. It acts as an iterator that returns
    individual values from a tuple, correctly handling quotes, parentheses, and escape characters.
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
        start = -1

    # Find the start of the first tuple
        while self.pos < self.len and self.text[self.pos] != '(':
            self.pos += 1
        if self.pos < self.len:
            self.pos += 1 # Skip the opening parenthesis
            start = self.pos

        while self.pos < self.len:
            char = self.text[self.pos]

            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == "'" and not in_double_quote and not in_backtick:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote and not in_backtick:
                in_double_quote = not in_double_quote
            elif char == '`' and not in_single_quote and not in_double_quote:
                in_backtick = not in_backtick
            elif not (in_single_quote or in_double_quote or in_backtick):
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    if paren_level == 0:
                        # End of tuple
                        if start != -1 and self.pos > start:
                            yield self.text[start:self.pos].strip()
                        start = -1
                        break # Finish after the first tuple
                    else:
                        paren_level -= 1
                elif char == ',' and paren_level == 0:
                    if start != -1:
                        yield self.text[start:self.pos].strip()
                    start = self.pos + 1

            self.pos += 1

        if start != -1 and start < self.len and self.text[start:self.pos].strip():
             # Last value in the tuple
            end_pos = self.text.rfind(')', start, self.pos)
            if end_pos != -1:
                yield self.text[start:end_pos].strip()

def parse_sql_values_to_tsv_row(tuple_str: str) -> str:
    """Converts an SQL tuple (as a string) to a TSV row."""
    parser = SqlTupleParser(tuple_str)
    fields = [field.strip("'\"`") for field in parser]
    return "\t".join(r'\N' if f.upper() == 'NULL' else f.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r') for f in fields)

# ---------- SQL Parser ----------
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
        parts = text_chunk.split("\n.\\\n", 1)
        if len(parts) > 1:
            yield "copy_data", (self.copy_table_name, parts[0])
            self.buf = list(parts[1])
            self.in_copy_data = False
            self.copy_table_name = None
        else:
            self.buf = list(text_chunk)

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
        elif char == '`' and not self.in_single_quote and not self.in_double_quote:
            self.in_backtick = not self.in_backtick
        elif not (self.in_single_quote or self.in_double_quote or self.in_backtick):
            if char == '(':
                self.paren_level += 1
            elif char == ')':
                if self.paren_level > 0:
                    self.paren_level -= 1
            elif char == ';' and self.paren_level == 0:
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
        
        # Reset state for the next statement
        self.in_single_quote, self.in_double_quote, self.in_backtick, self.is_escaped, self.paren_level = False, False, False, False, 0

        t = text.lstrip()[:30].upper()
        if t.startswith("CREATE TABLE") or t.startswith("CREATE TEMPORARY TABLE"):
            return "create", text
        elif t.startswith("INSERT INTO"):
            return "insert", text
        elif t.startswith("COPY ") and " FROM STDIN" in t.upper():
            m = re.search(r"COPY\s+(?P<name>[^\s\(;]+)", text, re.I)
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
    m = handler.insert_re.search(stmt) # type: ignore
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

        self.split_mode = bool(self.args.get('split_dir'))
        self.load_data_mode = bool(self.args.get('load_data_dir'))
        self.output_dir = self.args.get('split_dir') or self.args.get('load_data_dir')

        self.file_map = {}
        self.fout = None
        self.create_map = {}
        self.insert_buffers = {}
        self.total_rows = 0
        self.total_batches = 0

    def _setup_handler(self):
        db_type = self.args.get('db_type', 'auto')
        if db_type == "auto":
            db_type = detect_db_type(self.args['inpath'])
            if self.args.get('verbose'):
                print(f"[INFO] Detected DB type: {db_type}")
        return MySQLHandler() if db_type == "mysql" else PostgresHandler()

    def _setup_progress(self):
        global progress
        filesize = os.path.getsize(self.args['inpath'])
        if self.args.get('verbose') and tqdm:
            progress = tqdm(total=filesize, unit="B", unit_scale=True, desc="Processing")
        else:
            progress = None
        return progress

    def _get_writer(self, tname):
        if tname not in self.file_map:
            if self.load_data_mode:
                sql_fname = os.path.join(self.output_dir, f"{tname}.sql")
                tsv_fname = os.path.join(self.output_dir, f"{tname}.tsv")
                self.file_map[tname] = {
                    'sql': open(sql_fname, "w", encoding="utf-8"),
                    'tsv': open(tsv_fname, "w", encoding="utf-8"),
                    'tsv_path': os.path.abspath(tsv_fname),
                    'tsv_buffer': []
                }
                if self.args.get('verbose'): print(f"[INFO] Created files for table {tname}: {sql_fname}, {tsv_fname}")
            else: # split_mode
                fname = os.path.join(self.output_dir, f"{tname}.sql")
                self.file_map[tname] = open(fname, "w", encoding="utf-8")
                if self.args.get('verbose'): print(f"[INFO] Created file {fname}")
        
        return self.file_map[tname]['sql'] if self.load_data_mode else self.file_map[tname]

    def _flush_insert_buffer(self, tname):
        buf = self.insert_buffers.get(tname)
        if not buf or not buf["tuples"]:
            return
        cols = buf.get("cols_text", "")
        writer = self._get_writer(tname) if self.split_mode else self.fout
        writer.write(f"INSERT INTO {tname} {cols} VALUES\n")
        writer.write(",\n".join(buf["tuples"]))
        writer.write(";\n\n")
        self.total_batches += 1
        buf["tuples"].clear()

    def _flush_tsv_buffer(self, tname, force=False):
        if tname not in self.file_map: return
        
        tsv_info = self.file_map[tname]
        tsv_buffer_size = self.args.get('tsv_buffer_size', 200)
        if tsv_info.get('tsv_buffer') and (force or len(tsv_info['tsv_buffer']) >= tsv_buffer_size):
            tsv_info['tsv'].write("\n".join(tsv_info['tsv_buffer']) + "\n")
            tsv_info['tsv_buffer'].clear()

    def _handle_create(self, stmt):
        m = re.search(r"CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?(?P<name>[^\s\(;]+)", stmt, re.I)
        if not m: return

        tname = self.handler.normalize_table_name(m.group("name").strip())
        self.create_map[tname] = stmt
        target_table = self.args.get('target_table')
        if not target_table or tname == target_table:                        
            if self.split_mode or self.load_data_mode:
                writer = self._get_writer(tname)
                if self.load_data_mode and "IF NOT EXISTS" not in stmt.upper()[:100]:
                    create_stmt = re.sub(r"CREATE\s+TABLE", "CREATE TABLE IF NOT EXISTS", stmt, count=1, flags=re.I)
                    writer.write(create_stmt.strip() + ";\n\n")
                else:
                    writer.write(stmt.strip() + "\n\n")
            else:
                self.fout.write(stmt.strip() + "\n\n")

    # Helper functions for _handle_insert in load_data_mode
    def _parse_single_tuple_to_fields(self, tuple_str: str) -> list[str | None]:
        """Parses a SQL tuple string into a list of field values, handling quotes and NULL."""
        parser = SqlTupleParser(tuple_str)
        fields = []
        for field_str in parser:
            field_str = field_str.strip()
            if field_str.upper() == 'NULL':
                fields.append(None)
            elif (field_str.startswith("'") and field_str.endswith("'")) or \
                 (field_str.startswith('"') and field_str.endswith('"')) or \
                 (field_str.startswith('`') and field_str.endswith('`')):
                fields.append(field_str[1:-1]) # Remove quotes
            else:
                fields.append(field_str)
        return fields

    def _values_to_tsv_row(self, values: list[str | None]) -> str:
        """Converts a list of values into a TSV row string, handling None and escaping."""
        processed_values = []
        for v in values:
            processed_values.append(r'\N' if v is None else str(v).replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r'))
        return "\t".join(processed_values)

    def _handle_insert(self, stmt):
        target_table = self.args.get('target_table')
        tname = extract_table_from_insert(stmt, self.handler)
        if not tname or (target_table and tname != target_table):
            return

        prefix, values_body = extract_values_from_insert(stmt)

        if self.load_data_mode:
            if values_body:
                count = 0
                for tuple_str in SqlTupleParser(values_body):
                    fields = self._parse_single_tuple_to_fields(tuple_str)
                    self.file_map[tname]['tsv_buffer'].append(self._values_to_tsv_row(fields))
                    count += 1
                self._flush_tsv_buffer(tname)
                self.total_rows += count
            return

        writer = self._get_writer(tname) if self.split_mode else self.fout
        if not prefix or not values_body:
            writer.write(stmt)
            return

        cols_match = re.search(r"INSERT\s+INTO\s+[^\(]+(\([^\)]*\))\s*VALUES", prefix, re.I | re.S)
        cols_text = cols_match.group(1).strip() if cols_match else self.handler.extract_columns_from_create(self.create_map.get(tname, ""))
        
        tuples = list(SqlTupleParser(values_body)) if values_body else []
        if not tuples:
            writer.write(stmt)
            return

        buf = self.insert_buffers.setdefault(tname, {"cols_text": cols_text, "tuples": []})
        buf["tuples"].extend(tuples)
        self.total_rows += len(tuples)
        if len(buf["tuples"]) >= self.args.get('batch_size', 1000):
            self._flush_insert_buffer(tname)

    def _handle_copy(self, stype, stmt):
        if not self.load_data_mode: return
        
        target_table = self.args.get('target_table')
        if stype == "copy_start":
            tname = extract_table_from_insert(stmt, self.handler)
            if not tname or (target_table and tname != target_table):
                return
            self._get_writer(tname) # Ensure files for the table exist
        elif stype == "copy_data":
            tname, data = stmt
            if tname in self.file_map:
                self.file_map[tname]['tsv'].write(data)
                self.total_rows += data.count('\n')

    def _handle_other(self, stmt):
        target_table = self.args.get('target_table')
        if not target_table or (target_table in stmt):
            if not self.split_mode and not self.load_data_mode:
                self.fout.write(stmt)

    def run(self):
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.fout = open(self.args['outpath'], "w", encoding="utf-8") if not self.args.get('dry_run') else open(os.devnull, "w")
            if self.fout:
                self.fout.write("-- Optimized by SqlDumpOptimizer\n")
                self.fout.write(f"-- Source: {os.path.basename(self.args['inpath'])}\n")
                self.fout.write("--\n\n")

        with open_maybe_compressed(self.args['inpath'], "rt") as fin:
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
        if self.split_mode or self.load_data_mode:
            if self.load_data_mode:
                for tname in self.file_map.keys():
                    self._flush_tsv_buffer(tname, force=True)
                for tname, writers in self.file_map.items():
                    cols_str = self.handler.extract_columns_from_create(self.create_map.get(tname, ""))
                    load_stmt = self.handler.get_load_statement(tname, writers['tsv_path'], cols_str)
                    writers['sql'].write(load_stmt)
                    writers['sql'].close()
                    writers['tsv'].close()
            else: # split_mode
                for f in self.file_map.values():
                    f.close()
        else:
            if self.fout: self.fout.close()

        if self.args.get('verbose'):
            print(f"[INFO] Wrote {self.total_rows} records in {self.total_batches} batches.")
        
        if not self.args.get('dry_run') and not self.split_mode and not self.load_data_mode:
            print(f"Done. Saved to: {self.args['outpath']}")
        elif self.split_mode:
            print(f"Done. Split dump into files in directory: {self.args['split_dir']}")
        elif self.load_data_mode:
            print(f"Done. Generated files for import in directory: {self.args['load_data_dir']}")
        
        if self.progress:
            self.progress.close()

def optimize_dump(**kwargs):
    optimizer = DumpOptimizer(**kwargs)
    optimizer.run()

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(
        description="SQL Dump Optimizer: merges INSERTs, detects compression, supports MySQL/Postgres."
    )
    p.add_argument(
        "positional",
        nargs="*",
        help="Optional form: <input_file> <output_file>",
    )
    p.add_argument("--input", "-i", help="Dump file (can be .gz/.bz2/.xz/.zip)")
    p.add_argument("--output", "-o", help="Output file (optimized)")
    p.add_argument(
        "--db-type",
        choices=["auto", "mysql", "postgres"],
        default="auto",
        help="Database dialect: mysql/postgres or auto (default)",
    )
    p.add_argument(
        "--table",
        "-t",
        default=None,
        help="If specified, optimize only this table (name without schema)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of tuples in a single merged INSERT (default: 1000)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Print diagnostic information"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: does not write the output file",
    )
    p.add_argument(
        "--split",
        nargs="?",
        const=".",
        help="Split the dump into separate files per table. "
        "If a directory is provided, files will be saved there. "
        "If no value is given, use the current directory.",
    )
    p.add_argument(
        "--load-data",
        nargs="?",
        const=".",
        help="[MySQL ONLY] Generate .sql and .tsv files for LOAD DATA INFILE. "
        "Requires a directory (defaults to current). Mutually exclusive with --output and --split.",
    )
    p.add_argument(
        "--tsv-buffer-size",
        type=int,
        default=200,
        help="[--load-data ONLY] Number of rows buffered before writing to the .tsv file (default: 200)",
    )
    
    args = p.parse_args()

    if args.positional and not args.input:
        args.input = args.positional[0]
        if len(args.positional) > 1 and not args.output and not args.split and not args.load_data:
            args.output = args.positional[1]

    if not args.input:
        p.error("You must provide an input dump (--input or the first positional argument)")

    if not os.path.exists(args.input):
        print("File not found:", args.input)
        sys.exit(2)

    if args.load_data and (args.output or args.split):
        p.error("The --load-data option cannot be used with --output or --split.")

    if not args.output and not args.split and not args.load_data:
        p.error("You must specify an output mode: --output <file>, --split [dir], or --load-data [dir].")
    if args.output and (args.split or args.load_data):
        p.error("The --output option cannot be used with --split or --load-data.")

    optimize_dump(
        inpath=args.input,
        outpath=args.output,
        db_type=args.db_type,
        target_table=args.table,
        batch_size=args.batch_size,
        verbose=args.verbose,
        dry_run=args.dry_run,
        split_dir=args.split, 
        load_data_dir=args.load_data, 
        tsv_buffer_size=args.tsv_buffer_size
    )


if __name__ == "__main__":
    main()
