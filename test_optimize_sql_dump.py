import bz2
import gzip
import io
import lzma
import os
import subprocess
import sys
import zipfile
from unittest.mock import patch

import pytest
import optimize_sql_dump as opt

# Helper to get the main function for CLI tests
from optimize_sql_dump import main as cli_main


@pytest.fixture
def mysql_validator():
    """Provides a MySQLTypeValidator instance."""
    return opt.MySQLTypeValidator()


@pytest.fixture
def mysql_handler():
    """Provides a MySQLHandler instance."""
    return opt.MySQLHandler()


@pytest.fixture
def postgres_validator():
    """Provides a PostgresTypeValidator instance."""
    return opt.PostgresTypeValidator()


@pytest.fixture
def postgres_handler():
    """Provides a PostgresHandler instance."""
    return opt.PostgresHandler()


class TestMySQLTypeValidator:
    def test_parse_column_definition(self, mysql_validator):
        line = "`my_col` VARCHAR(255) NOT NULL,"
        name, col_type = mysql_validator.parse_column_definition(line)
        assert name == "my_col"
        assert col_type == "VARCHAR(255)"

    def test_validate_datetime_empty_string(self, mysql_validator):
        validated = mysql_validator.validate("''", "DATETIME")
        assert validated == "NULL"
        validated_ts = mysql_validator.validate('""', "TIMESTAMP")
        assert validated_ts == "NULL"

    def test_validate_datetime_valid_string(self, mysql_validator):
        validated = mysql_validator.validate("'2023-01-01 12:00:00'", "DATETIME")
        assert validated == "'2023-01-01 12:00:00'"


class TestMySQLHandler:
    def test_extract_columns_with_types(self, mysql_handler):
        create_stmt = """
        CREATE TABLE `my_table` (
          `id` int unsigned NOT NULL AUTO_INCREMENT,
          `name` varchar(100) DEFAULT NULL,
          `created_at` datetime NOT NULL,
          PRIMARY KEY (`id`)
        ) ENGINE=InnoDB;
        """
        columns = mysql_handler.extract_columns_with_types_from_create(create_stmt)
        assert columns == [
            ("id", "INT"),
            ("name", "VARCHAR(100)"),
            ("created_at", "DATETIME"),
        ]


class TestPostgresTypeValidator:
    def test_parse_column_definition(self, postgres_validator):
        line = '"my_col" VARCHAR(255) NOT NULL,'
        name, col_type = postgres_validator.parse_column_definition(line)
        assert name == "my_col"
        assert col_type == "VARCHAR(255)"

    def test_validate_postgres(self, postgres_validator):
        # Postgres validator is a stub for now, should return value as is
        validated = postgres_validator.validate("'2023-01-01 12:00:00'", "TIMESTAMP")
        assert validated == "'2023-01-01 12:00:00'"
        validated_null = postgres_validator.validate("NULL", "INTEGER")
        assert validated_null == "NULL"


class TestPostgresHandler:
    def test_normalize_table_name_postgres(self, postgres_handler):
        assert postgres_handler.normalize_table_name('"public"."my_table"') == "my_table"
        assert postgres_handler.normalize_table_name('my_table') == "my_table"
        assert postgres_handler.normalize_table_name('"my_table"') == "my_table"

    def test_get_truncate_statement_postgres(self, postgres_handler):
        stmt = postgres_handler.get_truncate_statement("my_table")
        assert stmt == 'TRUNCATE TABLE "my_table" RESTART IDENTITY CASCADE;\n'

    def test_get_load_statement_postgres(self, postgres_handler):
        cols_str = '("id", "name")'
        tsv_path = "/path/to/my_table.tsv"
        stmt = postgres_handler.get_load_statement("my_table", tsv_path, cols_str)
        expected_stmt = (
            "COPY \"my_table\" (\"id\", \"name\") FROM '/path/to/my_table.tsv' WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\n');\n"
        )
        assert stmt == expected_stmt

    def test_extract_columns_from_create_postgres(self, postgres_handler):
        create_stmt = """
        CREATE TABLE "public"."my_table" (
            "id" integer NOT NULL,
            "name" character varying(100) COLLATE pg_catalog."default",
            "created_at" timestamp without time zone DEFAULT now(),
            CONSTRAINT my_table_pkey PRIMARY KEY ("id")
        );
        """
        columns_str = postgres_handler.extract_columns_from_create(create_stmt)
        # Normalize returned string to a list of column names and compare as a set for robustness
        cols = [c.strip().strip('"') for c in columns_str.strip().lstrip('(').rstrip(')').split(',')]
        assert set(cols) == {"id", "name", "created_at"}

    def test_detect_db_type_postgres_specific(self, tmp_path):
        dump_file = tmp_path / "pg_dump.sql"
        dump_file.write_text("COPY public.users (id, name) FROM STDIN;\n1\tAlice\n\\.\n")
        db_type = opt.detect_db_type(str(dump_file))
        assert db_type == "postgres"


class TestSqlTupleParsers:
    def test_simple_tuple(self):
        """Tests parsing a single tuple into its fields."""
        sql = "(1, 'test', NULL);"
        fields = list(opt.SqlTupleFieldParser(sql))
        assert len(fields) == 3
        assert fields == ["1", "'test'", "NULL"]

    def test_multiple_tuples(self):
        """Tests that the multi-tuple parser correctly iterates over multiple tuples."""
        sql = "(1, 'a'), (2, 'b')"
        fields = list(opt.SqlMultiTupleParser(sql))
        assert fields == ["(1, 'a')", "(2, 'b')"]

    def test_tuples_with_nested_parens_and_quotes(self):
        """Tests parsing a single complex tuple into its fields."""
        sql = "(1, 'it\\'s a string', 'another (nested) string')"
        fields = list(opt.SqlTupleFieldParser(sql))
        assert len(fields) == 3
        assert fields == [
            "1",
            "'it\\'s a string'",
            "'another (nested) string'"
        ]


@pytest.mark.parametrize("compressor, extension", [
    (gzip.open, ".gz"),
    (bz2.open, ".bz2"),
    (lzma.open, ".xz"),
    (None, ".zip"), # Special case for zip
    (None, ".sql"), # No compression
])
def test_open_maybe_compressed(tmp_path, compressor, extension):
    """Tests detection and reading of various compression formats."""
    content = "CREATE TABLE t1 (id INT);"
    p = tmp_path / f"dump{extension}"

    if extension == ".zip":
        with zipfile.ZipFile(p, 'w') as zf:
            zf.writestr("dump.sql", content)
    elif extension != ".sql":
        with compressor(p, "wt") as f:
            f.write(content)
    else:
        p.write_text(content)

    with opt.open_maybe_compressed(str(p)) as f:
        read_content = f.read()

    assert read_content == content


def test_dump_analyzer(tmp_path, capsys):
    """Tests the --info mode for dump analysis."""
    dump_content = """
    CREATE TABLE `t1` (`id` int);
    INSERT INTO `t1` VALUES (1),(2),(3);
    CREATE TABLE `t2` (`name` varchar(10));
    INSERT INTO `t2` VALUES ('a');
    """
    dump_file = tmp_path / "in.sql"
    dump_file.write_text(dump_content)

    # Simulate running from command line with --info
    test_args = ["optimize_sql_dump.py", "--input", str(dump_file), "--info"]
    with patch.object(sys, 'argv', test_args):
        with patch('optimize_sql_dump._load_config', return_value={}): # Mock _load_config
            opt.main()

    captured = capsys.readouterr()
    assert "--- Dump Analysis Summary ---" in captured.out
    assert "t1" in captured.out
    assert "3" in captured.out # 3 rows in t1
    assert "t2" in captured.out
    assert "1" in captured.out # 1 row in t2
    assert "Found 2 tables with a total of 4 rows." in captured.out


def test_parse_sql_values_to_tsv_row():
    """Tests the conversion of an SQL tuple string to a TSV row."""
    sql_tuple = "(1, 'some text', NULL, 'it\\'s escaped', '\\n')"
    expected_tsv = "1\tsome text\t\\n\tit's escaped\t\n"
    assert opt.parse_sql_values_to_tsv_row(sql_tuple) == expected_tsv


def test_cli_split_mode(tmp_path):
    dump_content = """CREATE TABLE `t1` (`id` INT);
INSERT INTO `t1` VALUES (1),(2);
CREATE TABLE `t2` (`name` VARCHAR(10));
INSERT INTO `t2` VALUES ('a'),('b');"""

    dump_file = tmp_path / "in.sql"
    dump_file.write_text(dump_content)
    split_dir = tmp_path / "split_output"

    # Simulate running from command line
    test_args = ["optimize_sql_dump.py", "-i", str(dump_file), "--split", str(split_dir)]
    with patch.object(sys, 'argv', test_args):
        with patch('optimize_sql_dump._load_config', return_value={}): # Mock _load_config
            cli_main()

    assert (split_dir / "t1.sql").exists()
    assert (split_dir / "t2.sql").exists()

    t1_content = (split_dir / "t1.sql").read_text()
    t2_content = (split_dir / "t2.sql").read_text()

    # t1.sql
    assert "CREATE TABLE" in t1_content
    assert "INSERT INTO" in t1_content
    assert "1" in t1_content
    assert "2" in t1_content

    # t2.sql
    assert "CREATE TABLE" in t2_content
    assert "INSERT INTO" in t2_content
    assert "'a'" in t2_content
    assert "'b'" in t2_content


def test_cli_load_data_mode(tmp_path):
    """Tests the --load-data mode for generating .sql and .tsv files."""
    dump_content = """
    CREATE TABLE `users` (`id` int, `email` varchar(100), `notes` text);
    INSERT INTO `users` VALUES (1,'test@test.com','some notes'),(2,NULL,'other notes with a\ttab');
    """
    dump_file = tmp_path / "in.sql"
    dump_file.write_text(dump_content)
    load_data_dir = tmp_path / "load_data_output"

    # Simulate running from command line
    test_args = ["optimize_sql_dump.py", "-i", str(dump_file), "--load-data", str(load_data_dir)]
    with patch.object(sys, 'argv', test_args):
        with patch('optimize_sql_dump._load_config', return_value={}): # Mock _load_config
            cli_main()

    sql_file = load_data_dir / "users.sql"
    tsv_file = load_data_dir / "users.tsv"

    assert sql_file.exists()
    assert tsv_file.exists()

    sql_content = sql_file.read_text()
    assert "CREATE TABLE IF NOT EXISTS `users`" in sql_content
    assert "LOAD DATA LOCAL INFILE" in sql_content
    assert str(tsv_file) in sql_content

    tsv_content = tsv_file.read_text()
    expected_tsv = "1\ttest@test.com\tsome notes\n2\t\\N\tother notes with a\\ttab\n"
    assert tsv_content == expected_tsv


def test_cli_load_data_mode_postgres(tmp_path):
    """Tests the --load-data mode for generating .sql and .tsv files for PostgreSQL."""
    dump_content = """
    CREATE TABLE "public"."users" (
        "id" integer NOT NULL,
        "email" character varying(100) COLLATE pg_catalog."default",
        "notes" text COLLATE pg_catalog."default"
    );
    COPY public.users (id, email, notes) FROM STDIN;
    1	test@test.com	some notes
    2		other notes with a	tab
    \\.
    """
    dump_file = tmp_path / "in.sql"
    dump_file.write_text(dump_content)
    load_data_dir = tmp_path / "pg_load_data_output"

    # Simulate running from command line
    test_args = ["optimize_sql_dump.py", "-i", str(dump_file), "--load-data", str(load_data_dir), "--db-type", "postgres"]
    with patch.object(sys, 'argv', test_args):
        with patch('optimize_sql_dump._load_config', return_value={}): # Mock _load_config
            cli_main()

    sql_file = load_data_dir / "users.sql"
    tsv_file = load_data_dir / "users.tsv"

    assert sql_file.exists()
    assert tsv_file.exists()

    sql_content = sql_file.read_text()
    assert "COPY \"users\" (id, email, notes) FROM" in sql_content
    assert str(tsv_file) in sql_content

    tsv_content = tsv_file.read_text()
    expected_tsv = "1\ttest@test.com\tsome notes\n2\t\tother notes with a\ttab\n"
    assert tsv_content == expected_tsv

@pytest.mark.parametrize("invalid_args", [
    # Mutually exclusive modes
    ["--output", "out.sql", "--split", "dir"],
    ["--output", "out.sql", "--load-data", "dir"],
    ["--split", "dir", "--load-data", "dir"],
    ["--info", "--output", "out.sql"],
    # Missing required args
    ["--diff-from-db"], # requires --output
    ["--diff-from-db", "--output", "out.sql"], # requires db-user/db-name
])
def test_cli_invalid_arguments(tmp_path, invalid_args):
    """Tests that the CLI exits with an error for invalid argument combinations."""
    dump_file = tmp_path / "in.sql"
    dump_file.touch()
    base_args = ["optimize_sql_dump.py", "-i", str(dump_file)]

    with pytest.raises(SystemExit) as e:
        with patch.object(sys, 'argv', base_args + invalid_args):
            with patch('optimize_sql_dump._load_config', return_value={}): # Mock _load_config
                cli_main()
    assert e.type == SystemExit
    assert e.value.code != 0 # Ensure it's an error exit code
