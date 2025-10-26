import bz2
import gzip
import lzma
import sys
import zipfile
from unittest.mock import MagicMock, patch

import pytest
import optimize_sql_dump as opt

# Helper to get the main function for CLI tests
from optimize_sql_dump import main as cli_main

from optimize_sql_dump import (
    escape_sql_value,
)  # Assuming the function is in this module


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
        assert (
            postgres_handler.normalize_table_name('"public"."my_table"') == "my_table"
        )
        assert postgres_handler.normalize_table_name("my_table") == "my_table"
        assert postgres_handler.normalize_table_name('"my_table"') == "my_table"

    def test_get_truncate_statement_postgres(self, postgres_handler):
        stmt = postgres_handler.get_truncate_statement("my_table")
        assert stmt == 'TRUNCATE TABLE "my_table" RESTART IDENTITY CASCADE;\n'

    def test_get_load_statement_postgres(self, postgres_handler):
        cols_str = '("id", "name")'
        tsv_path = "/path/to/my_table.tsv"
        stmt = postgres_handler.get_load_statement("my_table", tsv_path, cols_str)
        expected_stmt = "COPY \"my_table\" (\"id\", \"name\") FROM '/path/to/my_table.tsv' WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\n');\n"
        assert (
            stmt == expected_stmt
        )

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
        cols = [  # noqa: E501
            c.strip().strip('"')
            for c in columns_str.strip().lstrip("(").rstrip(")").split(",")
        ]
        assert set(cols) == {"id", "name", "created_at"}

    def test_detect_db_type_postgres_specific(self, tmp_path):
        dump_file = tmp_path / "pg_dump.sql"
        dump_file.write_text("COPY public.users (id, name) FROM STDIN;\n1\tAlice\n\\.\n")
        db_type = opt.detect_db_type(str(dump_file))
        assert db_type == "postgres"


class TestHelperFunctions:
    @pytest.mark.parametrize(
        "stmt, expected_prefix, expected_values",
        [
            (
                "INSERT INTO `t` VALUES (1, 'a'), (2, 'b');",
                "INSERT INTO `t` VALUES",
                "(1, 'a'), (2, 'b')",
            ),
            (
                "insert into t (c1, c2) values (1, 'a');",
                "insert into t (c1, c2) values",
                "(1, 'a')",
            ),
            ("CREATE TABLE t (id INT);", None, None),
        ],
    )
    def test_extract_values_from_insert(self, stmt, expected_prefix, expected_values):
        prefix, values = opt.extract_values_from_insert(stmt)
        assert prefix == expected_prefix
        assert values == expected_values


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
        assert fields == ["1", "'it\\'s a string'", "'another (nested) string'"]


@pytest.mark.parametrize(
    "compressor, extension",
    [
        (gzip.open, ".gz"),
        (bz2.open, ".bz2"),
        (lzma.open, ".xz"),
        (None, ".zip"),  # Special case for zip
        (None, ".sql"),  # No compression
    ],
)
def test_open_maybe_compressed(tmp_path, compressor, extension):
    """Tests detection and reading of various compression formats."""
    content = "CREATE TABLE t1 (id INT);"
    p = tmp_path / f"dump{extension}"

    if extension == ".zip":
        with zipfile.ZipFile(p, "w") as zf:
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
    with patch.object(sys, "argv", test_args):
        with patch(
            "optimize_sql_dump._load_config", return_value={}
        ):  # Mock _load_config
            opt.main()

    captured = capsys.readouterr()
    assert "--- Dump Analysis Summary ---" in captured.out
    assert "t1" in captured.out
    assert "3" in captured.out  # 3 rows in t1
    assert "t2" in captured.out
    assert "1" in captured.out  # 1 row in t2
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
    test_args = [
        "optimize_sql_dump.py",
        "-i",
        str(dump_file),
        "--split",
        str(split_dir),
    ]
    with patch.object(sys, "argv", test_args):
        with patch(
            "optimize_sql_dump._load_config", return_value={}
        ):  # Mock _load_config
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
    test_args = [
        "optimize_sql_dump.py",
        "-i",
        str(dump_file),
        "--load-data",
        str(load_data_dir),
    ]
    with patch.object(sys, "argv", test_args):
        with patch(
            "optimize_sql_dump._load_config", return_value={}
        ):  # Mock _load_config
            cli_main()

    sql_file = load_data_dir / "users.sql"
    tsv_file = load_data_dir / "users.tsv"

    assert sql_file.exists()
    assert tsv_file.exists()

    sql_content = sql_file.read_text()
    assert "CREATE TABLE IF NOT EXISTS `users`" in sql_content
    assert "LOAD DATA LOCAL INFILE" in sql_content
    assert str(tsv_file) in sql_content

    tsv_content = tsv_file.read_text()  # noqa: E501
    expected_tsv = "1\ttest@test.com\tsome notes\n2\t\\n\tother notes with a\ttab\n"  # noqa: E501
    assert tsv_content == expected_tsv


@pytest.mark.parametrize(
    "invalid_args",
    [  # noqa: E501
        # Mutually exclusive modes
        ["--output", "out.sql", "--split", "dir"],
        ["--output", "out.sql", "--load-data", "dir"],
        ["--split", "dir", "--load-data", "dir"],
        ["--info", "--output", "out.sql"],
        # Missing required args
        ["--diff-from-db"],  # requires --output
        ["--diff-from-db", "--output", "out.sql"],  # requires db-user/db-name
    ],
)
def test_cli_invalid_arguments(tmp_path, invalid_args):
    """Tests that the CLI exits with an error for invalid argument combinations."""
    dump_file = tmp_path / "in.sql"
    dump_file.touch()
    base_args = ["optimize_sql_dump.py", "-i", str(dump_file)]

    with pytest.raises(SystemExit) as e:
        with patch.object(sys, "argv", base_args + invalid_args):
            with patch(
                "optimize_sql_dump._load_config", return_value={}
            ):  # Mock _load_config
                cli_main()
    assert e.type is SystemExit
    assert e.value.code != 0  # Ensure it's an error exit code


@pytest.mark.parametrize(
    "input_val, expected", [
        # Tests for strings
        ("tekst", "DEFAULT 'tekst'"),
        ("O'Reilly", "DEFAULT 'O''Reilly'"),
        ("C:\\Path\\File", "DEFAULT 'C:\\\\Path\\\\File'"),
        ("O'Reilly\\Test", "DEFAULT 'O''Reilly\\\\Test'"),
        # Test for empty string
        ("", "DEFAULT ''"),
        # Tests for numeric values
        (0, "DEFAULT 0"),
        (123, "DEFAULT 123"),
        (3.14, "DEFAULT 3.14"),
        # Tests for boolean values
        (True, "DEFAULT True"),
        (False, "DEFAULT False"),
        # Test for None
        (None, "DEFAULT NULL"),
    ])
def test_escape_sql_value(input_val, expected):
    assert escape_sql_value(input_val, prefix_str="DEFAULT").strip() == expected


class TestDatabaseDiffer:
    @pytest.fixture
    def differ(self, tmp_path):
        """Provides a DatabaseDiffer instance for testing."""
        # The differ's __init__ expects 'inpath' for progress setup.
        # We provide a dummy path to satisfy this requirement.
        dummy_inpath = tmp_path / "dummy.sql"
        dummy_inpath.touch()
        return opt.DatabaseDiffer(inpath=str(dummy_inpath), verbose=False)

    @pytest.mark.parametrize(
        "input_val, expected_sql", [
            (None, "NULL"),
            (123, "123"),
            (-45, "-45"),
            (3.14, "3.14"),
            (0, "0"),
            ("hello world", "'hello world'"),
            ("it's a string", "'it''s a string'"),
            ("", "''"),
            ('string with "quotes"', "'string with \"quotes\"'"),
            ("O'Reilly", "'O''Reilly'"),
            ("multiple 'quotes' here", "'multiple ''quotes'' here'"),
        ])
    def test_format_sql_value(self, differ, input_val, expected_sql):
        assert differ._format_sql_value(input_val) == expected_sql

    # def test_compare_schemas_add_column(self, differ):
    #     dump_cols = {
    #         "id": "`id` int NOT NULL",
    #         "name": "`name` varchar(255) NULL",
    #         "email": "`email` varchar(255) NOT NULL",
    #     }
    #     db_cols = {
    #         "id": {"COLUMN_NAME": "id", "COLUMN_TYPE": "int"},
    #         "name": {"COLUMN_NAME": "name", "COLUMN_TYPE": "varchar(255)"},
    #     }
    #     statements = differ.compare_schemas(dump_cols, db_cols, "users")
    #     assert len(statements) == 1
    #     assert "ADD COLUMN `email` varchar(255) NOT NULL" in statements[0]
    #     assert "AFTER `name`" in statements[0]

    # def test_compare_schemas_modify_column(self, differ):
    #     dump_cols = {
    #         "id": "`id` int NOT NULL",
    #         "name": "`name` text NULL",
    #     }
    #     db_cols = {
    #         "id": {"COLUMN_NAME": "id", "COLUMN_TYPE": "int"},
    #         "name": {
    #             "COLUMN_NAME": "name",
    #             "COLUMN_TYPE": "varchar(255)",
    #             "IS_NULLABLE": "YES",
    #             "COLUMN_DEFAULT": None,
    #             "EXTRA": "",
    #             "CHARACTER_SET_NAME": "utf8mb4",
    #             "COLLATION_NAME": "utf8mb4_unicode_ci",
    #         },
    #     }
    #     statements = differ.compare_schemas(dump_cols, db_cols, "users")
    #     assert len(statements) == 1
    #     assert "MODIFY COLUMN `name` text NULL" in statements[0]

    def test_compare_schemas_no_change(self, differ):
        dump_cols = {"id": "`id` int NOT NULL"}
        db_cols = {
            "id": {
                "COLUMN_NAME": "id",
                "COLUMN_TYPE": "int",
                "IS_NULLABLE": "NO",
                "COLUMN_DEFAULT": None,
                "EXTRA": "auto_increment",
                "CHARACTER_SET_NAME": None,
                "COLLATION_NAME": None,
            }
        }
        statements = differ.compare_schemas(dump_cols, db_cols, "users")
        assert len(statements) == 0

    # def test_compare_data_row_update(self, differ):
    #     dump_row = {"id": 1, "name": "new_name", "email": "test@test.com"}
    #     db_row = {"id": 1, "name": "old_name", "email": "test@test.com"}
    #     pk_cols = ["id"]
    #     update_stmt = differ.compare_data_row(dump_row, db_row, "users", pk_cols)
    #     assert update_stmt is not None
    #     assert "UPDATE `users` SET `name` = 'new_name' WHERE `id` = 1;" in update_stmt

    def test_compare_data_row_no_change(self, differ):
        dump_row = {"id": 1, "name": "same_name"}
        db_row = {"id": 1, "name": "same_name"}
        pk_cols = ["id"]
        update_stmt = differ.compare_data_row(dump_row, db_row, "users", pk_cols)
        assert update_stmt is None

    # def test_run_generates_delete(self, tmp_path):
    #     """
    #     Integration-like test for the DELETE generation logic in `run`.
    #     """
    #     in_file = tmp_path / "dump.sql"
    #     out_file = tmp_path / "diff.sql"
    #     in_file.write_text(
    #         "CREATE TABLE `users` (`id` int, PRIMARY KEY (`id`));\n"
    #         "INSERT INTO `users` VALUES (1);"
    #     )

    #     args = {
    #         "inpath": str(in_file),
    #         "outpath": str(out_file),
    #         "db_name": "testdb",
    #         "diff_data": True,
    #         "insert_only": False,
    #         "verbose": False,
    #     }

    #     # We need to patch mysql.connector if it's not installed
    #     if "mysql" not in sys.modules:
    #         sys.modules["mysql"] = MagicMock()
    #         sys.modules["mysql.connector"] = MagicMock()

    #     differ = opt.DatabaseDiffer(**args)

    #     # Mock database interactions
    #     differ.connect_db = MagicMock()
    #     differ.get_db_schema = MagicMock(return_value={"id": {}})  # Table exists
    #     # DB has PKs (1,) and (2,). Dump only has (1,). So (2,) should be deleted.
    #     differ.get_db_primary_keys = MagicMock(return_value={('1',), ('2',)})
    #     differ.get_db_row_by_pk = MagicMock(return_value={"id": 1})

    #     differ.run()

    #     output = out_file.read_text()

    #     assert "-- Deleting rows" in output
    #     assert "DELETE FROM `users` WHERE `id` = '2';" in output
    #     assert "DELETE FROM `users` WHERE `id` = '1';" not in output


class TestDumpWriter:
    @pytest.fixture
    def mock_handler(self):
        handler = MagicMock(spec=opt.DatabaseHandler)
        handler.normalize_table_name.side_effect = lambda x: x.strip('`')
        handler.insert_template = "INSERT INTO {table} {cols} VALUES\n{values};\n"
        handler.get_truncate_statement.side_effect = lambda t: f"TRUNCATE TABLE `{t}`;\n"
        handler.extract_columns_from_create.return_value = "(`id`, `name`)"
        handler.get_load_statement.return_value = "LOAD DATA MOCK"
        return handler

    # def test_setup_normal_mode(self, mock_handler, tmp_path):
    #     out_file = tmp_path / "out.sql"
    #     args = {"outpath": str(out_file), "inpath": "dummy.sql", "dry_run": False}
    #     writer = opt.DumpWriter(mock_handler, **args)
    #     writer.setup()
    #     writer.finalize({})

    #     assert out_file.exists()
    #     content = out_file.read_text()
    #     assert "-- Optimized by SqlDumpOptimizer" in content

    # def test_setup_dry_run(self, mock_handler):
    #     args = {"outpath": "out.sql", "inpath": "dummy.sql", "dry_run": True}
    #     writer = opt.DumpWriter(mock_handler, **args)
    #     writer.setup()
    #     assert writer.fout.name == os.devnull

    # def test_split_mode_file_creation(self, mock_handler, tmp_path):
    #     split_dir = tmp_path / "split"
    #     args = {"split_dir": str(split_dir), "inpath": "dummy.sql"}
    #     writer = opt.DumpWriter(mock_handler, **args)
    #     writer.setup()

    #     writer.get_writer_for_table("t1")
    #     writer.finalize({})

    #     assert (split_dir / "t1.sql").exists()

    # def test_load_data_mode_file_creation(self, mock_handler, tmp_path):
    #     load_dir = tmp_path / "load_data"
    #     args = {"load_data_dir": str(load_dir), "inpath": "dummy.sql"}
    #     writer = opt.DumpWriter(mock_handler, **args)
    #     writer.setup()

    #     writer.get_writer_for_table("t1")
    #     writer.finalize({"t1": "CREATE TABLE..."})

    #     assert (load_dir / "t1.sql").exists()
    #     assert (load_dir / "t1.tsv").exists()

    #     sql_content = (load_dir / "t1.sql").read_text()
    #     assert "LOAD DATA MOCK" in sql_content

    # def test_insert_only_mode(self, mock_handler, tmp_path):
    #     insert_dir = tmp_path / "insert_only"
    #     args = {"insert_only": str(insert_dir), "inpath": "dummy.sql"}
    #     writer = opt.DumpWriter(mock_handler, **args)
    #     writer.setup()

    #     writer.write_create_statement("t1", "CREATE TABLE `t1` (id int);")
    #     writer.finalize({})

    #     t1_file = insert_dir / "t1.sql"
    #     assert t1_file.exists()
    #     content = t1_file.read_text()
    #     assert "TRUNCATE TABLE `t1`;" in content
    #     # CREATE statement should not be written in insert_only mode
    #     assert "CREATE TABLE" not in content

    # def test_insert_buffering_and_flushing(self, mock_handler, tmp_path):
    #     out_file = tmp_path / "out.sql"
    #     args = {"outpath": str(out_file), "inpath": "dummy.sql", "batch_size": 2}
    #     writer = opt.DumpWriter(mock_handler, **args)
    #     writer.setup()

    #     writer.add_insert_tuples("t1", "(`id`)", ["(1)", "(2)"])
    #     # Buffer should be flushed here as batch_size is reached
    #     assert "t1" not in writer.insert_buffers or not writer.insert_buffers["t1"]["tuples"]

    #     writer.add_insert_tuples("t1", "(`id`)", ["(3)"])
    #     # Buffer should not be flushed yet
    #     assert len(writer.insert_buffers["t1"]["tuples"]) == 1

    #     writer.finalize({})

    #     content = out_file.read_text()
    #     assert "INSERT INTO t1 (`id`) VALUES\n(1),\n(2);\n" in content
    #     assert "INSERT INTO t1 (`id`) VALUES\n(3);\n" in content

    # def test_tsv_buffering_and_flushing(self, mock_handler, tmp_path):
    #     load_dir = tmp_path / "load_data"
    #     args = {"load_data_dir": str(load_dir), "inpath": "dummy.sql", "tsv_buffer_size": 2}
    #     writer = opt.DumpWriter(mock_handler, **args)
    #     writer.setup()
    #     writer.get_writer_for_table("t1")

    #     writer.add_tsv_rows("t1", ["1\ta", "2\tb"]) # Should flush
    #     writer.add_tsv_rows("t1", ["3\tc"]) # Should not flush
    #     writer.finalize({"t1": "CREATE..."})

    #     tsv_content = (load_dir / "t1.tsv").read_text()
    #     assert tsv_content == "1\ta\n2\tb\n3\tc\n"
