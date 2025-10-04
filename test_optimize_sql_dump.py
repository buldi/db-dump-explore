import io
import os
import subprocess
import sys

import pytest
import optimize_sql_dump as opt


@pytest.fixture
def mysql_validator():
    """Provides a MySQLTypeValidator instance."""
    return opt.MySQLTypeValidator()


@pytest.fixture
def mysql_handler():
    """Provides a MySQLHandler instance."""
    return opt.MySQLHandler()


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
class TestSqlTupleParser:
    def test_simple_tuple(self):
        sql = "(1, 'test', NULL);"
        fields = list(opt.SqlTupleParser(sql))
        assert len(fields) == 3
        assert fields == ["1", "'test'", "NULL"]

    def test_multiple_tuples(self):
        # SqlTupleParser is designed to parse a single tuple
        sql = "(1, 'a')"
        fields = list(opt.SqlTupleParser(sql))
        assert len(fields) == 2
        assert fields[0] == "1"
        assert fields[1] == "'a'"

    def test_tuples_with_nested_parens_and_quotes(self):
        sql = "(1, 'it\\'s a string', 'another (nested) string')"
        fields = list(opt.SqlTupleParser(sql))
        assert len(fields) == 3
        assert fields[0] == "1"
        assert fields[1] == "'it\\'s a string'"
        assert fields[2] == "'another (nested) string'"


def test_cli_split_mode(tmp_path):
    dump_content = """
    CREATE TABLE t1 (id INT);
    INSERT INTO t1 VALUES (1), (2);
    CREATE TABLE t2 (name VARCHAR(10));
    INSERT INTO t2 VALUES ('a'), ('b');
    """
    dump_file = tmp_path / "in.sql"
    dump_file.write_text(dump_content)
    
    split_dir = tmp_path / "split_output"

    script_path = os.path.join(os.path.dirname(__file__), "optimize_sql_dump.py")
    cmd = [sys.executable, script_path, str(dump_file), "--split", str(split_dir)]
    subprocess.check_call(cmd)

    assert (split_dir / "t1.sql").exists()
    assert (split_dir / "t2.sql").exists()
    assert "CREATE TABLE t1" in (split_dir / "t1.sql").read_text()
    assert "INSERT INTO t2" in (split_dir / "t2.sql").read_text()

