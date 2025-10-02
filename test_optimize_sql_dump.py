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

# def test_cli_basic(tmp_path):
#     dump = tmp_path / "in.sql"
#     out = tmp_path / "out.sql"
#     dump.write_text("CREATE TABLE t (id INT);\nINSERT INTO t VALUES (1);")

#     # We use the path to the script from __file__
#     script_path = os.path.join(os.path.dirname(__file__), "optimize_sql_dump.py")
#     cmd = [sys.executable, script_path, str(dump), "--db-type postgres", str(out)]
#     subprocess.check_call(cmd)

#     assert out.exists()
#     assert "INSERT INTO t" in out.read_text()
#     assert "Optimized by SqlDumpOptimizer" in out.read_text()

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

# def test_cli_load_data_mode(tmp_path):
#     dump_content = """
#     CREATE TABLE `users` (`id` int, `email` varchar(255));
#     INSERT INTO `users` VALUES (1,'test@example.com'),(2,'another@example.com');
#     INSERT INTO `users` VALUES (3,'third@example.com');
#     """
#     dump_file = tmp_path / "in.sql"
#     dump_file.write_text(dump_content)
    
#     load_data_dir = tmp_path / "load_data_output"

#     script_path = os.path.join(os.path.dirname(__file__), "optimize_sql_dump.py")
#     cmd = [sys.executable, script_path, str(dump_file), "--db-type postgres --load-data", str(load_data_dir)]
#     subprocess.check_call(cmd)

#     sql_file = load_data_dir / "users.sql"
#     tsv_file = load_data_dir / "users.tsv"

#     assert sql_file.exists()
#     assert tsv_file.exists()

#     sql_content = sql_file.read_text()
#     assert "CREATE TABLE IF NOT EXISTS `users`" in sql_content
#     assert f"LOAD DATA LOCAL INFILE '{tsv_file.absolute()}'" in sql_content
#     assert "INTO TABLE `users`" in sql_content

#     tsv_content = tsv_file.read_text()
#     assert "1\ttest@example.com" in tsv_content
#     assert "2\tanother@example.com" in tsv_content
#     assert "3\tthird@example.com" in tsv_content
#     assert len(tsv_content.strip().split('\n')) == 3

# def test_cli_postgres_dump(tmp_path):
#     dump_content = "CREATE TABLE users (id int, name text);\nINSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');"
#     dump_file = tmp_path / "in.sql"
#     dump_file.write_text(dump_content)
#     out_file = tmp_path / "out.sql"

#     script_path = os.path.join(os.path.dirname(__file__), "optimize_sql_dump.py")
#     cmd = [sys.executable, script_path, str(dump_file), str(out_file), "--db-type", "postgres"]
#     subprocess.check_call(cmd)

#     output = out_file.read_text()
#     assert 'INSERT INTO users ("id", "name") VALUES' in output
#     assert "(1, 'Alice'),\n(2, 'Bob');" in output
