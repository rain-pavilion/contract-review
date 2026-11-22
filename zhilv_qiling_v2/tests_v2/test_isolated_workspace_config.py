"""Separate local service versions must not claim one another's tasks."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from zhilv_qiling_v2.api import _queue_path, _workspace_db
from zhilv_qiling_v2.config import Config, config_from_env


class IsolatedWorkspaceConfigTests(unittest.TestCase):
    def test_environment_routes_queue_and_archive_to_same_explicit_store(self):
        with tempfile.TemporaryDirectory() as root:
            database = str(Path(root) / "isolated.sqlite")
            with patch.dict(os.environ, {"ZLQL_STORE_PATH": database}):
                config = config_from_env()
            self.assertEqual(config.store_path, database)
            self.assertEqual(_queue_path(config), Path(database))
            connection = _workspace_db(config)
            try:
                actual = connection.execute("PRAGMA database_list").fetchone()[2]
                self.assertEqual(Path(actual).resolve(), Path(database).resolve())
            finally:
                connection.close()

    def test_unspecified_path_preserves_existing_default(self):
        with patch.dict(os.environ, {"ZLQL_STORE_PATH": ""}):
            config = config_from_env(Config())
        self.assertEqual(config.store_path, "")
        self.assertEqual(_queue_path(config).name, "workspace.sqlite")


if __name__ == "__main__":
    unittest.main()
