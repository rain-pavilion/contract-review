"""The host CLI must bootstrap without printing or hardcoding credentials."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LibraryCliTests(unittest.TestCase):
    def test_bootstrap_creates_private_credentials_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            credentials = root / "administrator.json"
            command = [sys.executable, "-m", "zhilv_qiling_v2.library_cli", "--root", str(root / "db"),
                "init-admin", "--username", "admin", "--generate-credentials", str(credentials)]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            secret = json.loads(credentials.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(secret["password"]), 20)
            self.assertNotIn(secret["password"], result.stdout + result.stderr)
            before = credentials.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(credentials.read_bytes(), before)
            status = subprocess.run([sys.executable, "-m", "zhilv_qiling_v2.library_cli", "--root",
                str(root / "db"), "status"], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(status.returncode, 0, status.stderr)
            details = json.loads(status.stdout)
            self.assertTrue(details["initialized"])
            self.assertEqual(Path(details['users_db']).resolve(), (root / 'db/users.sqlite').resolve())
            self.assertTrue((root / 'db/users.sqlite').is_file())


if __name__ == "__main__":
    unittest.main()
