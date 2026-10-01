import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from entrypoint import command

class BundleTests(unittest.TestCase):
    def test_seal_and_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code = b"print('fixture')\n"
            (root / "serve.py").write_bytes(code)
            manifest = {"entrypoint": "serve.py", "files": {"serve.py": hashlib.sha256(code).hexdigest()}}
            (root / "release.json").write_text(json.dumps(manifest))
            self.assertEqual(command(root)[1], str(root.resolve() / "serve.py"))
            (root / "serve.py").write_bytes(b"changed")
            with self.assertRaises(ValueError): command(root)
            manifest["files"] = {"../serve.py": "0" * 64}
            manifest["entrypoint"] = "../serve.py"
            (root / "release.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): command(root)

if __name__ == "__main__": unittest.main()
