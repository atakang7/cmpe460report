import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("main.ipynb", "ablation.ipynb")


class NotebookTests(unittest.TestCase):
    def test_code_cells_compile(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    with self.subTest(notebook=name, cell=index):
                        compile(source, f"{name}:cell{index}", "exec")

    def test_checkpoint_loading_uses_safe_mode(self):
        for name in NOTEBOOKS:
            notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
            loads = []
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell["source"]))
                loads.extend(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr == "load"
                )

            with self.subTest(notebook=name):
                self.assertTrue(loads, "expected checkpoint loading")
                for call in loads:
                    options = {item.arg: item.value for item in call.keywords}
                    self.assertIn("weights_only", options)
                    self.assertIsInstance(options["weights_only"], ast.Constant)
                    self.assertIs(options["weights_only"].value, True)
                    self.assertIn("map_location", options)


if __name__ == "__main__":
    unittest.main()
