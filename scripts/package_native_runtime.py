"""Build the allowlisted runtime used by Render's native Python service."""
from pathlib import Path
import shutil
import ast

root = Path(__file__).resolve().parents[1]
target = root / '.native-runtime'
target.mkdir(exist_ok=True)
for name in ('adfm_engine', 'adfm_api'):
    shutil.copytree(root / name, target / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for path in target.rglob('*.py'):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or '']
        else:
            continue
        if any(name.split('.')[0] in {'streamlit', 'adfm_core', 'pages', 'Home'} for name in names):
            raise RuntimeError(f'Legacy runtime import in {path}')
print('Native runtime packaged and legacy-import check passed.')
