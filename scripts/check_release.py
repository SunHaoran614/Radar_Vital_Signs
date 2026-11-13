"""Check the public source boundary without reading datasets or model weights."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = {'src', 'tests', 'configs', 'docs', 'scripts', '.github'}
PUBLIC_FILES = {
    '.gitignore', '.dockerignore', 'README.md', 'AGENTS.md', 'Dockerfile',
    'compose.yml', 'compose.local.yml', 'environment.yml', 'pyproject.toml',
    'pytest.ini', 'requirements.txt', 'requirements-test.txt',
}
GENERATED_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.mat', '.bin', '.pkl', '.pickle',
                      '.docx', '.log', '.pyc', '.pdf', '.zip'}
HISTORICAL_NAMES = {'historical_archive', 'experiments', 'experiment_runs', 'audit',
                    'model_outputs', 'exports', 'data', 'Dataset', 'datasets', 'external'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def is_public(name):
    path = Path(name)
    return (name in PUBLIC_FILES or (path.parts[0] in PUBLIC_DIRS
            and not any(part in ('__pycache__', '.pytest_cache') for part in path.parts)))


def main():
    errors = []
    candidates = set(p.decode() for p in git('ls-files', '-z').split(b'\0') if p)
    candidates.update(p.decode() for p in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if p)
    for name in sorted(candidates):
        if not is_public(name):
            errors.append(f'Non-public Git path: {name}')
        elif Path(name).suffix.lower() in GENERATED_SUFFIXES:
            errors.append(f'Generated/binary artifact in source scope: {name}')
        elif not (ROOT / name).is_file():
            errors.append(f'Tracked file missing from public worktree: {name}')
        elif (ROOT / name).stat().st_size >= 10 * 1024 * 1024:
            errors.append(f'Large public file needs review: {name}')
    parsed, links = 0, 0
    for directory in ('src', 'tests', 'scripts'):
        for path in (ROOT / directory).rglob('*.py'):
            try:
                tree = ast.parse(path.read_bytes(), filename=str(path.relative_to(ROOT)))
            except (SyntaxError, UnicodeError) as exc:
                errors.append(str(exc))
                continue
            parsed += 1
            for node in ast.walk(tree):
                imports = [a.name for a in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if node.level:
                        parts = list(path.parent.relative_to(ROOT).parts)
                        module = '.'.join(parts[:len(parts) - node.level + 1] + ([module] if module else []))
                    imports.append(module)
                for module in imports:
                    if module.split('.')[0] in HISTORICAL_NAMES | {'scripts'}:
                        errors.append(f'Public code imports historical/tool module: {path.relative_to(ROOT)} -> {module}')
                    if module == 'src' or module.startswith('src.'):
                        target = ROOT.joinpath(*module.split('.'))
                        if not target.with_suffix('.py').is_file() and not (target / '__init__.py').is_file():
                            errors.append(f'Missing internal import: {path.relative_to(ROOT)} -> {module}')
    docs = [ROOT / 'README.md', ROOT / 'AGENTS.md', *(ROOT / 'docs').rglob('*.md'),
            *(ROOT / 'src').rglob('*.md')]
    for path in docs:
        for target in re.findall(r'\]\(([^)\n]+)\)', path.read_text()):
            target = unquote(target.split('#')[0])
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            links += 1
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(ROOT) or 'historical_archive' in resolved.relative_to(ROOT).parts:
                errors.append(f'Public documentation depends on local archive: {path.relative_to(ROOT)} -> {target}')
            elif not resolved.exists():
                errors.append(f'Broken public link: {path.relative_to(ROOT)} -> {target}')
    probe = 'historical_archive/DO_NOT_UPLOAD.txt'
    ignored = subprocess.run(['git', 'check-ignore', '--no-index', '-q', probe], cwd=ROOT)
    if ignored.returncode:
        errors.append('historical_archive is not ignored')
    if 'historical_archive' not in (ROOT / '.dockerignore').read_text().splitlines():
        errors.append('Docker context does not exclude historical_archive')
    for args in (('diff', '--check'), ('diff', '--cached', '--check')):
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            errors.append(result.stdout + result.stderr)
    print(json.dumps({'passed': not errors, 'public_candidate_files': len(candidates),
                      'python_files_parsed': parsed, 'documentation_links_checked': links,
                      'errors': errors}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
