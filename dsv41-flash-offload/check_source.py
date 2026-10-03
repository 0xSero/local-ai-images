#!/usr/bin/env python3
"""CPU-only image receipt checks. Does not qualify GPU serving or model quality."""
import ast
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

root = Path('/opt/dsv41')
expected_commit = sys.argv[1]
assert (root / 'COMMIT').read_text().strip() == expected_commit, 'serving source commit mismatch'
expected = json.loads((root / 'runtime/expected-receipts.json').read_text())
receipts = root / 'receipts'
vllm = Path(metadata.distribution('vllm').locate_file('vllm'))
records = json.loads((receipts / 'runtime-patch-receipt.json').read_text())['runtime_adaptations']
assert {r['file'] for r in records} == set(expected['vllm']), 'runtime receipt set mismatch'
for r in records:
    assert {k: r[k] for k in ('before', 'after')} == expected['vllm'][r['file']], r['file']
    assert hashlib.sha256((vllm / r['file']).read_bytes()).hexdigest() == r['after'], r['file']

steps = [json.loads((receipts / (name + '.json')).read_text()) for name in
         ('dma-patch-receipt', 'dma-segments-receipt', 'dma-gemm-receipt', 'dma-prefetch-receipt')]
for got, want in zip(steps, expected['vllm_exl3/exl3.py']):
    assert (got['before'], got['after']) == (want['before'], want['after']), 'DMA receipt mismatch'
for before, after in zip(steps, steps[1:]):
    assert before['after'] == after['before'], 'DMA receipt chain mismatch'
code = Path(steps[-1]['file']).read_bytes()
hook = b'\ntry:\n    import sys as _s, ct.ct_vllm as _dsct\n    _dsct.install(_s.modules[__name__])\nexcept Exception as _e:\n    print("dsv41 ct install failed", repr(_e), flush=True)\n'
assert code.endswith(hook), 'CPU tier hook missing'
assert hashlib.sha256(code[:-len(hook)]).hexdigest() == steps[-1]['after'], 'patched EXL3 source mismatch'

for path in [*root.glob('ct/*.py'), *root.glob('dsv41/*.py'), *root.glob('runtime/*.py'), *root.glob('pack/*.py')]:
    ast.parse(path.read_text(), filename=str(path))
assert (root / 'docker/entrypoint.sh').is_file(), 'entrypoint missing'
assert (root / 'pack/prepare_pack.py').is_file() and (root / 'pack/verify.py').is_file(), 'pack tools missing'
ctypes.CDLL(str(root / 'build/engram_disk.so'))
print(json.dumps({'check': 'CPU_SOURCE_RECEIPTS_ONLY_NOT_SERVING_ACCEPTANCE',
                  'commit': expected_commit, 'vllm_files': len(records), 'dma_steps': len(steps),
                  'packages': {p: metadata.version(p) for p in ('vllm', 'exllamav3', 'torch', 'marisa-trie')}}))
