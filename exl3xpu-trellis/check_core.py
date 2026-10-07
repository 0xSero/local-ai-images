"""Check the installed trellis core on CPU; no hardware serving claim."""
import json
import sys
from pathlib import Path
from trellis_core.format import FormatError
from trellis_core.format.safetensors_header import parse_header
import trellis_core.reference

assert Path('/opt/trellis-serve/COMMIT').read_text().strip() == sys.argv[1]
empty = {'w': {'dtype': 'F16', 'shape': [0], 'data_offsets': [4, 4]}}
assert parse_header(json.dumps(empty).encode(), 'synthetic').tensors['w'].nbytes == 0
bad = {'w': {'dtype': 'F16', 'shape': [2], 'data_offsets': [10, 4]}}
try:
    parse_header(json.dumps(bad).encode(), 'synthetic')
except FormatError:
    pass
else:
    raise AssertionError('reversed byte range accepted')
print('installed trellis core imports and zero-size/reversed-range regression passed')
