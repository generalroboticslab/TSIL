"""Verify the integrated executable sources without rewriting historical hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'projects/TSIL/integration_sources.json'


def verify_current():
    manifest = json.loads(MANIFEST.read_text())
    failures = [name for name, digest in manifest['sha256'].items()
                if not (ROOT/name).is_file()
                or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    if failures:
        raise RuntimeError('Integration source mismatch: '+', '.join(failures))
    return len(manifest['sha256'])


if __name__ == '__main__':
    print('Verified', verify_current(), 'integration files')
