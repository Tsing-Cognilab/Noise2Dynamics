#!/usr/bin/env python3
"""Verify all release parts and safely restore only inventoried binary assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or not rel.parts or ':' in name:
        raise RuntimeError('Unsafe archive path')
    dst = root.joinpath(*rel.parts)
    if root not in dst.resolve().parents:
        raise RuntimeError('Archive path leaves destination')
    cursor = dst
    while cursor != root:
        if cursor.is_symlink():
            raise RuntimeError('Symlink in destination path')
        cursor = cursor.parent
    return dst


def restore(root, assets):
    root, assets = root.resolve(), assets.resolve()
    manifest = json.loads((root / 'manifests/release-manifest.json').read_text())
    expected = {r['path']: r for r in map(json.loads, (root / 'manifests/binary_assets.jsonl').read_text().splitlines())}
    seen = set()
    for package in manifest['packages']:
        for part in package['parts']:
            path = assets / part['name']
            if path.name != part['name'] or path.stat().st_size != part['bytes'] or sha(path) != part['sha256']:
                raise RuntimeError('Missing or invalid release part: ' + part['name'])
        if len(package['parts']) == 1 and package['parts'][0]['name'] == package['name']:
            archive = assets / package['name']
        else:
            archive = assets / package['name']
            if archive.exists() and sha(archive) != package['sha256']:
                raise RuntimeError('Existing reassembled archive has a different checksum')
            if not archive.exists():
                tmp = archive.with_name(archive.name + '.partial')
                with tmp.open('wb') as out:
                    for part in package['parts']:
                        with (assets / part['name']).open('rb') as f:
                            shutil.copyfileobj(f, out, 4 * 1024 * 1024)
                if sha(tmp) != package['sha256']:
                    raise RuntimeError('Reassembled package checksum mismatch')
                tmp.replace(archive)
        if sha(archive) != package['sha256']:
            raise RuntimeError('Package checksum mismatch')
        proc = subprocess.Popen(['zstd', '-q', '-d', '-c', str(archive)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
                for member in tar:
                    if not member.isfile() or member.name not in expected or member.name in seen:
                        raise RuntimeError('Unexpected/duplicate archive member: ' + member.name)
                    seen.add(member.name)
                    item = expected[member.name]
                    if member.size != item['bytes']:
                        raise RuntimeError('Unexpected member size')
                    dst = safe_path(root, member.name)
                    content = tar.extractfile(member)
                    if dst.exists():
                        if not dst.is_file() or sha(dst) != item['sha256']:
                            content.close()
                            raise RuntimeError('Refusing to overwrite existing file: ' + member.name)
                        content.close()
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(dst.name + '.restore.partial')
                    with tmp.open('wb') as output:
                        shutil.copyfileobj(content, output, 4 * 1024 * 1024)
                    content.close()
                    if sha(tmp) != item['sha256']:
                        raise RuntimeError('Extracted file checksum mismatch')
                    # Atomic no-clobber finalization. A concurrently created file is not overwritten.
                    os.link(tmp, dst)
                    tmp.unlink()
            if proc.wait() != 0:
                raise RuntimeError('Decompression failed')
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            proc.stdout.close()
        print('Verified and restored ' + package['name'], flush=True)
    if seen != set(expected):
        raise RuntimeError('Release packages do not cover the binary inventory')
    print('All binary assets verified. Existing identical files were retained.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-dir', type=Path, required=True)
    args = parser.parse_args()
    restore(Path(__file__).resolve().parents[1], args.assets_dir)
