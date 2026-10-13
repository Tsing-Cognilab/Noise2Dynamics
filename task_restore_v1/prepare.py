"""Fetch official source, published checkpoints and original environment metadata."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = 'https://diffusion-policy.cs.columbia.edu/data/experiments/image'

def fetch(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    partial = dest.with_suffix(dest.suffix + '.partial')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(req, timeout=90) as src:
                resumed = src.status == 206 and offset > 0
                expected = int(src.headers.get('Content-Length', 0))
                received = 0
                with partial.open('ab' if resumed else 'wb') as out:
                    while True:
                        block = src.read(4 * 1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                        received += len(block)
                if expected and received != expected:
                    raise IOError(f'Incomplete response: {received}/{expected}')
            partial.replace(dest)
            print(f'DOWNLOADED {dest.name} bytes={dest.stat().st_size}', flush=True)
            return dest
        except (OSError, urllib.error.URLError) as err:
            print(f'RETRY {url} attempt={attempt+1}: {err}', flush=True)
            if attempt == 4:
                raise
            time.sleep(min(10, 2 ** attempt))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def fetch_checkpoint(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=45) as response:
        total = int(response.headers['Content-Length'])
    if dest.exists() and dest.stat().st_size == total:
        return dest
    count = 8
    chunk = (total + count - 1) // count
    def get_part(index):
        start, end = index * chunk, min(total, (index + 1) * chunk) - 1
        part = dest.with_suffix(f'.part{index}')
        expected = end - start + 1
        for attempt in range(5):
            current = part.stat().st_size if part.exists() else 0
            if current == expected:
                return part
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={start+current}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get('Content-Range', '').split('/')[0] != f'bytes {start+current}-{end}':
                        raise IOError('Server did not honor the exact range')
                    with part.open('ab') as out:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            out.write(block)
                if part.stat().st_size != expected:
                    raise IOError('Incomplete checkpoint part')
                return part
            except (OSError, urllib.error.URLError) as error:
                print(f'PART_RETRY {dest.parent.name} {index}: {error}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(10, 2 ** attempt))
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        parts = list(pool.map(get_part, range(count)))
    temp = dest.with_suffix('.assembled')
    with temp.open('wb') as out:
        for part in parts:
            with part.open('rb') as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    out.write(block)
    if temp.stat().st_size != total:
        raise IOError('Assembled checkpoint has wrong size')
    temp.replace(dest)
    for part in parts:
        part.unlink()
    print(f'DOWNLOADED {dest.parent.name}/{dest.name} bytes={total}', flush=True)
    return dest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='lift,can,square')
    parser.add_argument('--method', choices=['diffusion_policy_transformer', 'diffusion_policy_cnn'], default='diffusion_policy_transformer')
    args = parser.parse_args()
    tasks = args.tasks.split(',')
    if not set(tasks) <= {'lift', 'can', 'square'}:
        raise ValueError('Task is outside the fixed protocol')
    vendor = ROOT / 'vendor' / 'diffusion_policy'
    if not (vendor / 'diffusion_policy' / '__init__.py').exists():
        archive = fetch('https://codeload.github.com/real-stanford/diffusion_policy/tar.gz/refs/heads/main', ROOT / 'assets' / 'diffusion_policy_main.tar.gz')
        with tarfile.open(archive) as tf:
            # Reject unsafe upstream archive paths, then strip its top directory.
            vendor.mkdir(parents=True, exist_ok=True)
            for member in tf.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts or member.issym() or member.islnk():
                    continue
                if '..' in parts or Path(*parts).is_absolute():
                    raise ValueError('Unsafe source archive')
                member.name = str(Path(*parts))
                tf.extract(member, vendor)
    def get_task(task):
        dest = ROOT / 'assets' / args.method / task
        url = f'{BASE}/{task}_ph/{args.method}'
        cfg = fetch(f'{url}/config.yaml', dest / 'config.yaml')
        listing = urllib.request.urlopen(f'{url}/train_0/checkpoints/', timeout=45).read().decode()
        candidates = re.findall(r'epoch=(\d+)-test_mean_score=([0-9.]+)\.ckpt', listing)
        if not candidates:
            raise RuntimeError(f'No published checkpoint found for {task}')
        # Select the officially published highest clean score, before our noise tests.
        epoch, score = sorted(set(candidates), key=lambda x: (-float(x[1]), int(x[0])))[0]
        name = f'epoch={epoch}-test_mean_score={score}.ckpt'
        ckpt = fetch_checkpoint(f'{url}/train_0/checkpoints/{name}', dest / name)
        meta = fetch(f'https://downloads.cs.stanford.edu/downloads/rt_benchmark/{task}/ph/demo.hdf5', dest / 'original_demo.hdf5')
        return dict(task=task, method=args.method, source_url=f'{url}/train_0/checkpoints/{name}', checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256=sha256(ckpt), config=str(cfg.relative_to(ROOT)), metadata_hdf5=str(meta.relative_to(ROOT)), policy_training_run='train_0', selection_rule='published maximum clean score, tie earliest epoch; fixed before noisy evaluation')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(get_task, tasks))
    target = ROOT / 'assets' / 'manifest.json'
    target.write_text(json.dumps({'tasks': manifest, 'source_archive_sha256': sha256(ROOT / 'assets' / 'diffusion_policy_main.tar.gz')}, indent=2) + '\n')
    print(f'TASK_RESTORE_ASSETS_READY manifest={target}', flush=True)

if __name__ == '__main__':
    main()
