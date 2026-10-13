#!/usr/bin/env python3
"""User-run private publication; retries skip SHA256-verified uploaded chunks."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from bootstrap_gh import bootstrap

REPO = 'Audience2Li/Noise2Dynamics'
LOGIN = 'Audience2Li'


def log(msg):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + msg, flush=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def publish(root):
    os.umask(0o077)
    if not (root / 'manifests/PREPARATION.json').is_file():
        raise RuntimeError('CVM snapshot not prepared')
    gh = bootstrap(root)

    def call(*args, capture=True, allow404=False):
        result = subprocess.run([str(gh), *args], text=True, stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE if capture else None)
        if result.returncode:
            if allow404 and 'HTTP 404' in (result.stderr or ''):
                return None
            if capture:
                # No tokens or full command environments are printed.
                print((result.stderr or '')[-1800:], file=sys.stderr)
            raise RuntimeError('GitHub command failed: ' + args[0])
        return result.stdout if capture else ''

    result = subprocess.run([str(gh), 'auth', 'status', '--hostname', 'github.com'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError('Login first: python3 tools/bootstrap_gh.py --login')
    login = json.loads(call('api', 'user'))['login']
    if login != LOGIN:
        raise RuntimeError('Wrong authenticated account; expected Audience2Li')
    log('Authenticated as Audience2Li; checking every source/evidence checksum')
    files = [json.loads(line) for line in (root / 'manifests/files.jsonl').read_text().splitlines()]
    for record in files:
        path = root / record['path']
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise RuntimeError('Snapshot changed: ' + record['path'])
    packages = json.loads((root / 'manifests/release-manifest.json').read_text())
    parts = [p for pack in packages['packages'] for p in pack['parts']]
    extras = ['release-manifest.json', 'SHA256SUMS']
    parts += [{'name': n, 'bytes': (root / 'artifacts' / n).stat().st_size,
               'sha256': sha(root / 'artifacts' / n)} for n in extras]
    for part in parts:
        p = root / 'artifacts' / part['name']
        if p.stat().st_size != part['bytes'] or sha(p) != part['sha256']:
            raise RuntimeError('Invalid local upload part: ' + part['name'])

    metadata = call('api', 'repos/' + REPO, allow404=True)
    if metadata is None:
        call('repo', 'create', REPO, '--private', '--description',
             'CVM visual restoration with continuous decision supervision; code, records and private assets.')
        metadata = call('api', 'repos/' + REPO)
    metadata = json.loads(metadata)
    if metadata['full_name'] != REPO or not metadata['private']:
        raise RuntimeError('Destination must be the exact personal private repository; visibility is never changed')

    def git(*args, check=True):
        return subprocess.run(['git', '-C', str(root), *args], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=check)

    git('config', 'user.name', LOGIN)
    git('config', 'user.email', '208441070+Audience2Li@users.noreply.github.com')
    branch = git('symbolic-ref', '--short', 'HEAD').stdout.strip()
    if branch != 'main':
        raise RuntimeError('Expected prepared main branch')
    remote = git('remote', 'get-url', 'origin', check=False)
    url = 'https://github.com/' + REPO + '.git'
    if remote.returncode == 0 and remote.stdout.strip() != url:
        raise RuntimeError('Unexpected origin remote; refusing to replace it')
    if remote.returncode:
        git('remote', 'add', 'origin', url)
    git('add', '--all')
    tracked = git('ls-files', '-z').stdout.split('\0')
    forbidden = {r['path'] for r in files if r['kind'] != 'git'}
    for rel in filter(None, tracked):
        if rel in forbidden or (root / rel).stat().st_size >= 50 * 1024 * 1024:
            raise RuntimeError('Binary asset incorrectly tracked by Git: ' + rel)
    head = git('rev-parse', '--verify', 'HEAD', check=False)
    if head.returncode:
        if metadata['size'] != 0 or json.loads(call('api', 'repos/' + REPO + '/branches')):
            raise RuntimeError('Remote already has history. Do not overwrite it; inspect before continuing.')
        git('commit', '-m', 'Preserve CVM code, protocols and complete experiment records')
    elif git('status', '--porcelain').stdout.strip():
        raise RuntimeError('Prepared commit has subsequent changes; inspect before uploading')
    commit = git('rev-parse', 'HEAD').stdout.strip()
    log('Pushing CVM-only Git history, without data/model binaries')
    # Scoped credential helper: no global Git configuration and no token extraction.
    credential = '!' + shlex.quote(str(gh)) + ' auth git-credential'
    subprocess.run(['git', '-C', str(root), '-c', 'credential.helper=', '-c',
                    'credential.helper=' + credential, 'push', '-u', 'origin', 'main'], check=True)
    remote_head = json.loads(call('api', 'repos/' + REPO + '/git/ref/heads/main'))['object']['sha']
    if remote_head != commit:
        raise RuntimeError('Uploaded branch does not match prepared commit')

    tag = packages['tag']
    endpoint = 'repos/' + REPO + '/releases/tags/' + tag
    release_text = call('api', endpoint, allow404=True)
    notes = 'CVM-only data, frozen policies, trained restorers and supporting resources. ' \
            'SHA256 checksums and restore instructions are in the repository. Retired N2D/latent/Dreamer assets are excluded.'
    if release_text is None:
        call('release', 'create', tag, '--repo', REPO, '--target', commit, '--draft',
             '--title', 'CVM reproducibility assets', '--notes', notes)
        release_text = call('api', endpoint)
    release = json.loads(release_text)
    if release['tag_name'] != tag:
        raise RuntimeError('Unexpected release')
    progress_path = root / 'upload-progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for index, part in enumerate(parts, 1):
        release = json.loads(call('api', endpoint))
        matches = [a for a in release['assets'] if a['name'] == part['name']]
        if len(matches) > 1:
            raise RuntimeError('Duplicate release asset names')
        current = matches[0] if matches else None
        if current is not None:
            digest = current.get('digest')
            if current['size'] != part['bytes'] or current.get('state') != 'uploaded':
                raise RuntimeError('Conflicting existing release asset; refusing to delete/overwrite: ' + part['name'])
            if digest is not None:
                verified = digest == 'sha256:' + part['sha256']
            else:
                old = progress.get(part['name'], {})
                verified = old.get('id') == current['id'] and old.get('sha256') == part['sha256']
            if not verified:
                raise RuntimeError('Existing asset not checksum-verified; refusing to skip or overwrite: ' + part['name'])
            log('SKIP verified (' + str(index) + '/' + str(len(parts)) + ') ' + part['name'])
            continue
        log('UPLOAD (' + str(index) + '/' + str(len(parts)) + ') ' + part['name'])
        call('release', 'upload', tag, str(root / 'artifacts' / part['name']), '--repo', REPO, capture=False)
        release = json.loads(call('api', endpoint))
        current = next(a for a in release['assets'] if a['name'] == part['name'])
        if current['size'] != part['bytes'] or current.get('state') != 'uploaded':
            raise RuntimeError('Incomplete uploaded asset: ' + part['name'])
        digest = current.get('digest')
        if digest and digest != 'sha256:' + part['sha256']:
            raise RuntimeError('GitHub asset digest mismatch')
        if not digest:
            # Older API responses lack a digest. Verify once by streaming the asset back.
            h = hashlib.sha256()
            proc = subprocess.Popen([str(gh), 'api', 'repos/' + REPO + '/releases/assets/' + str(current['id']),
                                     '-H', 'Accept: application/octet-stream'], stdout=subprocess.PIPE)
            for block in iter(lambda: proc.stdout.read(4 * 1024 * 1024), b''):
                h.update(block)
            if proc.wait() != 0 or h.hexdigest() != part['sha256']:
                raise RuntimeError('Downloaded verification failed')
        progress[part['name']] = {'id': current['id'], 'sha256': part['sha256'], 'bytes': part['bytes']}
        atomic_json(progress_path, progress)
    private = json.loads(call('api', 'repos/' + REPO))['private']
    if not private:
        raise RuntimeError('Repository is no longer private; release not published')
    call('release', 'edit', tag, '--repo', REPO, '--draft=false')
    atomic_json(root / 'UPLOAD_COMPLETE.json', {'repository': REPO, 'private': True, 'commit': commit,
                                             'tag': tag, 'verified_upload_files': len(parts),
                                             'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    log('COMPLETE: https://github.com/' + REPO + ' (PRIVATE)')


if __name__ == '__main__':
    try:
        root = Path(__file__).resolve().parents[1]
        with (root / '.upload.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            publish(root)
    except (Exception, KeyboardInterrupt) as exc:
        log('STOPPED: ' + str(exc))
        log('Completed verified chunks are retained. Rerun the same command; nothing is force-pushed or deleted.')
        sys.exit(1)
