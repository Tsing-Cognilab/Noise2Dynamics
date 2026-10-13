#!/usr/bin/env python3
"""User-run dual publication with repository-specific checksummed checkpoints."""
import argparse
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
COMPANY_REPO = 'Tsing-Cognilab/Noise2Dynamics'
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


def find_release(call, repo, tag):
    """List releases, including drafts; tag lookup alone cannot find drafts."""
    matches = []
    for page in range(1, 101):
        batch = json.loads(call('api', 'repos/' + repo + '/releases?per_page=100&page=' + str(page)))
        matches.extend(r for r in batch if r['tag_name'] == tag)
        if len(batch) < 100:
            break
    else:
        raise RuntimeError('Too many release pages; inspect before proceeding')
    if len(matches) > 1:
        raise RuntimeError('Multiple releases with the same tag; refusing to choose one')
    return matches[0] if matches else None


def publish(root, repo=REPO, check_only=False):
    os.umask(0o077)
    if repo not in (REPO, COMPANY_REPO):
        raise RuntimeError('Repository is not one of the two authorized destinations')
    expected_private = repo == REPO
    owner = repo.split('/')[0]
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
    log('TARGET ' + repo + (' (PRIVATE)' if expected_private else ' (PUBLIC)'))
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

    metadata = call('api', 'repos/' + repo, allow404=True)
    if metadata is None:
        if check_only or not expected_private:
            raise RuntimeError('Expected repository is not accessible: ' + repo)
        call('repo', 'create', repo, '--private', '--description',
             'CVM visual restoration with continuous decision supervision; code, records and private assets.')
        metadata = call('api', 'repos/' + repo)
    metadata = json.loads(metadata)
    if metadata['full_name'] != repo or metadata['private'] != expected_private:
        raise RuntimeError('Wrong repository or visibility for ' + repo + '; uploader never changes visibility')
    if not metadata.get('permissions', {}).get('push', False):
        raise RuntimeError('No push permission for ' + repo)
    tag = packages['tag']
    release = find_release(call, repo, tag)
    if check_only:
        log('PRECHECK OK: ' + repo + '; existing release ID=' + str(release['id'] if release else None))
        return

    def git(*args, check=True):
        return subprocess.run(['git', '-C', str(root), *args], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=check)

    git('config', 'user.name', LOGIN)
    git('config', 'user.email', '208441070+Audience2Li@users.noreply.github.com')
    branch = git('symbolic-ref', '--short', 'HEAD').stdout.strip()
    if branch != 'main':
        raise RuntimeError('Expected prepared main branch')
    remote_name = 'origin' if expected_private else 'company'
    remote = git('remote', 'get-url', remote_name, check=False)
    url = 'https://github.com/' + repo + '.git'
    if remote.returncode == 0 and remote.stdout.strip() != url:
        raise RuntimeError('Unexpected origin remote; refusing to replace it')
    if remote.returncode:
        git('remote', 'add', remote_name, url)
    git('add', '--all')
    tracked = git('ls-files', '-z').stdout.split('\0')
    forbidden = {r['path'] for r in files if r['kind'] != 'git'}
    for rel in filter(None, tracked):
        if rel in forbidden or (root / rel).stat().st_size >= 50 * 1024 * 1024:
            raise RuntimeError('Binary asset incorrectly tracked by Git: ' + rel)
    head = git('rev-parse', '--verify', 'HEAD', check=False)
    if head.returncode:
        if metadata['size'] != 0 or json.loads(call('api', 'repos/' + repo + '/branches')):
            raise RuntimeError('Remote already has history. Do not overwrite it; inspect before continuing.')
        git('commit', '-m', 'Preserve CVM code, protocols and complete experiment records')
    elif git('status', '--porcelain').stdout.strip():
        raise RuntimeError('Prepared commit has subsequent changes; inspect before uploading')
    commit = git('rev-parse', 'HEAD').stdout.strip()
    log('Pushing CVM-only Git history to ' + repo + ', without data/model binaries')
    # Scoped credential helper: no global Git configuration and no token extraction.
    credential = '!' + shlex.quote(str(gh)) + ' auth git-credential'
    subprocess.run(['git', '-C', str(root), '-c', 'credential.helper=', '-c',
                    'credential.helper=' + credential, 'push', remote_name, 'main'], check=True)
    remote_head = json.loads(call('api', 'repos/' + repo + '/git/ref/heads/main'))['object']['sha']
    if remote_head != commit:
        raise RuntimeError('Uploaded branch does not match prepared commit')

    notes = 'CVM-only data, frozen policies, trained restorers and supporting resources. ' \
            'SHA256 checksums and restore instructions are in the repository. Retired N2D/latent/Dreamer assets are excluded.'
    if release is None:
        call('release', 'create', tag, '--repo', repo, '--target', commit, '--draft',
             '--title', 'CVM reproducibility assets', '--notes', notes)
        for attempt in range(5):
            release = find_release(call, repo, tag)
            if release is not None:
                break
            time.sleep(2)
        if release is None:
            raise RuntimeError('Created release not yet visible. Rerun; no duplicate release is created.')
    endpoint = 'repos/' + repo + '/releases/' + str(release['id'])
    release = json.loads(call('api', endpoint))
    if release['tag_name'] != tag:
        raise RuntimeError('Unexpected release')
    if release['draft'] and release.get('target_commitish') != commit:
        # Retain the existing draft ID/assets, but tag the corrected tool revision.
        call('api', endpoint, '--method', 'PATCH', '-f', 'target_commitish=' + commit)
        release = json.loads(call('api', endpoint))
        if release.get('target_commitish') != commit:
            raise RuntimeError('Draft target commit update was not confirmed')
    progress_path = root / ('upload-progress-' + owner + '.json')
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if expected_private and not progress and (root / 'upload-progress.json').exists():
        progress = json.loads((root / 'upload-progress.json').read_text())
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
        started = time.monotonic()
        call('release', 'upload', tag, str(root / 'artifacts' / part['name']), '--repo', repo, capture=False)
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
            proc = subprocess.Popen([str(gh), 'api', 'repos/' + repo + '/releases/assets/' + str(current['id']),
                                     '-H', 'Accept: application/octet-stream'], stdout=subprocess.PIPE)
            for block in iter(lambda: proc.stdout.read(4 * 1024 * 1024), b''):
                h.update(block)
            if proc.wait() != 0 or h.hexdigest() != part['sha256']:
                raise RuntimeError('Downloaded verification failed')
            proc.stdout.close()
        elapsed = time.monotonic() - started
        progress[part['name']] = {'id': current['id'], 'sha256': part['sha256'], 'bytes': part['bytes'],
                                 'elapsed_seconds': round(elapsed, 3),
                                 'verified_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        atomic_json(progress_path, progress)
        rate = part['bytes'] / max(elapsed, 0.001) / 1e6
        log('VERIFIED ' + part['name'] + ': ' + str(round(rate, 2)) + ' MB/s including verification')
    private = json.loads(call('api', 'repos/' + repo))['private']
    if private != expected_private:
        raise RuntimeError('Repository visibility changed; release not published')
    call('release', 'edit', tag, '--repo', repo, '--draft=false')
    release = json.loads(call('api', endpoint))
    if release['draft'] or release['tag_name'] != tag:
        raise RuntimeError('Release publication not confirmed')
    complete = {'repository': repo, 'private': expected_private, 'commit': commit,
                'tag': tag, 'release_id': release['id'], 'verified_upload_files': len(parts),
                'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    atomic_json(root / ('UPLOAD_COMPLETE_' + owner + '.json'), complete)
    if expected_private:
        atomic_json(root / 'UPLOAD_COMPLETE.json', complete)
    log('COMPLETE: https://github.com/' + repo + (' (PRIVATE)' if expected_private else ' (PUBLIC)'))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--target', choices=('personal', 'company', 'both'), default='personal')
        parser.add_argument('--check', action='store_true', help='Read-only preflight, no commits/push/releases')
        args = parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        with (root / '.upload.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            selected = (REPO, COMPANY_REPO) if args.target == 'both' else (REPO if args.target == 'personal' else COMPANY_REPO,)
            for destination in selected:
                publish(root, destination, check_only=args.check)
            if args.target == 'both' and not args.check:
                log('ALL TARGETS COMPLETE: personal PRIVATE backup + company PUBLIC delivery')
    except (Exception, KeyboardInterrupt) as exc:
        log('STOPPED: ' + str(exc))
        log('Completed verified chunks are retained. Rerun the same command; nothing is force-pushed or deleted.')
        sys.exit(1)
