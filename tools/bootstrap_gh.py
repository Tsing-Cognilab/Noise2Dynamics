#!/usr/bin/env python3
"""Install a checksum-verified official GitHub CLI in this snapshot only."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import tarfile
import urllib.request


def fetch(url):
    if not url.startswith(('https://api.github.com/repos/cli/cli/', 'https://github.com/cli/cli/releases/download/')):
        raise RuntimeError('Unexpected CLI download host/path')
    request = urllib.request.Request(url, headers={'User-Agent': 'CVM-private-repository-preparation'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def bootstrap(root):
    runtime = root / 'tools/.runtime'
    gh = runtime / 'gh'
    info = runtime / 'cli.json'
    if gh.is_file() and info.is_file():
        record = json.loads(info.read_text())
        if hashlib.sha256(gh.read_bytes()).hexdigest() != record['binary_sha256']:
            raise RuntimeError('Installed CLI differs from verified checksum')
        return gh
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise RuntimeError('This bootstrap supports the existing Linux amd64 server only')
    release = json.loads(fetch('https://api.github.com/repos/cli/cli/releases/latest'))
    version = release['tag_name'].removeprefix('v')
    name = 'gh_' + version + '_linux_amd64.tar.gz'
    urls = {a['name']: a['browser_download_url'] for a in release['assets']}
    checksums = fetch(urls['gh_' + version + '_checksums.txt']).decode()
    expected = next(line.split()[0] for line in checksums.splitlines() if line.split()[-1] == name)
    blob = fetch(urls[name])
    if hashlib.sha256(blob).hexdigest() != expected:
        raise RuntimeError('Official CLI archive checksum mismatch')
    runtime.mkdir(parents=True, exist_ok=True)
    archive = runtime / name
    archive.write_bytes(blob)
    with tarfile.open(archive, 'r:gz') as tar:
        member = tar.getmember('gh_' + version + '_linux_amd64/bin/gh')
        if not member.isfile():
            raise RuntimeError('Unexpected binary member')
        binary = tar.extractfile(member).read()
    gh.write_bytes(binary)
    gh.chmod(0o755)
    info.write_text(json.dumps({'version': version, 'source': urls[name], 'archive_sha256': expected,
                               'binary_sha256': hashlib.sha256(binary).hexdigest()}, indent=2) + '\n')
    print('Verified official GitHub CLI ' + version, flush=True)
    return gh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--login', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cli = bootstrap(root)
    if args.login:
        print('Authorize GitHub CLI for Audience2Li in your own browser. Do not send tokens to chat.', flush=True)
        subprocess.run([str(cli), 'auth', 'login', '--hostname', 'github.com', '--git-protocol', 'https', '--web'], check=True)
    else:
        subprocess.run([str(cli), '--version'], check=True)
