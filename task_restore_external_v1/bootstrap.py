"""Fetch pinned official MIT source and the author-linked LOL-v1 checkpoint."""
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = '1e9a0efce4b306b6701b824768370ff26066c32a'
BASE = 'https://raw.githubusercontent.com/caiyuanhao1998/Retinexformer/' + COMMIT + '/'
FILES = {
    'vendor/retinexformer_arch.py': BASE + 'basicsr/models/archs/RetinexFormer_arch.py',
    'vendor/RetinexFormer_LOL_v1.yml': BASE + 'Options/RetinexFormer_LOL_v1.yml',
    'vendor/LICENSE.txt': BASE + 'LICENSE.txt',
    'assets/LOL_v1.pth': 'https://drive.usercontent.google.com/download?id=1xDwQtTCj3tlAVCTJgYrzonBGVwqeOhKu&export=download&authuser=0&confirm=t',
}

def main():
    entries = {}
    for name, url in FILES.items():
        path = HERE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url, timeout=45) as response:
                value = response.read(10000001)
            if len(value) > 10000000:
                raise RuntimeError('Unexpected oversized asset')
            if name.endswith('.pth') and (len(value) != 6475021 or not value.startswith(b'PK')):
                raise RuntimeError('Not the official 6475021-byte PyTorch checkpoint; no substitute allowed')
            temporary = path.with_suffix(path.suffix + '.download')
            temporary.write_bytes(value)
            temporary.replace(path)
        entries[name] = dict(url=url, bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print('ASSET_READY', name, entries[name]['bytes'], entries[name]['sha256'], flush=True)
    manifest = dict(repository='https://github.com/caiyuanhao1998/Retinexformer', commit=COMMIT,
                    official_folder='https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV',
                    weights='LOL_v1.pth; chosen before robotic evaluation', files=entries,
                    gt_mean=False, self_ensemble=False,
                    architecture=dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2]))
    path = HERE / 'assets/manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Official asset manifest changed; refusing overwrite')
    path.write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
