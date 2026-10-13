"""Export only the two genuine 84x84 observation panels, without video loss."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'outputs/cvm_video_hd_v1/selected_20261011'
DEST=OUT/'native_inputs_lossless'
DEST.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report=[]
for job in json.loads((HERE/'jobs.json').read_text()):
    source=OUT/job['name']
    proof=json.loads((source/'hd_audit.json').read_text())
    assert proof['low_resolution_reproduction']['equal_frames']==100
    assert proof['original_source_sha256']==job['source_sha256']
    assert all(proof['original_episode_comparisons'].values())
    with np.load(source/'trajectory_and_native_inputs.npz') as data:
        panels=data['panels'];steps=data['steps']
    assert panels.shape==(100,3,84,84,3) and panels.dtype==np.uint8
    assert steps.tolist()==list(range(4,401,4))
    folder=DEST/job['name'];folder.mkdir(exist_ok=True)
    frames=[]
    for i,row in enumerate(panels):
        pixels=np.concatenate([row[1],row[2]],axis=1)
        path=folder/f'input_{steps[i]:03d}.png'
        imageio.imwrite(path,pixels)
        assert np.array_equal(imageio.imread(path),pixels)
        frames.append({'file':path.name,'step':int(steps[i]),'sha256':sha(path)})
    report.append({'name':job['name'],'panel_resolution':[84,84],'frames':frames,
                   'source_pixels_sha256':sha(source/'trajectory_and_native_inputs.npz'),
                   'original_source_video_sha256':job['source_sha256'],
                   'lossless_pixel_equality_verified':True})
    print('LOSSLESS_INPUTS_EXPORTED',job['name'],flush=True)
(DEST/'native_input_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
archive=OUT/'native_inputs_lossless.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    tar.add(DEST,arcname='raw_native')
print('LOSSLESS_ARCHIVE_READY',archive.name,archive.stat().st_size,sha(archive),flush=True)
