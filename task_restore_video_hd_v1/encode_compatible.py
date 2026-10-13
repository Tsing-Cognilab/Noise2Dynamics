"""Encode saved native policy-input pixels using broadly compatible H.264."""
import hashlib, json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'outputs/cvm_video_hd_v1/selected_20261011'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for job in json.loads((HERE/'jobs.json').read_text()):
    folder=OUT/job['name'];path=folder/'inputs_native.mp4'
    audit=json.loads((folder/'hd_audit.json').read_text())
    assert audit['original_source_sha256']==job['source_sha256']
    assert audit['low_resolution_reproduction']['equal_frames']==100
    assert all(audit['original_episode_comparisons'].values())
    with np.load(folder/'trajectory_and_native_inputs.npz') as data:
        panels=data['panels'];assert panels.shape==(100,3,84,84,3)
    encoded=folder/'inputs_native_encoding.mp4'
    with imageio.get_writer(encoded,fps=5,codec='libx264',macro_block_size=1,
                            pixelformat='yuv420p',ffmpeg_params=['-crf','14']) as writer:
        for row in panels:writer.append_data(np.concatenate([row[1],row[2]],axis=1))
    count=0
    with imageio.get_reader(encoded) as reader:
        for frame in reader:
            assert frame.shape==(84,168,3);count+=1
    assert count==100
    encoded.replace(path)
    audit['native_input_video_sha256']=sha(path)
    audit['native_input_video_pixel_format']='yuv420p'
    audit['native_input_source_pixels_sha256']=sha(folder/'trajectory_and_native_inputs.npz')
    audit['native_input_encoder_sha256']=sha(__file__)
    (folder/'hd_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('HD_INPUT_ENCODING_VERIFIED',job['name'],flush=True)
