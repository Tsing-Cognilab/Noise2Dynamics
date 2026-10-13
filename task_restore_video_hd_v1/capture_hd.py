"""Repeat a locked selected episode, validate it, then render its states offline.

Lives outside all hashed training/evaluation modules. Original code, weights,
84x84 policy inputs, noise, state banks, and published result files stay intact.
"""
import argparse, hashlib, inspect, json, os, random, runpy, sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

REPO = Path('/workspace')
HERE = Path(__file__).resolve().parent
OUTPUT = REPO / 'outputs/cvm_video_hd_v1/selected_20261011'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

class Capture:
    def __init__(self, path, factory, *args, **kwargs):
        self.writer = factory(path, *args, **kwargs)
        self.states, self.panels, self.steps = [], [], []
        self.model, self.cfg, self.record = None, None, None
    def append_data(self, data):
        frame = inspect.currentframe().f_back
        local = frame.f_locals
        state = local['env'].get_state()
        if self.model is None:
            self.model = state['model']
            self.cfg = local['cfg']
            self.record = dict(local['record'])
        self.states.append(np.asarray(state['states'], dtype=np.float64).copy())
        self.steps.append(int(local['steps']))
        self.panels.append(np.stack([
            (x.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
            for x in local['images']
        ]))
        self.writer.append_data(data)
        del local, frame
    def close(self):
        self.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    job = next(j for j in json.loads((HERE/'jobs.json').read_text()) if j['name']==args.job)
    source = REPO/job['source_relative']
    original_protocol = json.loads((source.parent/'protocol.json').read_text())
    if original_protocol != job['protocol'] or sha(source) != job['source_sha256']:
        raise RuntimeError('Original recording/protocol differs from audited selection')
    if sha(REPO/job['bank']) != original_protocol['state_bank_sha256']:
        raise RuntimeError('State-bank hash mismatch')
    if job['checkpoint'] and sha(REPO/job['checkpoint']) != original_protocol['adapter_sha256']:
        raise RuntimeError('Frozen restorer hash mismatch')
    if args.preflight:
        print('HD_PREFLIGHT_OK',args.job,flush=True)
        return
    output = OUTPUT/args.job
    if output.exists():
        raise RuntimeError('Use a fresh output job; existing audited data is not overwritten')
    output.mkdir(parents=True)
    script = REPO/job['script']
    sys.path.insert(0,str(script.parent))
    sys.path.insert(0,str(REPO/'task_restore_method_v1'))
    # Preserve the evaluator's normal imageio writer and compression settings.
    factory = imageio.get_writer
    captures = []
    def recording_writer(path, *values, **options):
        if Path(path).parent == output:
            capture = Capture(path,factory,*values,**options)
            captures.append(capture)
            return capture
        return factory(path,*values,**options)
    imageio.get_writer = recording_writer
    p = original_protocol
    sys.argv = [str(script),'--task',p['task'],'--arm',p['arm'],'--condition',p['condition'],
        '--output',str(output),'--state-bank',str(REPO/job['bank']),
        '--episodes',str(p['episodes']),'--start-seed',str(p['start_seed']),
        '--offset',str(job['expected_row']['seed']-p['start_seed']),'--count','1',
        '--max-steps',str(p['max_steps']),'--video-episodes',str(p['episodes'])]
    if 'bank_episodes' in p:
        sys.argv += ['--bank-episodes',str(p['bank_episodes'])]
    if job['checkpoint']:
        sys.argv += ['--checkpoint',str(REPO/job['checkpoint'])]
    if p.get('training_seed') is not None:
        sys.argv += ['--training-seed',str(p['training_seed'])]
    try:
        namespace = runpy.run_path(str(script),run_name='__main__')
    finally:
        imageio.get_writer = factory
    row = json.loads((output/'result.json').read_text())['rows'][0]
    fields = ['task','arm','condition','seed','steps','success_ever','success_final','first_success_step',
        'initial_state_sha256','initial_model_sha256','state_bank_entry_sha256']
    comparisons = {k: row.get(k)==job['expected_row'].get(k) for k in fields}
    if not all(comparisons.values()):
        write_json(output/'failed_reproduction.json',{'comparisons':comparisons,'row':row})
        raise RuntimeError('Selected episode outcomes/initial state did not reproduce')
    if len(captures)!=1 or captures[0].steps!=list(range(4,401,4)):
        raise RuntimeError('Incorrect recorded-step sequence')
    capture = captures[0]
    fresh = output/f'episode_{row["seed"]}.mp4'
    equal_frames, changed_frames, max_pixel_error = 0,0,0
    with imageio.get_reader(source) as prior, imageio.get_reader(fresh) as again:
        for old, new in zip(prior,again):
            equal = np.array_equal(old,new)
            equal_frames += int(equal);changed_frames += int(not equal)
            max_pixel_error = max(max_pixel_error,int(np.abs(old.astype(np.int16)-new.astype(np.int16)).max()))
    if equal_frames != 100 or changed_frames:
        write_json(output/'failed_reproduction.json',{'equal_frames':equal_frames,'changed_frames':changed_frames,'max_pixel_error':max_pixel_error,'row':row})
        raise RuntimeError('Original low-resolution frame sequence is not bitwise reproduced; do not publish it as the same episode')
    print('HD_REPRODUCTION_VERIFIED',args.job,'100/100 frames bitwise equal',flush=True)
    np.savez_compressed(output/'trajectory_and_native_inputs.npz',
        states=np.stack(capture.states),steps=np.array(capture.steps),panels=np.stack(capture.panels))
    (output/'model.xml').write_text(capture.model)
    create_env, model_seeds = namespace['create_env'], namespace['MODEL_SEEDS']
    random.seed(model_seeds[p['task']]);np.random.seed(model_seeds[p['task']])
    env, meta = create_env(capture.record,capture.cfg)
    max_state_error = 0.
    try:
        if meta != json.loads((REPO/job['bank']).read_text())['env_metadata']:
            raise RuntimeError('Offline renderer metadata changed')
        env.reset_to({'model':capture.model,'states':capture.states[0]})
        hd_path, native_path = output/'simulation_hd.mp4', output/'inputs_native.mp4'
        with factory(hd_path,fps=5,codec='libx264',macro_block_size=1,ffmpeg_params=['-crf','16'],pixelformat='yuv420p') as hd, \
             factory(native_path,fps=5,codec='libx264',macro_block_size=1,ffmpeg_params=['-crf','14'],pixelformat='yuv444p') as native:
            for i,state in enumerate(capture.states):
                env.reset_to({'states':state})
                actual = np.asarray(env.get_state()['states'],dtype=np.float64)
                delta = np.abs(actual-state)
                max_state_error = max(max_state_error,float(delta.max()))
                if np.any(delta>8*np.finfo(np.float64).eps*np.maximum(1.,np.abs(state))):
                    raise RuntimeError('Offline state restoration exceeds original precision contract')
                clean = env.render(mode='rgb_array',height=640,width=640,camera_name='agentview')
                if clean.shape!=(640,640,3) or clean.dtype != np.uint8:
                    raise RuntimeError('Unexpected native high-resolution render shape')
                if not np.array_equal(np.asarray(env.get_state()['states']),actual):
                    raise RuntimeError('Rendering changed simulator state')
                hd.append_data(clean)
                native.append_data(np.concatenate([capture.panels[i][1],capture.panels[i][2]],axis=1))
                if i in [0,49,99]:
                    imageio.imwrite(output/f'hd_preview_{(i+1)*4:03d}.png',clean)
    finally:
        env.env.close()
    audit = dict(name=args.job,native_render_resolution=[640,640],native_input_panel_resolution=[84,84],
        fps=5,frames=100,steps=list(range(4,401,4)),original_source_sha256=sha(source),
        low_resolution_reproduction=dict(equal_frames=equal_frames,changed_frames=changed_frames,max_pixel_error=max_pixel_error),
        original_episode_comparisons=comparisons,original_row=job['expected_row'],rerecorded_row=row,
        maximum_state_restore_error=max_state_error,
        simulator_video_sha256=sha(hd_path),native_input_video_sha256=sha(native_path),
        capture_script_sha256=sha(__file__),source_evaluator_sha256=sha(script),
        description='Offline native 640x640 MuJoCo rendering of exactly reproduced recorded physical states; original policy input remains 84x84. Not additional evaluation evidence.')
    write_json(output/'hd_audit.json',audit)
    print('HD_CAPTURE_DONE',args.job,flush=True)

if __name__=='__main__':
    main()
