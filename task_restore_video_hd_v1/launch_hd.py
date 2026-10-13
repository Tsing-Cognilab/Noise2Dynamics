"""Bounded recording jobs on idle GPUs; never interrupt unrelated jobs."""
import argparse, concurrent.futures, hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO/'outputs/cvm_video_hd_v1/selected_20261011'
sys.path.insert(0,str(REPO/'task_restore_v1'))
from run import assert_gpu_free, IMAGE

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jobs',default='all')
    p.add_argument('--gpus',default='0,1,2,3')
    a=p.parse_args()
    jobs=json.loads((HERE/'jobs.json').read_text())
    names=None if a.jobs=='all' else set(a.jobs.split(','))
    jobs=[j for j in jobs if names is None or j['name'] in names]
    if names and names != {j['name'] for j in jobs}:raise RuntimeError('Unknown job')
    pending=[]
    for job in jobs:
        audit=OUT/job['name']/'hd_audit.json'
        if audit.exists():
            data=json.loads(audit.read_text())
            if data['low_resolution_reproduction']['equal_frames']!=100:raise RuntimeError('Invalid prior HD audit')
            print('HD_ALREADY_VERIFIED',job['name'],flush=True)
        else:pending.append(job)
    if not pending:return
    gpus=[int(g) for g in a.gpus.split(',')]
    for gpu in gpus:assert_gpu_free(gpu)
    (OUT/'_logs').mkdir(parents=True,exist_ok=True)
    groups=[pending[i::len(gpus)] for i in range(len(gpus))]
    def worker(index):
        gpu=gpus[index]
        for job in groups[index]:
            assert_gpu_free(gpu)
            name='cvm-video-hd-'+hashlib.sha256(job['name'].encode()).hexdigest()[:12]
            cmd=['docker','run','--rm','--init','--name',name,'--shm-size','2g','--gpus',f'device={gpu}',
                '-e','CUDA_VISIBLE_DEVICES=0','-e','PYTHONPATH=',
                '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
                '-e','NUMEXPR_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
                '-e','TORCH_HOME=/workspace/task_restore_v1/cache/torch',
                '-v',f'{REPO}:/workspace','-w','/workspace',IMAGE,
                '/opt/taskrestore/bin/python','-u','task_restore_video_hd_v1/capture_hd.py','--job',job['name']]
            print('HD_JOB_STARTED',job['name'],'gpu',gpu,flush=True)
            with (OUT/'_logs'/f'{job["name"]}.log').open('w') as log:
                process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith(('HD_','METHOD_EPISODE','EXTERNAL_EPISODE')):print(line.strip(),flush=True)
                if process.wait():raise RuntimeError(f'HD capture failed: {job["name"]}; inspect its log')
            print('HD_JOB_FINISHED',job['name'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker,range(len(gpus))))
    print('HD_SELECTED_RECORDINGS_FINISHED',flush=True)

if __name__=='__main__':main()
