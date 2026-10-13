FROM noise2dynamics:franka-v1
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libglew-dev libosmesa6-dev patchelf curl git && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/taskrestore && /opt/taskrestore/bin/pip install --no-cache-dir 'numpy==1.23.5' 'scipy==1.10.1' 'Cython==0.29.37' 'opencv-python-headless==4.8.1.78' 'gym==0.23.1' 'free-mujoco-py==2.1.6' 'robomimic==0.2.0' 'numba==0.57.1' 'zarr==2.16.1' 'numcodecs==0.11.0' 'diffusers==0.11.1' 'huggingface-hub==0.23.4' 'dill==0.3.8'
RUN mkdir -p /opt/mujoco && curl --fail --location --retry 3 https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz -C /opt/mujoco && /opt/taskrestore/bin/pip install --no-deps https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz
RUN /opt/taskrestore/bin/pip install --no-cache-dir 'matplotlib==3.7.5' 'h5py==3.8.0'
ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV MUJOCO_PY_FORCE_CPU=1
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=1
ENV USE_FLAX=0
ENV USE_TF=0
ENV USE_TORCH=1
ENV PYTHONPATH=""
COPY legacy_compat.py /opt/taskrestore/lib/python3.10/site-packages/sitecustomize.py
RUN /opt/taskrestore/bin/python -c "import collections, collections.abc; collections.Iterable = collections.abc.Iterable; import mujoco_py, robosuite, robomimic, torch; assert robosuite.__version__ == '1.2.0' and robomimic.__version__ == '0.2.0'; print('TASK_RESTORE_ENV_OK', robosuite.__version__, robomimic.__version__, torch.__version__)"
WORKDIR /workspace
