# NCHC_hackthon
Implement NVLM with the Flax library

# How to setup environment

```
pip install -e .
```
After doing this step, the inference and model code will be under the `NVLMF` python package. You can access the model and the inference code under `import NVLMF.*`.

## For installing JAX

### CPU only:

```
pip install -U --pre jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```

### Google Cloud TPU:

```
pip install -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### NVIDIA GPU (CUDA 12):

```
pip install -U --pre jax jaxlib jax-cuda12-plugin[with_cuda] jax-cuda12-pjrt -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```


