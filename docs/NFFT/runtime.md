# Runtime Comparison

The test script `test/tuntime_test.py` compares the runtime of this implementation with some other libraries.
For the comparisons, we always perform all precomputations, which only depend on the input shape and dimension,
but no precomputations depending on the basis points `x` or the function values `f` or `f_hat`.
We fix `M=100000`, `m=2` and `sigma=2`. Then, we obtain the following execution times on a NVIDIA RTX 4090 GPU averaged over 10 runs. 
Moreover, we run all methods with single precision. Note that pyNFFT3 only has a CPU implementation (24-core AMD Ryzen Threadripper 7960X s) and
always uses double precision such that the comparison with it is not really fair.

We always use the input shape `(batch_x,1,M,d)` for `x`, `(batch_x,batch_f,M)` for `f` and `(batch_x,batch_f,N_1,...,N_d)` for `f_hat`.

## One-dimensional NFFT

We use `N=4096`. Then the execution times (seconds) the forward NFFT were the following.

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.00226 | 0.00113 | 0.00140 | 0.00078 |
| `batch_x=1`, `batch_f=10` | 0.00505 | 0.00108 | 0.00333 | 0.00096 |
| `batch_x=10`, `batch_f=1` | 0.01340 | 0.00447 | 0.00346 | 0.00085 |
| `batch_x=10`, `batch_f=10` | 0.06744 | 0.00500 | 0.02241 | 0.00288 |

For the adjoint NFFT, we obtain the following execution times

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.00289 | 0.00270 | 0.00156 | 0.00091 |
| `batch_x=1`, `batch_f=10` | 0.01650 | 0.00289 | 0.00325 | 0.00183 |
| `batch_x=10`, `batch_f=1` | 0.02491 | 0.00613 | 0.00310 | 0.00183 |
| `batch_x=10`, `batch_f=10` | 0.15513 | 0.00804 | 0.02222 | 0.01483 |

## Two-dimensional NFFT

We use `N=(N_1,N_2)=(256,256)`. Then the execution times (seconds) for the forward NFFT were the following.

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.00489 | 0.00237 | 0.00257 | 0.00092 |
| `batch_x=1`, `batch_f=10` | 0.02079 | 0.00263 | 0.00973 | 0.00168 |
| `batch_x=10`, `batch_f=1` | 0.07189 | 0.01557 | 0.01020 | 0.00240 |
| `batch_x=10`, `batch_f=10` | 0.21706 | 0.02359 | 0.06724 | 0.01905 |

For the adjoint NFFT, we obtain the following execution times

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.00491 | 0.00502 | 0.00223 | 0.00117 |
| `batch_x=1`, `batch_f=10` | 0.02924 | 0.00649 | 0.00893 | 0.00725 |
| `batch_x=10`, `batch_f=1` | 0.07695 | 0.02437 | 0.00950 | 0.00791 |
| `batch_x=10`, `batch_f=10` | 0.30029 | 0.03246 | 0.06537 | 0.07389 |

## Three-dimensional NFFT

We use `N=(N_1,N_2,N_3)=(128,128,128)`. For `batch_x=batch_f=10`, we got an memory error on the GPU 
(should not be surprising when trying to perform 100 three-dimensional NFFTs in parallel).
Then the execution times (seconds) for the forward NFFT were the following.

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.03677 | 0.00888 | 0.01297 | 0.00458 |
| `batch_x=1`, `batch_f=10` | 0.27084 | 0.02778 | 0.06780 | 0.04576 |
| `batch_x=10`, `batch_f=1` | 0.30655 | 0.08501 | 0.06860 | 0.04744 |

For the adjoint NFFT, we obtain the following execution times

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.04423 | 0.01856 | 0.00950 | 0.00675 |
| `batch_x=1`, `batch_f=10` | 0.34766 | 0.04173 | 0.10051 | 0.06409 |
| `batch_x=10`, `batch_f=1` | 0.38263 | 0.11934 | 0.10062 | 0.06685 |