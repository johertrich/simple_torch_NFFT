# Background on Fast Kernel Summation via Slicing

Back to: [Main Page](../../readme.md), [Overview of Fast Kernel Summation](overview.md)

We describe the theoretical background for applying the NFFT for computing the kernel sums

$$s_m=\sum_{n=1}^N w_n K(x_n,y_m)$$

In the following, we briefly describe the theoretical background behind this implementation. First, we look into the fast Fourier summation for the one-dimensional case. Afterwards, we summarize the slicing approach for higher-dimensional kernel summation and finally, we describe how derivatives can be computed efficiently.

## Fast Fourier Summation in the one-dimensional Case

The following procedure for fast kernel summation in low (whenever the NFFT remains tracktable, i.e., $d\leq 3$) dimensions was proposed by [this paper](https://doi.org/10.1137/S1064827502400984) and is also implemented [here](https://www-user.tu-chemnitz.de/~potts/nfft). To keep the notations light, we stick to the one-dimensional case, even though the same procedure is also applicable for $d=2$ and $d=3$. However, for the slicing procedure later, we only need the case $d=1$.

Let $x_1,...,x_N$ and $y_1,...,y_M$ be contained in $(-\frac{1}{2},\frac{1}{2})$ such that $\|x_n - y_m\|<\frac{1}{2}$ for all $n$ and $m$. Moreover, let $\mathrm{k}\colon\mathbb{R}\times\mathbb{R}\to\mathbb{R}$ be a kernel of the form $\mathrm{k}(x,y)=g(x-y)$. Then, we can expand $g\colon(-\frac{1}{2},\frac{1}{2})\to\mathbb{R}$ into its Fourier series $g(x)=\sum_{k\in\mathbb{Z}}c_k(g)\mathrm{e}^{-2\pi i k x}$. By truncating the Fourier series to the coefficients $\mathcal{C}=\\{-\frac{N_{ft}}{2},\frac{N_{ft}}{2}-1\\}$, the kernel sum can be computed as

$$s_m=\sum_{n=1}^N w_n g(x_n-y_m)\approx\sum_{k\in\mathcal C}c_k(g)\sum_{n=1}^N w_N \mathrm{e}^{2\pi i k(y_m-x_n)}=\sum_{k\in\mathcal C}c_k(g)\mathrm{e}^{2\pi i y_m}\sum_{n=1}^N w_N \mathrm{e}^{-2\pi i x_n}$$

This can now by one adjoint and one forward application of the NFFT:

1. Compute $\hat w_k=\sum_{n=1}^N w_N \mathrm{e}^{-2\pi i x_n}$ by the adjoint NFFT
2. Compute $s_m=\sum_{k\in\mathcal C}c_k(g)\hat w_k\mathrm{e}^{2\pi i y_m}$ by the forward NFFT

## Fast Kernel Summation in Higher Dimensions via Slicing

Let $K$ be a sliced kernel of the form

$$K(x,y)=\mathbb{E}_{\xi\in\mathbb{S}^{d-1}}[\mathrm{k}(\langle \xi,x\rangle,\langle\xi, y\rangle)].$$

Then, we can compute the kernel sums by discretizing the expectation as

$$s_m\approx\frac1P\sum_{p=1}^P\sum_{n=1}^N w_n\mathrm{k}(\langle \xi_p,x\rangle,\langle\xi_p, y\rangle)].$$

In particular, we can compute $P$ one-dimensional kernel sums instead of one high-dimensional kernel sum.
Efficient ways how to find the kernel pairs $K$ and $\mathrm{k}$ were investigated [here](https://doi.org/10.1137/24M1632085), see also the papers [here](https://arxiv.org/abs/2408.11612) and [here](https://arxiv.org/abs/2410.01316). 
The slices $\xi_1,...,\xi_P$ can be chosen either iid from the unit sphere or by some certain QMC rules.

## Efficient Derivatives for Sliced Kernel Summation

Backpropagation through the sliced kernel summation is not memory efficient, since the computation over all slices must be traced. To avoid this, we overwrite the backward pass again by using again the fast kernel evaluation (and slicing) for computing the derivatives.

### Derivative with respect to $x_n$

Let $K$ be a sliced kernel as above.
Assuming that $\mathrm{k}(x,y)=g(x-y)$, the kernel can be rewritten as

$$K(x,y)=\mathbb{E}_{\xi\in\mathbb{S}^{d-1}}[g(\langle \xi,x-y\rangle)].$$

Under mild assumptions on the kernel this implies that

$$\nabla_x K(x,y)=\mathbb{E}_{\xi\in\mathbb{S}^{d-1}}[g'(\langle \xi,x-y\rangle)\xi]$$

In particular, it holds that

$$\nabla_{x_n} s_m = \mathbb{E}_{\xi\in\mathbb{S}^{d-1}}\left[w_n g'(\langle \xi,x_n-y_m\rangle)\xi\right]$$

Discretizing the expectation gives

$$\nabla_{x_n} s_m \approx \frac1P\sum_{p=1}^P w_n g'(\langle \xi_p,x_n-y_m\rangle)\xi_p.$$

Hence, given sensitivities $\bar s_1,...,\bar s_M$, we can compute the sensitivities $\bar x_1,...,\bar x_N$ by

$$\bar x_n=\frac{w_n}{P}\sum_{p=1}^P\left(\sum_{m=1}^M\bar s_m g'(\langle \xi_p,x_n-y_m\rangle)\right)\xi_p.$$

Now, the inner sum can be computed by the NFFT as described above. Note that the Fourier coefficients of $g'$ can be easily derived by the differentiation-multiplication formula of the Fourier transform as $c_k(g')=2\pi i k c_k(g)$.

### Derivative with respect to $y_m$

Similar as for the derivative wrt. $x$, it holds

$$\nabla_{y_m} s_m \approx -\frac1P\sum_{p=1}^P \sum_{n=1}^N w_n g'(\langle \xi_p,x_n-y_m\rangle)\xi_p.$$

Hence, given sensitivities $\bar s_1,...,\bar s_M$, we can compute the sensitivities $\bar y_1,...,\bar y_M$ by

$$\bar y_m=-\frac{\bar s_m}{P}\sum_{p=1}^P\left(\sum_{n=1}^Nw_n g'(\langle \xi_p,x_n-y_m\rangle)\right)\xi_p.$$

As for the derivative wrt. $x$, this can now be computed by the NFFT.

### Derivative with respect to $w_n$

It holds that
$$\nabla_{w_n} s_m= K(x_n,y_m)$$
Hence, given sensitivities $\bar s_1,...,\bar s_M$ the computation of the sensitivities $\bar w_1,...,\bar w_N$ is given by 
$$\bar w_n=\sum_{m=1}^M \bar s_m K(x_n,y_m),$$
which is again a kernel summation, which can be computed via slicing.

## Citation

The fast kernel summation via slicing was introduced in this paper:

```
@article{H2024,
  title={Fast Kernel Summation in High Dimensions via Slicing and {F}ourier transforms},
  author={Hertrich, Johannes},
  journal={SIAM Journal on Mathematics of Data Science},
  volume={6},
  number={4},
  pages={1109--1137},
  year={2024}
}
```

We proved tighter error bounds and considered the non-iid slicing directions in:

```
@article{HJQ2024,
  title={Fast Summation of Radial Kernels via {QMC} Slicing},
  author={Hertrich, Johannes and Jahn, Tim and Quellmalz, Michael},
  journal={arXiv preprint arXiv:2410.01316},
  year={2024}
}
```


