from nfft import NFFT
import torch

device='cuda'

N=1024
J=1024
k=.5*(torch.rand((J,),device=device)-.5)
k=k[None,:]
f=torch.randn(k.shape,dtype=torch.complex64,device=device)
m=8
sigma=2

nfft=NFFT(N,m,sigma)


fHat=nfft.adjoint(k,f)

print(fHat)

f_=nfft(k,fHat)

print(torch.sum(torch.abs(f_-f)**2))
