from nfft import NFFT, ndft_adjoint, ndft_forward
import torch
import torchkbnufft as tkbn

device='cuda'

N=1024
J=1024
k=.1*(torch.rand((J,),device=device)-.5)
k=k[None,:]
m=8
sigma=2

n=2*N

# for NDFT comparison
ft_grid=torch.arange(-N//2,N//2,dtype=torch.float32,device=device)

# init nfft
nfft=NFFT(N,m,sigma)

#################################
###### Test adjoint... ##########
#################################

# test data
f=torch.randn(k.shape,dtype=torch.complex64,device=device)

# compute NFFT
fHat=nfft.adjoint(k,f)

# comparison with NDFT
fHat_dft=ndft_adjoint(k.squeeze(),f.squeeze(),ft_grid)

# relative error
print("Relativer Fehler", torch.sqrt(torch.sum(torch.abs(fHat-fHat_dft)**2)/torch.sum(torch.abs(fHat_dft)**2)))


#################################
###### Test forward... ##########
#################################

# test data
fHat=torch.randn((k.shape[0],N),dtype=torch.complex64,device=device)

# compute NFFT
f=nfft(k,fHat)

# comparison with NDFT
f_dft=ndft_forward(k.squeeze(),fHat.squeeze(),ft_grid)

# relative error
print("Relativer Fehler", torch.sqrt(torch.sum(torch.abs(f-f_dft)**2)/torch.sum(torch.abs(f_dft)**2)))


