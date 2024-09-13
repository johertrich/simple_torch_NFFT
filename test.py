from nfft import NFFT, ndft_adjoint, ndft_forward
import torch
import torchkbnufft as tkbn
import time

device='cuda'

N=1024
J=20000
k=.3*(torch.rand((J,),device=device)-.5)
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


##############################################
###### Compare with torchkbnufft... ##########
##############################################


J=2000000
k=.3*(torch.rand((J,),device=device)-.5)
k=k[None,:]

# test data
f=torch.randn(k.shape,dtype=torch.complex64,device=device)

torch.cuda.synchronize()
tic=time.time()
# compute NFFT
fHat=nfft.adjoint(k,f)
torch.cuda.synchronize()
toc=time.time()-tic
print(toc)


k_kbnufft=k*torch.pi

adjnufft_ob = tkbn.KbNufftAdjoint(im_size=(N,)).to(device)


torch.cuda.synchronize()
tic1=time.time()
interp_mats= tkbn.calc_tensor_spmatrix(
   k_kbnufft,
   im_size=(N,)
)
torch.cuda.synchronize()
tic=time.time()
fHat=adjnufft_ob(f[None],k_kbnufft,interp_mats)
torch.cuda.synchronize()
toc=time.time()-tic
toc1=toc+tic-tic1
print(toc,toc1)

