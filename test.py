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




"""
adjoint=True
if adjoint:
    


    print("Fehler", torch.sum(torch.abs(fHat-fHat_dft)**2))
    print("Fehler nur itensität", torch.sum((torch.abs(fHat)-torch.abs(fHat_dft))**2))

    import matplotlib.pyplot as plt
    plt.plot(ft_grid.detach().cpu().numpy(),torch.abs(fHat_dft).detach().cpu().numpy(),'rx')
    plt.plot(ft_grid.detach().cpu().numpy(),torch.abs(fHat).squeeze().detach().cpu().numpy(),'b+')
    plt.show()


    sortperm1=torch.sort(torch.abs(fHat_dft))[1]
    sortperm2=torch.sort(torch.abs(fHat.squeeze()))[1]

    fHat_perm=fHat.squeeze()[sortperm2]
    fHat_dft_perm=fHat_dft[sortperm1]
    print(fHat_perm[-5:],fHat_dft_perm[-5:])
    exit()



    print(fHat_perm[:5])
    print(fHat_dft_perm[:5])

    diffs=torch.abs(fHat_perm-fHat_dft_perm)
    print("failed")
    print(fHat_perm[diffs>1][:5])
    print(fHat_dft_perm[diffs>1][:5])

    print(torch.sum(diffs))

    print(torch.mean(torch.abs(fHat_dft))/torch.mean(torch.abs(fHat)))
    print(torch.mean(torch.abs(fHat_dft)))
    print((fHat.squeeze()-fHat_dft).shape)
    print(torch.sum(torch.abs(torch.flip(fHat.squeeze(),[0])-fHat_dft)**2)/torch.sum(torch.abs(fHat_dft)**2))
    plt.plot(ft_grid.detach().cpu().numpy(),diffs.detach().cpu().numpy())
    plt.show()

    plt.plot(ft_grid.detach().cpu().numpy(),torch.abs(fHat_dft_perm).detach().cpu().numpy(),'rx')
    plt.plot(ft_grid.detach().cpu().numpy(),torch.abs(fHat_perm).squeeze().detach().cpu().numpy(),'b+')
    plt.show()




    print(fHat)

    f_=nfft(k,fHat)

    print(torch.sum(torch.abs(f_-f)**2))
    
else:
    #fHat=torch.randn((k.shape[0],,dtype=torch.complex64,device=device)
    ft_grid=torch.arange(-N//2,N//2,dtype=torch.float32,device=device)
    fHat_dft=ndft_adjoint(k.squeeze(),f.squeeze(),ft_grid)
"""
