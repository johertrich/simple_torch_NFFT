import torch
# Sehr einfache aber vektorisierte torch-implementierung der eindimensionalen NFFT.

def transposed_sparse_convolution(x,f,n,m,phi_conj,device):
    # x ist zweidimensional: erst batch-dimension, dann die Stützpunkte
    # f hat die gleiche Größe wie x oder ist broadcastable
    # n ist gerade
    # m ist gerade
    # phi_conj ist function handle
    l=torch.arange(0,2*m,device=device,dtype=torch.long).view(2*m,1,1)
    inds=(torch.ceil(n*x).long()-m)[None]+l
    increments=phi_conj(x[None,:,:]-inds/n)*f
    
    g_linear=torch.zeros((x.shape[0]*n,),device=device)
    inds=(inds+n//2)+x.shape[1]*torch.arange(0,x.shape[0],device=device,dtype=torch.long)[:,None] # +n//2 weil index shift von -n/2 bis n/2-1 zu 0 bis n-1, anderer Term für lineare Indizes
    g_linear.index_put_((inds.view(-1),),increments.view(-1),accumulate=True)
    return g_linear.view(x.shape[0],-1)

def adjoint_nfft(x,f,N,n,m,phi_conj,phi_hat,device):
    # x ist zweidimensional: erst batch-dimension, dann die Stützpunkte
    # f hat die gleiche Größe wie x oder ist broadcastable
    # n ist gerade
    # m ist gerade
    # phi_conj ist function handle
    # N ist gerade
    # phi_hat fängt mit negativen indizes an
    cut=(n-N)//2
    g=transposed_sparse_convolution(x,f,n,m,phi_conj,device)
    g=torch.fft.ifftshift(g)
    g_hat=torch.fft.fft(g_linear)
    g_hat=torch.fft.fftshift(g_hat)[:,cut:-cut]
    f_hat=g_hat/(n*phi_hat)
    # f_hat fängt mit negativen indizes an
    return f_hat

def sparse_convolution(x,g,n,m,M,phi,device):
    # x ist zweidimensional: erst batch-dimension, dann die Stützpunkte
    # g lebt auf [-1/2,1/2)
    # n ist gerade
    # m ist gerade
    # M ist beliebig
    # phi ist ein function handle
    l=torch.arange(0,2*m,device=device,dtype=torch.long).view(2*m,1,1)
    inds=(torch.ceil(n*x).long()-m)[None,:,:]+l
    increments=phi(x[None,:,:]-inds/n)
    inds=inds+n//2+x.shape[1]*torch.arange(0,x.shape[0],device=device,dtype=torch.long)[:,None] # +n//2 weil index shift von -n/2 bis n/2-1 zu 0 bis n-1, anderer Term für lineare Indizes
    g_l=g.view(-1)[inds].view(increments.shape)
    increments*=g_l
    f=torch.sum(increments,0)
    return f
    

def forward_nfft(x,f_hat,N,n,m,phi,phi_hat,device):
    # x ist zweidimensional: erst batch-dimension, dann die Stützpunkte
    # f_hat hat die größe (batch_size,N)
    # n ist gerade
    # m ist gerade
    # phi ist function handle
    # N ist gerade
    # phi_hat fängt mit negativen indizes an
    # f_hat fängt mit negativen indizes an
    g_hat=f_hat/(N*phi_hat)
    pad=torch.zeros((x.shape[0],(n-N)//2),device=device)
    g_hat=torch.fft.ifftshift(torch.cat((pad,g_hat,pad),1))
    g=torch.fft.fftshift(torch.fft.ifft(g_hat)) # damit g wieder auf [-1/2,1/2) lebt
    f=sparse_convolution(x,g,n,m,x.shape[1],phi)
    # f hat die gleiche Größe wie x
    return f
    
class KaiserBesselWindow(torch.nn.Module):
    def __init__(self,n,m,sigma,device='cuda' if torch.cuda.is_available() else 'cpu'):
        # n: Anzahl der oversampled Fourierkoeffizienten
        # m: Window size
        # sigma: oversampling --> Warum??? --> Damit die Fourietrreihe 0 wird außerhalb von -n/2,n/2-1 !!!
        super().__init__()
        self.n=n
        self.m=m
        self.sigma=sigma
        inds=torch.arange(-self.n//2,self.n//2-1)
        self.ft=self.Fourier_coefficients(inds)

    def forward(self,k):
        # undefined for abs(k)>self.m/self.n
        b=(2-1/self.sigma)*torch.pi
        out=b/torch.pi*torch.ones_like(k)
        arg=torch.sqrt(self.m**2-self.n**2*k**2)
        out[torch.abs(k)<self.m/self.n]=(torch.sinh(b*arg)/(arg*torch.pi))[torch.abs(k)<self.m/self.n] # das * pi ist in Gabis Buch nicht drin... Aber in NFFT.jl
        return out
        
        
    def Fourier_coefficients(self,inds):
        b=(2-1/self.sigma)*torch.pi
        return torch.special.bessel_j0(self.m*torch.sqrt(b**2-(2*torch.pi*inds/self.n)**2))
    
class NFFT(torch.nn.Module):
    def __init__(self,N,m,sigma,window=None,device='cuda' if torch.cuda.is_available() else 'cpu'):
        # N: Anzahl Fourierkoeffizienten
        # sigma: oversampling
        # m: Window size
        super().__init__()
        self.N=N # Anzahl Fourierkoeffizienten
        self.n=int(sigma*N)
        self.m=m
        self.device=device
        if window is None:
            self.window=KaiserBesselWindow(self.n,self.m,self.n/self.N,device=device)
        else:
            self.window=window

    def forward(self,x,f_hat): # TODO redefine autograd
        return forward_nfft(x,f_hat,self.N,self.n,self.m,self.window,self.window.ft,self.device)

    def adjoint(self,x,f): # TODO redefine autograd
        return adjoint_nfft(x,f,self.N,self.n,self.m,self.window,self.window.ft,self.device)


if __name__=='__main__':
    print(transposed_sparse_convolution(torch.rand(3,3),torch.rand(3,3),1000,4,lambda x:x,'cpu').shape)
