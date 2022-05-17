import numpy as np
from tqdm import tqdm

def diagonalize(matrix):
    einVals, einVects = np.linalg.eig(matrix)
    D = np.zeros((len(einVals),len(einVals)), dtype=np.complex128)
    Dexp = np.zeros((len(einVals),len(einVals)), dtype=np.complex128)
    
    for i in range(len(einVals)):
        D[i,i] = einVals[i]
        Dexp[i,i] = np.exp(einVals[i])
        
    U = np.stack([einVects[:, i] for i in range(len(einVects))], axis=-1)
    
    return U, D, Dexp

def diagonalEvolveSingleCore(H, rho0, times, e_ops = []):
    rhot = []
    rho0, H = (np.array(rho0), np.array(H)) 
    e_ops = [np.array(i) for i in e_ops]
    
    e_opst = []
    for t in tqdm(times):
        U1, D1, D1exp = diagonalize(-1j*H*t)
        U2, D2, D2exp = diagonalize(1j*H*t)
        
        rho = U1 @ D1exp @ np.linalg.inv(U1) @ rho0 @ U2 @ D2exp @ np.linalg.inv(U2) 
        rhot.append(rho)
        
        e_opst_temp = []
        for op in e_ops:
            e_opst_temp.append(np.trace(rho @ op))
            
        e_opst.append(e_opst_temp)
        
    return rhot, e_opst

def expm(matrix):
    matrix = np.array(matrix)
    U, D, Dexp = diagonalize(matrix)
    
    return U @ Dexp @ np.linalg.inv(U)

def apply_pulse(rho, duration, rot_axis):
    rot_axis = np.array(rot_axis)
    return expm(-1j*duration*rot_axis)@ np.array(rho) @ np.conj(expm(-1j*duration*rot_axis)).T

def FT_Stephen(signal, times):
    nt = len(times) #number of points
    
    # zero pad the ends to "interpolate" in frequnecy domain
    zn = 10; # power of zeros
    N_z = 2*(2**zn) + nt # number of elements in padded array
    zero_pad = np.zeros(N_z, dtype=np.complex128)
    
    M0_trunc_z = zero_pad
    num = 2**zn
    M0_trunc_z[num:(num + nt)] = signal
    
    # figure out the "frequency axis" after the FFT
    dt = times[2]-times[1]
    Fs = 1.0/(dt); # max frequency sampling
#     print(Fs)
    # axis goes from -Fs/2 to Fs/2, with N_z steps
    freq_ax = (  (np.linspace(0,N_z,N_z) - 1/2)/N_z - 1/2)*Fs;
    
    M_fft = np.fft.fftshift( np.fft.fft( np.fft.fftshift(M0_trunc_z) ) )
    
    return(freq_ax, M_fft)