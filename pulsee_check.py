from pulsee.simulation import *

spin_par = {'quantum number' : 1.,
            'gamma/2pi' : 1.}
    
zeem_par = {'field magnitude' : 1.,
            'theta_z' : 0.,
            'phi_z' : 0.}
                
spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par=spin_par, quad_par=None, zeem_par=zeem_par, initial_state='canonical', temperature=1e-4)

plot_real_part_density_matrix(dm_0)

f, p = power_absorption_spectrum(spin, h_unperturbed, normalized=True)

plot_power_absorption_spectrum(f, p)

mode = pd.DataFrame([(2 * np.pi, 0.2, 0., np.pi/2, 0.)], 
                     columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])


dm_evolved = evolve(spin, h_unperturbed, dm_0, solver=magnus, \
                    mode=mode, pulse_time=1 / (4 * 0.1), \
                    picture = 'IP')
    
plot_real_part_density_matrix(dm_evolved)


t, fid = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=100, T2=10)

plot_real_part_FID_signal(t, fid)


f, ft = fourier_transform_signal(fid, t)
    
plot_fourier_transform(f, ft)

