from pydoc import doc
from typing import Type
import numpy as np
import pandas as pd
import math
from fractions import Fraction

from qutip import Options, mesolve, Qobj, tensor, expect, qeye
from qutip.parallel import parallel_map
from qutip.ipynbtools import parallel_map as ipynb_parallel_map
from qutip.solver import Result

import matplotlib.pylab as plt
from matplotlib import colors as clrs, docstring
from matplotlib import colorbar as clrbar
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import xticks, yticks
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from .operators import canonical_density_matrix, \
                       free_evolution, magnus_expansion_1st_term, \
                       magnus_expansion_2nd_term, magnus_expansion_3rd_term, \
                       changed_picture, exp_diagonalize

from .nuclear_spin import NuclearSpin, ManySpins

from .hamiltonians import h_zeeman, h_quadrupole, \
                         h_multiple_mode_pulse, \
                         h_changed_picture, \
                         h_j_coupling, \
                         h_CS_isotropic, h_D1, h_D2,\
                         h_HF_secular, h_j_secular, h_tensor_coupling,\
                         h_userDefined
    

def nuclear_system_setup(spin_par, 
                         quad_par=None,
                         zeem_par=None, 
                         j_matrix=None, 
                         cs_param=None, 
                         D1_param=None, 
                         D2_param=None, 
                         hf_param=None, 
                         h_tensor_inter=None,
                         j_sec_param=None,
                         h_userDef=None,
                         initial_state='canonical', 
                         temperature=1e-4):
    """
    Sets up the nuclear system under study, returning the objects representing
    the spin (either a single one or a multiple spins' system), the unperturbed
    Hamiltonian (made up of the Zeeman, quadrupolar and J-coupling
    contributions) and the initial state of the system.
    
    Parameters
    ----------
    - spin_par: dict / list of dict
  
      Map/list of maps containing information about the nuclear spin/spins under
      consideration. The keys and values required to each dictionary in this
      argument are shown in the table below.

      
      |           key          |         value        |
      |           ---          |         -----        |
      |    'quantum number'    |  half-integer float  |
      |       'gamma/2pi'      |         float        |
    
      The second item is the gyromagnetic ratio over 2 pi, measured in MHz/T.

      E.g., spin_par = {'quantum number': 1 / 2, 'gamma2/pi': 1}

    - quad_par: dict / list of dict
    
      Map/maps containing information about the quadrupolar interaction between
      the electric quadrupole moment and the EFG for each nucleus in the system.
      The keys and values required to each dictionary in this argument are shown
      in the table below:
                                           
      |           key           |       value        |
      |           ---           |       -----        |
      |   'coupling constant'   |       float        |
      |  'asymmetry parameter'  |   float in [0, 1]  |
      |        'alpha_q'        |       float        |
      |        'beta_q'         |       float        |
      |        'gamma_q'        |       float        |
    
      where 'coupling constant' stands for the product e2qQ in the expression of
      the quadrupole term of the Hamiltonian (to be provided in MHz), 'asymmetry
      parameter' refers to the same-named property of the EFG, and 'alpha_q',
      'beta_q' and 'gamma_q' are the Euler angles for the conversion from the LAB
      coordinate system to the system of the principal axes of the EFG tensor
      (PAS) (to be expressed in radians).
    
      When it is None, the quadrupolar interaction of all the spins in the
      system is not taken into account.  Default value is None.
    
    - zeem_par: dict
       
      Map containing information about the magnetic field interacting with the
      magnetic moment of each nucleus in the system. The keys and values
      required to this argument are shown in the table below:
      
      |         key         |       value      |
      |         ---         |       -----      |
      |      'theta_z'      |       float      |
      |       'phi_z'       |       float      |
      |  'field magnitude'  |  positive float  |

      where 'theta_z' and 'phi_z' are the polar and azimuthal angles of the
      magnetic field with respect to the LAB system (to be measured in radians),
      while field magnitude is to be expressed in tesla.
      
      When it is None, the Zeeman interaction is not taken into account.
      Default value is None.
    
    - j_matrix: np.ndarray
    
      Array whose elements represent the coefficients Jmn which determine the
      strength of the J-coupling between each pair of spins in the system. For
      the details on these data, see the description of the same-named parameter
      in the docstrings of the function h_j_coupling in the module
      Hamiltonians.py.
      
      When it is None, the J-coupling effects are not taken into account.      
      Default value is None.
    
    - cs_param: dict
    
     Map containing information about the chemical shift. The keys and values
     required to this argument are shown in the table below:
     
      |         key         |       value      |
      |         ---         |       -----      |
      |      'delta_iso'    |       float      |
      
      where delta_iso is the magnitude of the chemical shift in Hz.
      
      When it is None, the chemical shift is not taken into account.      
      Default value is None.
    
    - D1_param: dict
    
     Map containing information about the dipolar interaction in the secular
     approximation for homonuclear & heteronuclear spins. The keys and values
     required to this argument are shown in the table below:
     
      |         key         |       value      |
      |         ---         |       -----      |
      |        'b_d'        |       float      |
      |       'theta'       |       float      |
      
      where b_d is the magnitude of dipolar constant,  b_D\equiv
      \frac{\mu_0\gamma_1\gamma_2}{4\pi r^3_{21}}, and theta is the polar angle
      between the two spins (expressed in radians).
      
      When it is None, the dipolar interaction in the secular approximation  for
      homonuclear & heteronuclear spins is not taken into account. Default
      value is None.
      
    - D2_param: dict
    
     Map containing information about the dipolar interaction in the secular
     approximation for heteronuclear spins. The keys and values required to this
     argument are shown in the table below:
     
      |         key         |       value      |
      |         ---         |       -----      |
      |        'b_d'        |       float      |
      |       'theta'       |       float      |
      
      where b_d is the magnitude of dipolar constant,  b_D\equiv
      \frac{\mu_0\gamma_1\gamma_2}{4\pi r^3_{21}}, and theta is the polar angle
      between the two spins (expressed in radians).
      
      When it is None, the dipolar interaction in the secular approximation  for
      heteronuclear spins is not taken into account. Default value is None.
      
    - hf_param: dict
    
     Map containing information about the hyperfine interaction in the secular
     approximation between two spins. The keys and values required to this
     argument are shown in the table below:
     
      |         key         |       value      |
      |         ---         |       -----      |
      |         'A'         |       float      |
      |         'B'         |       float      |
      
      where A, B are constant of the hyperfine interaction inthe secular
      approximation, see paper. 
      
      When it is None, the hyperfine interaction in the secular approximation
      between two spins is not taken into account.      Default value is None.
      
    - j_sec_param: dict
    
     Map containing information about the J-couping in the secular
     approximation. The keys and values required to this argument are shown in
     the table below:
     
      |         key         |       value      |
      |         ---         |       -----      |
      |         'J'         |       float      |
      
      where J is the J-coupling constant in Hz.
      
      When it is None, the J-couping in the secular approximation is not taken
      into account. Default value is None.
    
    - h_tensor_inter: numpy.ndarray  or [numpy.ndarray, numpy.ndarray, ...]
      
      Rank-2 tensor describing a two-spin interaction of the form 
      $\mathbf{I}_1 J \mathbf{I}_2$ where $J$ is the tensor and $\mathbf{I}_i$
      are vector spin operators

      When it is None, the interaction is not taken into account. 

      Default value is None.

    - h_userDef: numpy.ndarray

      Square matrix array which will give the hamiltonian of the system, adding to
      previous terms (if any). When passing, must ensure compability with the rest
      of the system.

      Default value is None.

    - initial_state: either string or numpy.ndarray
  
      Specifies the state of the system at time t=0.
    
      If the keyword canonical is passed, the function will return a
      DensityMatrix object representing the state of thermal equilibrium at the
      temperature specified by the same-named argument.
      
      If a square complex array is passed, the function will return a
      DensityMatrix object directly initialised with it.
      
      Default value is 'canonical'.
    
    - temperature: float
  
      Temperature of the system (in kelvin).
    
      Default value is 1e-4.
    
    Returns
    -------
    - [0]: NuclearSpin / ManySpins
    
           The single spin/spin system subject to the NMR/NQR experiment.

    - [1]: List[Qobj]
  
           The unperturbed Hamiltonian, consisting of the Zeeman, quadrupolar
           and J-coupling terms (expressed in MHz).
           
    - [2]: DensityMatrix
  
           The density matrix representing the state of the system at time t=0,
           initialised according to initial_state.
    """

    if not isinstance(spin_par, list):
        spin_par = [spin_par]
    if quad_par is not None and not isinstance(quad_par, list):
        quad_par = [quad_par]
    
    if quad_par is not None and len(spin_par) != len(quad_par):
        raise IndexError("The number of passed sets of spin parameters must be" +\
         " equal to the number of the quadrupolar ones.")
    
    spins = []
    h_q = []
    h_z = []        
    
    for i in range(len(spin_par)):
        spins.append(NuclearSpin(spin_par[i]['quantum number'], \
                                  spin_par[i]['gamma/2pi']))
        
        if quad_par is not None:
            h_q.append(h_quadrupole(spins[i], quad_par[i]['coupling constant'], \
                                              quad_par[i]['asymmetry parameter'], \
                                              quad_par[i]['alpha_q'], \
                                              quad_par[i]['beta_q'], \
                                              quad_par[i]['gamma_q']))
        else:
            h_q.append(h_quadrupole(spins[i], 0., 0., 0., 0., 0.))
        
        if zeem_par is not None:
            h_z.append(h_zeeman(spins[i], zeem_par['theta_z'], \
                                          zeem_par['phi_z'], \
                                          zeem_par['field magnitude']))
        else:
            h_z.append(h_zeeman(spins[i], 0., 0., 0.))
        
        if cs_param is not None:
            if cs_param != 0.0:
                h_z.append(h_CS_isotropic(spins[i], cs_param['delta_iso'], 
                                          zeem_par['field magnitude']))
    
    spin_system = ManySpins(spins)
    
    h_unperturbed = []
    
    for i in range(spin_system.n_spins):
        h_i = h_q[i] + h_z[i]
        for j in range(i):
            h_i = tensor(Qobj(qeye(spin_system.spin[j].d)), h_i)
        for k in range(spin_system.n_spins)[i+1:]:
            h_i = tensor(h_i, Qobj(qeye(spin_system.spin[k].d)))
        h_unperturbed = h_unperturbed + [Qobj(h_i)]
    
    if j_matrix is not None:
        h_j = h_j_coupling(spin_system, j_matrix)
        h_unperturbed = h_unperturbed + [Qobj(h_j)]
        
    if D1_param is not None:
        if ((D1_param['b_D'] == 0.) and (D1_param['theta'] ==0.)):
            pass
        else:
            h_d1 = h_D1(spin_system, D1_param['b_D'], \
                                     D1_param['theta'])
            h_unperturbed = h_unperturbed + [Qobj(h_d1)]
    
    if D2_param is not None:
        if ((D2_param['b_D'] == 0.) and (D2_param['theta'] ==0.)):
            pass
        else:
            h_d2 = h_D2(spin_system, D2_param['b_D'], \
                                     D2_param['theta'])
            h_unperturbed = h_unperturbed + [Qobj(h_d2)]
    
    if hf_param is not None:
        if ((hf_param['A'] == 0.) and (hf_param['B'] ==0.)):
            pass
        else:
            h_hf = h_HF_secular(spin_system, hf_param['A'], \
                                     hf_param['B'])
            h_unperturbed = h_unperturbed + [Qobj(h_hf)]
        
    if j_sec_param is not None:
        if (j_sec_param['J']==0.0):
                pass
        else:
            h_j = h_j_secular(spin_system, j_sec_param['J'])
            h_unperturbed = h_unperturbed + [Qobj(h_j)]

    if h_tensor_inter is not None:
        if type(h_tensor_inter) != list:
            h_unperturbed += [Qobj(h_tensor_coupling(spin_system, 
                                                     h_tensor_inter))]
        else:
            for hyp_ten in h_tensor_inter:
                h_unperturbed += [Qobj(h_tensor_coupling(spin_system, hyp_ten) \
                                                )]

    if h_userDef is not None:
        h_unperturbed += (h_userDefined(h_userDef))
    if isinstance(initial_state, str) and initial_state == 'canonical':
        dm_initial = canonical_density_matrix(Qobj(sum(h_unperturbed)),
                                                temperature)
    else:
        dm_initial = Qobj(initial_state)
    
    if len(spins) == 1:
        return spins[0], h_unperturbed, dm_initial
    else:
        return spin_system, h_unperturbed, dm_initial


def power_absorption_spectrum(spin, h_unperturbed, normalized=True, dm_initial=None):
    """
    Computes the spectrum of power absorption of the system due to x-polarized
    monochromatic pulses.
    
    Parameters
    ----------
    - spin: NuclearSpin / ManySpins
  
            Single spin/spin system under study.
  
    - h_unperturbed: Operator
    
                     Unperturbed Hamiltonian of the system (in MHz).
    
    - normalized: bool
                
                  Specifies whether the difference between the states'
                  populations are to be taken into account in the calculation of
                  the line intensities. When normalized=True, they are not, when
                  normalized=False, the intensities are weighted by the
                  differences p(b)-p(a) just like in the formula above.
                  
                  Default value is True.
  
    - dm_initial: DensityMatrix or None
  
                  Density matrix of the system at time t=0, just before the
                  application of the pulse.
                  
                  The default value is None, and it should be left so only when
                  normalized=True, since the initial density matrix is not
                  needed.
                  
    Action
    ------
    Diagonalises h_unperturbed and computes the frequencies of transitions
    between its eigenstates.
    
    Then, it determines the relative proportions of the power absorption for
    different lines applying the formula derived from Fermi golden rule (taking
    or not taking into account the states' populations, according to the value
    of normalized).
    
    Returns:
    -------
    [0]: The list of the frequencies of transition between the eigenstates of
         h_unperturbed (in MHz);
    
    [1]: The list of the corresponding intensities (in arbitrary units).
    """
    h_unperturbed = Qobj(np.sum(h_unperturbed), dims=h_unperturbed[0].dims)
    energies, o_change_of_basis = h_unperturbed.eigenstates()
    
    transition_frequency = []
    
    transition_intensity = []
    
    d = h_unperturbed.dims[0][0] # assume that this Hamiltonian is a rank-1 tensor
    
    # Operator of the magnetic moment of the spin system
    if isinstance(spin,  ManySpins):
        magnetic_moment = Qobj(qeye(spin.d))*0
        for i in range(spin.n_spins):
            mm_i = spin.spin[i].gyro_ratio_over_2pi*spin.spin[i].I['x']
            for j in range(i):
                mm_i = tensor(Qobj(qeye(spin.spin[j].d)), mm_i)
            for k in range(spin.n_spins)[i+1:]:
                mm_i = tensor(mm_i, Qobj(qeye(spin.spin[k].d)))
            magnetic_moment = magnetic_moment + mm_i
    else:
        magnetic_moment = spin.gyro_ratio_over_2pi*spin.I['x']
    
    mm_in_basis_of_eigenstates = magnetic_moment.transform(o_change_of_basis)
    
    for i in range(d):
        for j in range(d):
            if i < j:
                nu = np.absolute(energies[j] - energies[i])
                transition_frequency.append(nu)
                
                intensity_nu = nu*\
                    (np.absolute(mm_in_basis_of_eigenstates[j, i]))**2
                
                if not normalized:
                    p_i = dm_initial[i, i]
                    p_j = dm_initial[j, j]
                    intensity_nu = np.absolute(p_i-p_j)*intensity_nu
                    
                transition_intensity.append(intensity_nu)
            else:
                pass
    
    return transition_frequency, transition_intensity


def plot_power_absorption_spectrum(frequencies, intensities, show=True, 
                                   fig_dpi = 400, save=False, name='PowerAbsorptionSpectrum', destination=''):
    """
    Plots the power absorption intensities as a function of the corresponding
    frequencies.
    
    Parameters
    ----------
    - frequencies: array-like
                
                   Frequencies of the transitions (in MHz).
    
    - intensities: array-like
    
                   Intensities of the transitions (in a.u.).
    
    - show: bool
  
            When False, the graph constructed by the function will not be
            displayed.
            
            Default value is True.

    - fig_dpi: int

            Image quality of the figure when showing and saving. Useful for
            publications. Default set to very high value.
            
    - save: bool
  
            When False, the plotted graph will not be saved on disk. When True,
            it will be saved with the name passed as name and in the directory
            passed as destination.
            
            Default value is False.
    
    - name: string
  
            Name with which the graph will be saved.
    
            Default value is 'PowerAbsorptionSpectrum'.
    
    - destination: string
  
                   Path of the directory where the graph will be saved (starting
                   from the current directory). The name of the directory must
                   be terminated with a slash /.
                   
                   Default value is the empty string (current directory).
                   
    Action
    ------
    If show=True, generates a graph with the frequencies of transition on the x
    axis and the corresponding intensities on the y axis.
    
    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure
    built up by the function.  
    """
    fig = plt.figure()
    
    plt.vlines(frequencies, 0, intensities, colors='b')
    
    plt.xlabel("\N{GREEK SMALL LETTER NU} (MHz)")    
    plt.ylabel("Power absorption (a. u.)")
    
    if save: plt.savefig(destination + name, dpi= fig_dpi)
    
    if show: plt.show()
    
    return fig


def evolve(spin, h_unperturbed, dm_initial, solver=mesolve, mode=None, 
            pulse_time=0, picture='RRF', RRF_par={'nu_RRF': 0, 'theta_RRF': 0, 
            'phi_RRF': 0}, n_points=30, order=None, opts=None):
    
    """
    Simulates the evolution of the density matrix of a nuclear spin under the
    action of an electromagnetic pulse in a NMR/NQR experiment.
    
    Parameters
    ----------
    - `spin`: NuclearSpin
            Spin under study.
    
    - `h_unperturbed`: List[Qobj or (Qobj, function)]
                     Hamiltonian of the nucleus at equilibrium (in MHz).
    
    - `dm_initial`: Qobj
                  Density matrix of the system at time t=0, just before the
                  application of the pulse.
                  
    - `solver`: function (Qobj, Qobj, ndarray, **kwargs) -> qutip.solver.Result
                OR 
                String
                Solution method to be used when calculating time evolution of
                state. If string, must be either `mesolve` or `magnus.`

    - `mode`: pandas.DataFrame
            Table of the parameters of each electromagnetic mode in the pulse.
            It is organised according to the following template:

    | index |  'frequency'  |  'amplitude'  |  'phase'  |  'theta_p'  |  'phi_p'  |
    | ----- | ------------- | ------------- | --------- | ----------- | --------- |
    |       |     (MHz)     |      (T)      |   (rad)   |    (rad)    |   (rad)   |
    |   0   |    omega_0    |      B_0      |  phase_0  |   theta_0   |   phi_0   |
    |   1   |    omega_1    |      B_1      |  phase_1  |   theta_1   |   phi_1   |
    |  ...  |      ...      |      ...      |    ...    |     ...     |    ...    |
    |   N   |    omega_N    |      B_N      |  phase_N  |   theta_N   |   phi_N   |

            where the meaning of each column is analogous to the corresponding
            parameters in h_single_mode_pulse.
            
            When it is None, the evolution of the system is performed for the
            given time duration without any applied pulse.
            
            The default value is None.
    
    - `pulse_time`: float
                  Duration of the pulse of radiation sent onto the sample (in
                  microseconds).
                  
                  The default value is 0.
    
    - `picture`: string
               Sets the dynamical picture where the density matrix of the system
               is evolved for the `magnus` solver. May take the values:
               
        1. IP', which sets the interaction picture;
        2.'RRF' (or anything else), which sets the picture corresponding to a
        rotating reference frame whose features are specified in argument
        RRF_par.
        
               The default value is RRF.
               
               The choice of picture has no effect on solvers other than `magnus`.
    
    - `RRF_par`: dict
               Specifies the properties of the rotating reference frame where
               evolution is carried out when picture='RRF'. The details on the
               organisation of these data can be found in the description of
               function RRF_Operator.  By default, all the values in this map
               are set to 0 (RRF equivalent to the LAB frame).
               
    - `n_points`: float
                Counts the number of points in which the time interval [0,
                pulse_time] is sampled in the discrete approximation of the
                time-dependent Hamiltonian of the system.  Default value is 10.

    - `order`: integer 
               The order of the simulation method to use. For `magnus` must be <= 3. 
               Defaults to 2 for `magnus` and 12 for `mesolve` and any other solver.
  
    Action
    ------
    If
    - pulse_time is equal to 0;
    - dm_initial is very close to the identity (with an error margin of 1e-10
        for each element)
    
    the function returns dm_initial without performing any evolution.
  
    Otherwise, evolution is carried out in the picture determined by the
    same-named parameter. The evolution operator is built up appealing to the
    Magnus expansion of the full Hamiltonian of the system (truncated to the
    order specified by the same-named argument).
    
    Returns
    -------
    The DensityMatrix object representing the state of the system (in the
    Schroedinger picture) evolved through a time pulse_time under the action of
    the specified pulse.  
    """
    
    if pulse_time == 0 or np.all(np.absolute((dm_initial.full() \
                                    - np.eye(spin.d))) < 1e-10):
        return dm_initial
    
        
    if mode is None:
        mode = pd.DataFrame([(0., 0., 0., 0., 0)], 
                            columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
        

    times = np.linspace(0, pulse_time, num=max(2, int(n_points * pulse_time)))

    # Split into operator and time-dependent coefficient as per QuTiP scheme.
    h_perturbation = h_multiple_mode_pulse(spin, mode, t=0, factor_t_dependence=True)

    # match tolerance to operators.posititivity tolerance.
    if order is None and (solver == magnus or solver == 'magnus'):
        order = 3
    elif order is None: 
        order = 12

    if opts is None: 
        # opts = Options(atol=1e-14, rtol=1e-14, rhs_reuse=False, order=order)
        opts = Options(atol=1e-14, rtol=1e-14, rhs_reuse=False, nsteps=10000)
    else: 
        opts.order = order
        
    # h_unperturbed and h_perturbation are both lists. If H = H0 + H1*f1(t) + H2*f1(t) + ..., then
    # h is of the form [H0, [H1, f1(t)], [H2, f2(t)], ...] (refer to QuTiP's mesolve documentation for further detail)
    h = h_unperturbed + h_perturbation
    # result = None
    if solver == magnus or solver == 'magnus': 
        o_change_of_picture = None
        if picture == 'IP':
            o_change_of_picture = Qobj(sum(h_unperturbed), dims=h_unperturbed[0].dims)
        else:
            o_change_of_picture = RRF_operator(spin, RRF_par)
        h_total = Qobj(sum(h_unperturbed), dims=h_unperturbed[0].dims)
        h_new_picture = []
        for t in times: 
            h_new_picture.append(h_changed_picture(spin, mode, h_total, o_change_of_picture, t))

        result = magnus(h_new_picture, Qobj(dm_initial), times, options=opts)
        dm_evolved = changed_picture(result.states[-1], o_change_of_picture, times[-1] - times[0], invert=True)
        return dm_evolved

    elif solver == mesolve or solver == 'mesolve':
        scaled_h = [] 

        # Magnus expansion solver includes 2 pi factor in exponentiations; 
        # scale Hamiltonians by this factor for `mesolve` for consistency.
        for h_term in h: 
            if type(h_term) == list or type(h_term) == tuple: # of the form: (Hm, fm(t))
                scaled_h.append([h_term[0] * 2 * np.pi, h_term[1]])
            else: # of the form: H0
                scaled_h.append(2 * np.pi * h_term) 

        result = mesolve(scaled_h, Qobj(dm_initial), times, options=opts, progress_bar=True)
        final_state = result.states[-1]
        return final_state # return last time step of density matrix evolution.

    elif type(solver) == str:
        raise ValueError('Invalid solver: ' + solver)

    else: 
        result = solver(h, Qobj(dm_initial), times, options=opts)
        final_state = result.states[-1]
        return final_state # return last time step of density matrix evolution.

# Operator which generates a change of picture equivalent to moving to the rotating reference frame
# (RRF)
def RRF_operator(spin, RRF_par):
    """
    Returns the operator for the change of picture equivalent to moving to the RRF.
  
    Parameters
    ----------
    - spin: NuclearSpin
            
            Spin under study.
  
    - RRF_par: dict
                
               Specifies the properties of the rotating reference frame. The
               keys and values required to this argument are shown in the table
               below:
               
               |      key      |  value  |
               |      ---      |  -----  |
               |    'nu_RRF'   |  float  |
               |  'theta_RRF'  |  float  |
               |   'phi_RRF'   |  float  |
    
    where 'nu_RRF' is the frequency of rotation of the RRF (in MHz), while
    'theta_RRF' and 'phi_RRF' are the polar and azimuthal angles of the normal
    to the plane of rotation in the LAB frame (in radians).
    
    Returns
    -------
    An Observable object representing the operator which generates the change to the RRF picture.
    """
    nu = RRF_par['nu_RRF']
    theta = RRF_par['theta_RRF']
    phi = RRF_par['phi_RRF']
    RRF_o = nu*(spin.I['z']*math.cos(theta) + \
                spin.I['x']*math.sin(theta)*math.cos(phi) + \
                spin.I['y']*math.sin(theta)*math.sin(phi))
    return Qobj(RRF_o)


def plot_real_part_density_matrix(dm, many_spin_indexing = None, 
                                  show=True, fig_dpi = 400, 
                                  save=False, xmin=None, xmax=None, 
                                  ymin=None, ymax=None, show_legend = True, 
                                  name='RealPartDensityMatrix', destination=''):
    """
    Generates a 3D histogram displaying the real part of the elements of the
    passed density matrix.
    
    Parameters
    ----------
    - dm: DensityMatrix/numpy array as a square matrix
  
          Density matrix to be plotted.
          
    - many_spin_indexing: either None or list
  
                          If it is different from None, the density matrix dm is
                          interpreted as the state of a many spins' system, and
                          this parameter provides the list of the dimensions of
                          the subspaces of the full Hilbert space related to the
                          individual nuclei of the system. The ordering of the
                          elements of many_spin_indexing should match that of
                          the single spins' density matrices in their tensor
                          product resulting in dm. Default value is None.
    
    - show: bool
  
            When False, the graph constructed by the function will not be
            displayed.
            
            Default value is True.

    - fig_dpi: int

            Image quality of the figure when showing and saving. Useful for
            publications. Default set to very high value.
            
    - save: bool
  
            When False, the plotted graph will not be saved on disk. When True,
            it will be saved with the name passed as name and in the directory
            passed as destination.
            
            Default value is False.
    
    - xmin, xmax, ymin, ymax : Float

            Set axis limits of the graph.
    
    - name: string
  
            Name with which the graph will be saved.
    
            Default value is 'RealPartDensityMatrix'.
    
    - destination: string
  
                   Path of the directory where the graph will be saved (starting
                   from the current directory). The name of the directory must
                   be terminated with a slash /.
                   
                   Default value is the empty string (current directory).

    Action
    ------
    If show=True, draws a histogram on a 2-dimensional grid representing the
    density matrix, with the real part of each element indicated along the z
    axis. Blue bars indicate the positive matrix elements, red bars indicate the
    negative elements in absolute value.
    
    Returns
    -------
    An object of the class matplotlib.figure.Figure and an object of the class
    matplotlib.axis.Axis representing the figure built up by the function.
    
    """    
    real_part = np.vectorize(np.real)
    if isinstance(dm, Qobj):
        data_array = real_part(dm)
    else:
        data_array = real_part(dm)
    
    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data
    # You can think of this as the floor of the plot
    x_data, y_data = np.meshgrid(np.arange(data_array.shape[1])+0.25,
                                 np.arange(data_array.shape[0])+0.25)
    
    # Set width of the vertical bars
    dx = dy = 0.5

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays: x_data, y_data, z_data. The following
    # call boils down to picking one entry from each array and plotting a bar from (x_data[i],
    # y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
        
    bar_color = np.zeros(len(z_data), dtype=object)
        
    for i in range(len(z_data)):
        if z_data[i]<-1e-10:
            bar_color[i] = 'tab:red'
        else:
            bar_color[i] = 'tab:blue'
                            
    ax.bar3d(x_data,
             y_data,
             np.zeros(len(z_data)),
             dx, dy, np.absolute(z_data), color=bar_color)
    
    d = data_array.shape[0]
    tick_label = []
    
    if many_spin_indexing is None:
        for i in range(d):
            tick_label.append('|' + str(Fraction((d-1)/2-i)) + '>')

    else:        
        d_sub = many_spin_indexing
        n_sub = len(d_sub)
        m_dict = []
        
        for i in range(n_sub):
            m_dict.append({})
            for j in range(d_sub[i]):
                m_dict[i][j] = str(Fraction((d_sub[i]-1)/2-j))
        
        for i in range(d):
            tick_label.append('>')

        for i in range(n_sub)[::-1]:
            d_downhill = int(np.prod(d_sub[i+1:n_sub]))
            d_uphill = int(np.prod(d_sub[0:i]))
            
            for j in range(d_uphill):
                for k in range(d_sub[i]):
                    for l in range(d_downhill):
                        tick_label[j*d_sub[i]*d_downhill + k*d_downhill + l] = m_dict[i][k] + \
                            ', ' + tick_label[j*d_sub[i]*d_downhill + k*d_downhill + l]
        
        for i in range(d):
            tick_label[i] = '|' + tick_label[i]
            
        ax.tick_params(axis='both', which='major', labelsize=6)
            
    xticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), tick_label)
    yticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), tick_label)
    
    ax.set_zlabel("Re(\N{GREEK SMALL LETTER RHO})")
    
    
    legend_elements = [Patch(facecolor='tab:blue', label='<m|\N{GREEK SMALL LETTER RHO}|m> > 0'), \
                       Patch(facecolor='tab:red', label='<m|\N{GREEK SMALL LETTER RHO}|m> < 0')]
    if(show_legend):
        ax.legend(handles=legend_elements, loc='upper left')
    
    if (xmin is not None) and (xmax is not None):
        plt.xlim(xmin, xmax)
    if (ymin is not None) and (ymax is not None):
        plt.ylim(ymin, ymax)
    
    if save:
        plt.savefig(destination + name, dpi=fig_dpi)
    
    if show:
        plt.show()
        
    return fig, ax

def complex_phase_cmap():
    """
    Create a cyclic colormap for representing the phase of complex variables

    From QuTiP 4.0
    https://qutip.org.

    Returns
    -------
    cmap :
        A matplotlib linear segmented colormap.
    """
    cdict = {'blue': ((0.00, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 1.0, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
             'green': ((0.00, 0.0, 0.0),
                       (0.25, 1.0, 1.0),
                       (0.50, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 0.0)),
             'red': ((0.00, 1.0, 1.0),
                     (0.25, 0.5, 0.5),
                     (0.50, 0.0, 0.0),
                     (0.75, 0.0, 0.0),
                     (1.00, 1.0, 1.0))}

    cmap = clrs.LinearSegmentedColormap('phase_colormap', cdict, 256)

    return cmap

def plot_complex_density_matrix(dm, many_spin_indexing = None, show=True, phase_limits=None, phi_label = r'$\phi$', show_legend = True, fig_dpi = 400, save=False, name='ComplexDensityMatrix', destination='', figsize=None, labelsize=6):
    """
    Generates a 3D histogram displaying the amplitude and phase (with colors)
    of the elements of the passed density matrix.

    Inspired by QuTiP 4.0's matrix_histogram_complex function.
    https://qutip.org

    Parameters
    ----------
    - dm: DensityMatrix/numpy array as a square matrix

          Density matrix to be plotted.

    - many_spin_indexing: either None or list

                          If it is different from None, the density matrix dm is
                          interpreted as the state of a many spins' system, and
                          this parameter provides the list of the dimensions of
                          the subspaces of the full Hilbert space related to the
                          individual nuclei of the system. The ordering of the
                          elements of many_spin_indexing should match that of
                          the single spins' density matrices in their tensor
                          product resulting in dm.  Default value is None.

    - show: bool

            When False, the graph constructed by the function will not be
            displayed.
            
            Default value is True.

    - phase_limits: list/array with two float numbers

        The phase-axis (colorbar) limits [min, max] (optional)

    - phi_label: str

            Label for the legend for the angle of the complex number.

    - show_legend: bool

            Show the legend for the complex angle.

    - fig_dpi: int

            Image quality of the figure when showing and saving. Useful for
            publications. Default set to very high value.
            
    - save: bool

            When False, the plotted graph will not be saved on disk. When True,
            it will be saved with the name passed as name and in the directory
            passed as destination.
            
            Default value is False.

    - name: string

            Name with which the graph will be saved.

            Default value is 'RealPartDensityMatrix'.

    - destination: string

                   Path of the directory where the graph will be saved (starting
                   from the current directory). The name of the directory must
                   be terminated with a slash /.
                   
                   Default value is the empty string (current directory).

    Action
    ------
    If show=True, draws a histogram on a 2-dimensional grid representing the
    density matrix, with phase sentivit data.
    
    Returns
    -------
    An object of the class matplotlib.figure.Figure and an object of the class 
e   matplotlib.axis.Axis representing the figure built up by the function.
    
    """
    if isinstance(dm, Qobj):
        data_array = np.array(dm)
    else:
        data_array = dm

    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    if(figsize):
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data
    # You can think of this as the floor of the plot
    x_data, y_data = np.meshgrid(np.arange(data_array.shape[1])+0.25,
                                 np.arange(data_array.shape[0])+0.25)

    # Set width of the vertical bars
    dx = dy = 0.5

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays: x_data, y_data, z_data. The following
    # call boils down to picking one entry from each array and plotting a bar from (x_data[i],
    # y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()

    if phase_limits:  # check that limits is a list type
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -np.pi
        phase_max = np.pi

    norm = clrs.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()

    colors = cmap(norm(np.angle(z_data)))

    ax.bar3d(x_data,
             y_data,
             np.zeros(len(z_data)),
             dx, dy, np.absolute(z_data), color=colors)

    d = data_array.shape[0]
    tick_label = []

    if many_spin_indexing is None:
        for i in range(d):
            tick_label.append('|' + str(Fraction((d-1)/2-i)) + '>')

    else:
        d_sub = many_spin_indexing
        n_sub = len(d_sub)
        m_dict = []

        for i in range(n_sub):
            m_dict.append({})
            for j in range(d_sub[i]):
                m_dict[i][j] = str(Fraction((d_sub[i]-1)/2-j))

        for i in range(d):
            tick_label.append('>')

        for i in range(n_sub)[::-1]:
            d_downhill = int(np.prod(d_sub[i+1:n_sub]))
            d_uphill = int(np.prod(d_sub[0:i]))

            for j in range(d_uphill):
                for k in range(d_sub[i]):
                    for l in range(d_downhill):
                        tick_label[j*d_sub[i]*d_downhill + k*d_downhill + l] = m_dict[i][k] + \
                            ', ' + tick_label[j*d_sub[i]*d_downhill + k*d_downhill + l]

        for i in range(d):
            tick_label[i] = '|' + tick_label[i]

        ax.tick_params(axis='both', which='major', labelsize=labelsize)

    xticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), tick_label)
    yticks(np.arange(start=1., stop=data_array.shape[0]+1.), tick_label)

    if(show_legend):
        cax, kw = clrbar.make_axes(ax, location = 'right', shrink=.75, pad=.06)
        cb = clrbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.set_ticklabels(
            (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        cb.set_label(phi_label)

    if save:
        plt.savefig(destination + name, dpi=fig_dpi)

    if show:
        plt.show()

    return fig, ax



def FID_signal(spin, h_unperturbed, dm, acquisition_time, T2=100, theta=0, 
               phi=0, reference_frequency=0, n_points=30):
    """ 
    Simulates the free induction decay signal (FID) measured after the shut-off
    of the electromagnetic pulse, once the evolved density matrix of the system,
    the time interval of acquisition, the relaxation time T2 and the direction
    of the detection coils are given.
    
    Parameters
    ----------
    - spin: NuclearSpin
    
            Spin under study.
    
    - h_unperturbed: Operator
  
                     Unperturbed Hamiltonian of the system (in MHz).
    
    - dm: DensityMatrix
  
          Density matrix representing the state of the system at the beginning
          of the acquisition of the signal.
          
    - acquisition_time: float
  
                        Duration of the acquisition of the signal, expressed in
                        microseconds.
                        
    - T2: iterable[float or function with signature (float) -> float] or float
            or function with signature (float) -> float
    
          If float, characteristic time of relaxation of the component of the
          magnetization on the plane of detection vanishing, i.e., T2. It is
          measured in
          microseconds.

          If function, the decay envelope. 

          If iterable, total decay envelope will be product of decays in list.

          Default value is 100 (microseconds).
    
    - theta, phi: float
  
                  Polar and azimuthal angles which specify the normal to the
                  plane of detection of the FID signal (in radians).
                  
                  Default values are theta=0, phi=0.
                  
    - reference_frequency: float
    
                           Specifies the frequency of rotation of the
                           measurement apparatus with respect to the LAB system.
                           Default value is 0.
    
    - n_points: float
  
                Counts (approximatively) the number of points per microsecond in
                which the time interval [0, acquisition_time] is sampled for the
                generation of the FID signal.
                
                Default value is 100.
    
    Action
    ------
    Samples the time interval [0, acquisition_time] with n_points points per
    microsecond.
    
    The FID signal is simulated under the assumption that it is directly related
    to the time-dependent component of the magnetization on the plane specified
    by (theta, phi) of the LAB system.
    
    Returns
    -------
    [0]: numpy.ndarray
  
         Vector of equally spaced sampled instants of time in the interval [0,
         acquisition_time] (in microseconds).
         
    [1]: numpy.ndarray
  
         FID signal evaluated at the discrete times reported in the first output
         (in arbitrary units).  
    """
    times = np.linspace(start=0, stop=acquisition_time, num=int(acquisition_time*n_points))
    
    FID = []

    decay_envelopes = []
    try:
        for d in T2: 
            if not callable(d):
                decay_envelopes.append(lambda t: np.exp(-t / d))
            else: 
                decay_envelopes.append(d)
    except TypeError:
        if not callable(T2): 
            decay_envelopes.append(lambda t: np.exp(-t / T2))
        else: 
            decay_envelopes.append(T2)

    # Computes the FID assuming that the detection coils record the time-dependence of the
    # magnetization on the plane perpendicular to (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    Iz = spin.I['z']
    Iy = spin.I['y']
    I_plus_rotated = (1j * phi * Iz).expm() * (1j * theta * Iy).expm() \
            * spin.I['+'] * (-1j * theta * Iy).expm() * (-1j * phi * Iz).expm()
    for t in times:

        # Obtain total decay envelope at that time.
        env = 1
        for dec in decay_envelopes: 
            env *= dec(t) # Different name to avoid bizarre variable scope bug
                          # (can't have same name as iteration var in line 1117.)

        dm_t = free_evolution(dm, Qobj(sum(h_unperturbed)), t)
        FID.append((Qobj(np.array(dm_t)) * Qobj(np.array(I_plus_rotated)) * env * np.exp(-1j * 2 * np.pi * reference_frequency * t)).tr())
    
    return times, np.array(FID)


def plot_real_part_FID_signal(times, FID, show=True, fig_dpi = 400, save=False, name='FIDSignal', destination=''):
    """
    Plots the real part of the FID signal as a function of time.
  
    Parameters
    ----------
    - time: array-like
  
            Sampled instants of time (in microseconds).
    
    - FID: array-like
  
           Sampled FID values (in arbitrary units).
    
    - show: bool
  
            When False, the graph constructed by the function will not be
            displayed.
            
            Default value is True.

    - fig_dpi: int

            Image quality of the figure when showing and saving. Useful for
            publications. Default set to very high value.

    - save: bool
  
            When False, the plotted graph will not be saved on disk. When True,
            it will be saved with the name passed as name and in the directory
            passed as destination.
            
            Default value is False.
    
    - name: string
  
            Name with which the graph will be saved.
    
            Default value is 'FIDSignal'.
    
    - destination: string
  
                   Path of the directory where the graph will be saved (starting
                   from the current directory). The name of the directory must
                   be terminated with a slash /.
                   
                   Default value is the empty string (current directory).
    
    Action
    ------ 
    If show=True, generates a plot of the FID signal as a function of time.
      
    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure
    built up by the function.
    
    """
    fig = plt.figure()
    
    plt.plot(times, np.real(FID), label='Real part')
        
    plt.xlabel("time (\N{GREEK SMALL LETTER MU}s)")    
    plt.ylabel("Re(FID) (a. u.)")
    
    if save: plt.savefig(destination + name, dpi=fig_dpi)
    
    if show: plt.show()
    
    return fig

 

def fourier_transform_signal(signal, times, abs=False, padding=None):
    """
    Computes the Fourier transform of the passed time-dependent signal.

    Parameters
    ----------
     - `signal`: array-like:
              Sampled signal to be transformed in the frequency domain (in a.u.).
    - `times`: array-like
             Sampled time domain (in microseconds).
    - `abs`: Boolean 
             Whether to return the absolute value of the computer Fourier transform. 
    - `padding`: Integer
             Amount of zero-padding to add to signal in the power of zeroes.
    
    Returns
    -------
    The frequency and fourier-transformed signal as a tuple (f, ft)
    """
    if padding is not None: 
        # This code by Stephen Carr
        nt = len(times) #number of points
        
        # zero pad the ends to "interpolate" in frequency domain
        zn = padding # power of zeros
        N_z = 2 * (2 ** zn) + nt # number of elements in padded array
        zero_pad = np.zeros(N_z, dtype=complex)
        
        M0_trunc_z = zero_pad
        num = 2 ** zn
        M0_trunc_z[num:(num + nt)] = signal
        
        # figure out the "frequency axis" after the FFT
        dt = times[2] - times[1]
        Fs = 1.0 / dt # max frequency sampling

        # axis goes from - Fs / 2 to Fs / 2, with N_z steps
        freq_ax = ((np.linspace(0, N_z, N_z) - 1/2) / N_z - 1/2) * Fs
        
        M_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(M0_trunc_z)))
        if abs:
            M_fft = np.abs(M_fft)
        return freq_ax, M_fft

    ft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(times), (times[-1] - times[0]) / len(times))
    if abs: 
        ft = np.abs(ft)
    return freq, ft


def legacy_fourier_transform_signal(times, signal, frequency_start, 
                                    frequency_stop, opposite_frequency=False):
    """
    Deprecated since QuTiP integration; see simulation.fourier_transform_signal.

    Computes the Fourier transform of the passed time-dependent signal over the
    frequency interval [frequency_start, frequency_stop]. The implemented
    Fourier transform operation is
    
    where S is the original signal and T is its duration. In order to have a
    reliable Fourier transform, the signal should be very small beyond time T.
    
    Parameters
    ----------
    - times: array-like
  
             Sampled time domain (in microseconds).
             
    - signal: array-like
  
              Sampled signal to be transformed in the frequency domain (in a.u.).

    - frequency_start, frequency_stop: float
  
                                       Left and right bounds of the frequency
                                       interval of interest, respectively (in
                                       MHz).
                                       
    - opposite_frequency: bool
  
                          When it is True, the function computes the Fourier
                          spectrum of the signal in both the intervals
                          frequency_start -> frequency_stop and -frequency_start
                          -> -frequency_stop (the arrow specifies the ordering
                          of the Fourier transform's values when they are stored
                          in the arrays to be returned).
                          
    Returns
    -------
    [0]: numpy.ndarray
  
         Vector of 1000 equally spaced sampled values of frequency in the
         interval [frequency_start, frequency_stop] (in MHz).
         
    [1]: numpy.ndarray
  
         Fourier transform of the signal evaluated at the discrete frequencies
         reported in the first output (in a.u.).
    
    If opposite_frequency=True, the function also returns:
    
    [2]: numpy.ndarray
  
         Fourier transform of the signal evaluated at the discrete frequencies
         reported in the first output changed by sign (in a.u.).  
    """
    dt = times[1]-times[0]
    
    frequencies = np.linspace(start=frequency_start, stop=frequency_stop, num=1000)
    
    fourier = [[], []]
    
    if opposite_frequency == False:
        sign_options = 1
    else:
        sign_options = 2
    
    for s in range(sign_options):
        for nu in frequencies:
            integral = np.zeros(sign_options, dtype=complex)
            for t in range(len(times)):
                integral[s] = integral[s] + np.exp(-1j*2*np.pi*(1-2*s)*nu*times[t])*signal[t]*dt
            fourier[s].append(integral[s])
        
    if opposite_frequency == False:
        return frequencies, np.array(fourier[0])
    else:
        return frequencies, np.array(fourier[0]), np.array(fourier[1])


# Finds out the phase responsible for the displacement of the real and imaginary parts of the Fourier
# spectrum of the FID with respect to the ideal absorptive/dispersive lorentzian shapes
def fourier_phase_shift(frequencies, fourier, fourier_neg=None, peak_frequency=0, int_domain_width=.5):
    """
    Computes the phase factor which must multiply the Fourier spectrum
    (`fourier`) in order to have the real and imaginary part of the adjusted
    spectrum showing the conventional dispersive/absorptive shapes at the peak
    specified by `peak_frequency`.
    
    Parameters
    ----------
    - frequencies: array-like
  
                   Sampled values of frequency (in MHz). 
  
    - fourier: array-like
  
               Values of the Fourier transform of the signal (in a.u.) sampled
               at the frequencies passed as the first argument.
               
    - fourier_neg: array-like
  
                   Values of the Fourier transform of the signal (in a.u.)
                   sampled at the opposite of the frequencies passed as the
                   first argument. When fourier_neg is passed, it is possible to
                   specify a peak_frequency located in the range frequencies
                   changed by sign.
                   
                   Default value is None.
  
    - peak_frequency: float
  
                      Position of the peak of interest in the Fourier spectrum.
    
                      Default value is 0.

    - int_domain_width: float
  
                        Width of the domain (centered at peak_frequency) where
                        the real and imaginary parts of the Fourier spectrum
                        will be integrated.
                        
                        Default value is .5.

    Action
    ------  
    The function integrates both the real and the imaginary parts of the
    spectrum over an interval of frequencies centered at peak_frequency whose
    width is given by int_domain_width. Then, it computes the phase shift.
    
    Returns
    -------
    A float representing the desired phase shift (in radians).
    """
    
    if fourier_neg is not None:
        fourier = np.concatenate((fourier, fourier_neg))
        frequencies = np.concatenate((frequencies, -frequencies))
    
    integration_domain = np.nonzero(np.isclose(frequencies, peak_frequency, atol=int_domain_width/2))[0]
    
    int_real_fourier = 0
    int_imag_fourier = 0
    
    for i in integration_domain:
        int_real_fourier = int_real_fourier + np.real(fourier[i])
        int_imag_fourier = int_imag_fourier + np.imag(fourier[i])
        
    if np.absolute(int_real_fourier) < 1e-10 :
        if int_imag_fourier > 0:
            return 0
        else:
            return np.pi
    
    atan = math.atan(- int_imag_fourier / int_real_fourier)
    
    if int_real_fourier > 0:
        phase = atan + np.pi/2
    else:
        phase = atan - np.pi/2
        
    return phase


# If another set of data is passed as fourier_neg, the function plots a couple of graphs, with the
# one at the top interpreted as the NMR signal produced by a magnetization rotating counter-clockwise,
# the one at the bottom corresponding to the opposite sense of rotation
def plot_fourier_transform(frequencies, fourier, fourier_neg=None, square_modulus=False, xlim=None, ylim=None,
                           scaling_factor=None, norm=True, fig_dpi = 400, show=True, save=False, 
                           name='FTSignal', destination=''):
    """
    Plots the Fourier transform of a signal as a function of the frequency.
  
    Parameters
    ----------
    - frequencies: array-like
                     Sampled values of frequency (in MHz).
    
    - fourier: array-like
               Sampled values of the Fourier transform (in a.u.).
    
    - fourier_neg: array-like
                   Sampled values of the Fourier transform (in a.u.) evaluated
                   at the frequencies in frequencies changed by sign.

                   Default value is `None`.
    
    - square_modulus: bool
                      When True, makes the function plot the square modulus of
                      the Fourier spectrum rather than the separate real and
                      imaginary parts, which is the default option (by default,
                      `square_modulus=False`).
                      
    - `xlim`: 2-element iterable or `None`
              Lower and upper x-axis limits of the plot.
              When `None` uses `matplotlib` default.

    - `ylim`: 2-element iterable or `None`
              Lower and upper y-axis limits of the plot.
              When `None` uses `matplotlib` default.
                      
    - scaling_factor: float
                      When it is not None, it specifies the scaling factor which
                      multiplies the data to be plotted. It applies
                      simultaneously to all the plots in the resulting figure.
    
    - `norm`: Boolean 
              Whether to normalize the fourier transform; i.e., scale it such 
              that its maximum value is 1.
                      
    - fig_dpi: int
            Image quality of the figure when showing and saving. Useful for
            publications. Default set to very high value.
            
    - show: bool
            When False, the graph constructed by the function will not be
            displayed.

            Default value is `True`.
    
    - save: bool
            When `False`, the plotted graph will not be saved on disk. When `True`,
            it will be saved with the name passed as name and in the directory
            passed as destination.
            
            Default value is False.
    
    - name: string
            Name with which the graph will be saved.
    
            Default value is `'FTSignal'`.
    
    - destination: string
                   Path of the directory where the graph will be saved (starting
                   from the current directory). The name of the directory must
                   be terminated with a slash /.
                   
                   Default value is the empty string (current directory).
    
    Action
    ------
    Builds up a plot of the Fourier transform of the passed complex signal as a function of the frequency.
    If fourier_neg is different from None, two graphs are built up which
    represent respectively the Fourier spectra for counter-clockwise and
    clockwise rotation frequencies.
    
    If show=True, the figure is printed on screen.  
  
    Returns
    -------
    An object of the class matplotlib.figure.Figure and an object of the class 
    matplotlib.axis.Axis representing the figure
    built up by the function.
    
    """
    fourier = np.array(fourier)
    frequencies = np.array(frequencies)

    if fourier_neg is None:
        n_plots = 1
        fourier_data = [fourier]
    else:
        n_plots = 2
        fourier_data = [fourier, fourier_neg]
        plot_title = ["Counter-clockwise precession", "Clockwise precession"]
    
    if norm: 
        for i in range(n_plots):
            fourier_data[i] = fourier_data[i] / np.amax(fourier_data[i])

    fig, ax = plt.subplots(n_plots, 1, sharey=True, gridspec_kw={'hspace':0.5})
    
    if fourier_neg is None:
        ax = [ax]
        
    if scaling_factor is not None:
        for i in range(n_plots):
            fourier_data[i] = scaling_factor*fourier_data[i]
        
    for i in range(n_plots):
        if not square_modulus:
            ax[i].plot(frequencies, np.real(fourier_data[i]), label='Real part')
            ax[i].plot(frequencies, np.imag(fourier_data[i]), label='Imaginary part')
        else:
            ax[i].plot(frequencies, np.abs(fourier_data[i]) ** 2, label='Square modulus')
        
        if n_plots>1:
            ax[i].title.set_text(plot_title[i])
        
        ax[i].legend(loc='upper left')
        ax[i].set_xlabel("frequency (MHz)")    
        ax[i].set_ylabel("FT signal (a. u.)")  

        if xlim is not None: 
            ax[i].set_xlim(*xlim)
         
        if ylim is not None: 
            ax[i].set_ylim(*ylim)

    if save: plt.savefig(destination + name, dpi=fig_dpi)
        
    if show: plt.show()
        
    return fig, ax 


def magnus(h_list, rho0, tlist, options=Options()):
    """
    Magnus expansion solver. 

    Parameters: 
    -----------
    - `h_list`: Iterable[Qobj]
                List of Hamiltonians at each time in `tlist`.
    - `rho0`: Qobj
              Initial density matrix 
    - `tlist`: Iterable[float]
               List of times at which the system will be solved. 
    - `options`: qutip.Options
                 Options for this solver. Default: see default `Options`
                 instance as specified by QuTiP.  
    
    Returns:
    --------
    qutip.Result instance with the evolved density matrix. 
    """

    if options.order > 3: 
        raise ValueError('Magnus expansion solver does not support order > 3. ' + \
                        f'Given order {options.order}.')
    output = Result()
    output.times = tlist 
    output.solver = 'magnus'
    time_step = tlist[1] - tlist[0]

    magnus_exp = magnus_expansion_1st_term(h_list, time_step)
    if options.order > 1:
        magnus_exp = magnus_exp + magnus_expansion_2nd_term(h_list, time_step)
        if options.order > 2:
            magnus_exp = magnus_exp + magnus_expansion_3rd_term(h_list, time_step)
        
    dm_evolved_new_picture = rho0.transform((- magnus_exp).expm())
    output.states = [rho0, dm_evolved_new_picture]
    return output


def _ed_evolve_solve_t(t, h, rho0, e_ops):
    """
    Helper function for `ed_evolve`; uses exact diagonalization to evolve 
    the given initial state rho0 by a time `t`. 

    Params
    ------
    - `t`: float
           The time up to which to evolve.
    - `h`: Qobj or List[Qobj]:
           The Hamiltonian describing the system. 
    - `rho0`: Qobj
              The initial state of the system as a density matrix. 
    - `e_ops`: List[Qobj]:
               List of oeprators for which to return the expectation values. 
    
    Returns
    ------
    The evolved density matrix at the time specified by `t`, and the expectation 
    values of each operartor in `e_ops` at `t`. The latter is in the format
    [e_op1[t], e_op2[t], ..., e_opn[t]]. 
    """
    u1, d1, d1exp = exp_diagonalize(1j * 2 * np.pi * h * t)
    u2, d2, d2exp = exp_diagonalize(-1j * 2 * np.pi * h * t)

    rho = u1 * d1exp * u1.inv() * rho0 * u2 * d2exp * u2.inv() 

    if e_ops == None:
        return rho

    exp = np.transpose([[expect(op, rho) for op in e_ops]])
    
    return rho, exp


def ed_evolve(h, rho0, spin, tlist, e_ops=[], state=True, fid=False, par=False, 
    all_t=False, T2=100):
    """
    Evolve the given density matrix with the interactions given by the provided 
    Hamiltonian using exact diagonalization. 

    Params
    ------
    - `h`: Qobj or List[Qobj]:
           The Hamiltonian describing the system. 
    - `rho0`: Qobj
              The initial state of the system as a density matrix. 
    - `spin`: NuclearSpin
              The NuclearSpin object representing the system under study. 
    - `tlist`: List[float]
               List of times at which the system will be evolved. 
    - `e_ops`: List[Qobj]:
               List of operators for which to return the expectation values. 
    - `state`: Boolean 
               Whether to return the density matrix at all. Default `True`.
    - `fid`: Boolean
             Whether to return the free induction decay (FID) signal as 
             an expectation value. If True, appends FID signal to the end of 
             the `e_ops` expectation value list. 
    - `par`: Boolean
             Whether to use QuTiP's parallel computing implementation `parallel_map` 
             to evolve the system.
    - `all_t`: Boolean 
               Whether to return the density matrix and for all times in the
               evolution (as opposed to the last state)
    - T2: iterable[float or function with signature (float) -> float] or float
            or function with signature (float) -> float
    
          If float, characteristic time of relaxation of the component of the
          magnetization on the plane of detection vanishing, i.e., T2. It is
          measured in
          microseconds.

          If function, the decay envelope. 

          If iterable, total decay envelope will be product of decays in list.

          Default value is 100 (microseconds).

    Returns
    ------
    - [0]: The density matrix at time `tlist[-1]` OR the evolved density matrix
    at times specified by `tlist`. 
    
    - [1]: the expectation values of each operator in `e_ops` at the times in
    `tlist`. The latter is in the format `[[e_op1[t1], e_op1[t2], ...] , 
    [e_op2[t1], e_op2[t2]], ..., [e_opn[t1], e_opn[t2], ...]]`. 

    OR 

    The expectation values of each operator in `e_ops` at the times in `tlist`.
    """
    if type(h) is not Qobj and type(h) is list: 
        h = Qobj(sum(h), dims=h[0].dims)

    if fid: 
        e_ops.append(Qobj(np.array(spin.I['+'])))

    rhot = []
    e_opst = []

    decay_envelopes = []
    try:
        for d in T2: 
            if not callable(d):
                decay_envelopes.append(lambda t: np.exp(-t / d))
            else: 
                decay_envelopes.append(d)
    except TypeError:
        if not callable(T2): 
            decay_envelopes.append(lambda t: np.exp(-t / T2))
        else: 
            decay_envelopes.append(T2)

    if par:
        # Check if Jupyter notebook to use QuTiP's Jupyter-optimized parallelization
        try:
            get_ipython().__class__.__name__
            res = ipynb_parallel_map(_ed_evolve_solve_t, tlist, (h, rho0, e_ops))
        except NameError:
            res = parallel_map(_ed_evolve_solve_t, tlist, (h, rho0, e_ops,))
        
        rhot = []
        e_opst = []
        for r, e in res: 
            rhot.append(r)
            e_opst.append(e)

        e_opst = np.concatenate(e_opst, axis=1)

    elif e_ops == []:
        rhot = []
        for t in tlist:
            rhot.append(_ed_evolve_solve_t(t, h, rho0, None))
        e_opst = []

    else:
        rhot = []
        e_opst = [[] for _ in range(len(e_ops))]
        for t in tlist: 
            rho, exp = _ed_evolve_solve_t(t, h, rho0, e_ops)
            e_opst = np.concatenate([e_opst, exp], axis=1)
            rhot.append(rho)

    if fid:
        fid_exp = []
        fids = e_opst[-1]
        for i in range(len(fids)):
            # Obtain total decay envelope at that time.
            env = 1
            for dec in decay_envelopes:
                env *= dec(tlist[i]) # Different name to avoid bizarre variable scope bug
                            # (can't have same name as iteration var in line 1117.)
            fid_exp.append(fids[i] * env)
        
        e_opst[-1] = fid_exp

    if not state: 
        return e_opst

    if all_t:
        return rhot, e_opst 
    else: 
        return rhot[-1], e_opst


def apply_rot_pulse(rho, duration, rot_axis):
    """
    Apply a "pulse" to the given state by rotating the given state by  
    the given duration. I.e., transforms the density matrix by 
    `U = exp(- i * duration * rot_axis)`:
        `U * rho * U.dag()`

    Parameters:
    -----------
    - `rho`: Qobj
             The density matrix of the state to apply the pulse to.
    - `duration`: float
             The duration of the applied pulse as an angle in radians.
    - `rot_axis`: Qobj
             Angular momentum operator for the corresponding axis of rotation. 

    Returns:
    --------
    The transformed density matrix as a Qobj. 
    """
    return rho.transform((-1j * duration * rot_axis).expm())



