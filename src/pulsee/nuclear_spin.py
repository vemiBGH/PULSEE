import numpy as np

from qutip import Qobj, tensor, spin_J_set, qeye

class NuclearSpin:
    """
    An instance of the following class is to be thought as an all-round representation of the
    nuclear spin angular momentum. Indeed, it includes all the operators typically associated
    with the spin and also specific parameters like the spin quantum number and the spin multiplicity.
    
    Attributes
    ----------
    - quantum_number: float
                      Half-integer spin quantum number.
    - d: int
         Dimensions of the spin Hilbert space.
    - gyro_ratio_over_2pi: float
                           Gyromagnetic ratio (over 2 pi) of the nuclear spin measured
                            in units of MHz/T. The gyromagnetic ratio is the constant of
                            proportionality between the intrinsic magnetic moment and the
                            spin angular momentum of a particle.
    - I: dict
         Dictionary whose values are Operator objects representing the cartesian and spherical
         components of the spin.
    - I['+']: spin raising operator;
    - I['-']: spin lowering operator;
    - I['x']: spin x component;
    - I['y']: spin y component;
    - I['z']: spin z component;
                           
    Methods
    -------
    """
    def __init__(self, s=1, gamma_over_2pi=1):
        """
        Constructs an instance of NuclearSpin.
        
        Parameters
        ----------
        - s: float
             Spin quantum number. The constructor checks if it is a half-integer, and raises
             appropriate errors in case this condition is not obeyed.
             Default value is 1;
        - gamma_over_2pi: float
                          Gyromagnetic ratio over 2 pi (in units of MHz/T).
        
        Action
        ------
        Assigns the passed argument s to the attribute quantum_number.
        Assigns the passed argument gamma to the attribute gyromagnetic_ratio.
        Initialises the attribute d with the method multiplicity() (see below).
        Initialises the elements of the dictionary I using the methods described later,
         according to the following correspondence.
        |  I  | Method                    |
        | --- | ------------------------- |
        |  x  | cartesian_operator()[0]   |
        |  y  | cartesian_operator()[1]   |
        |  z  | cartesian_operator()[2]   |
        |  +  | raising_operator()        |
        |  -  | lowering_operator()       |
        |  I  | quantum number            |

    Returns
    -------
    The initialised NuclearSpin object.
    
    Raises
    ------
    ValueError, when the argument s is not a half-integer number (within a relative
    tolerance of 10^(-10)).
        """
        s = float(s)
        if not np.isclose(int(2*s), 2*s, rtol=1e-10):
            raise ValueError("The given spin quantum number is not a half-integer number")
        self.quantum_number = s
        self.d = self.multiplicity()

        # Spin operators
        self.Ix, self.Iy, self.Iz = spin_J_set(self.quantum_number)
        # Raising & lowering operators
        self.Ip, self.Im = (self.Ix + 1j*self.Iy, self.Ix - 1j*self.Iy)
        # Pack everything into a dict
        self.I = {'-': self.Im,
                  '+': self.Ip,
                  'x': self.Ix,
                  'y': self.Iy,
                  'z': self.Iz,
                  'I': self.quantum_number}

        self.gyro_ratio_over_2pi = float(gamma_over_2pi)
        # Helper dimension size of the space
        self.shape = self.I['-'].shape
        self.dims = self.I['-'].dims
    def multiplicity(self):
        """
        Returns the spin states' multiplicity: 2*quantum_number+1 (cast to int).
        """
        return int((2*self.quantum_number)+1)

    def cartesian_operator(self):
        """
        Returns a list of 3 Observable objects representing in the order the x, y and z
        components of the spin.
        Returns:
        -------
        List:
        - [0]: an Observable object standing for the x component of the spin;
        - [1]: an Observable object standing for the y component of the spin;
        - [2]: an Observable object standing for the z component of the spin;
        """
        return [self.Ix, self.Iy, self.Iz]
class ManySpins(NuclearSpin):
    """
    An instance of this class represents a system made up of many nuclear spins, and its
     attributes include the individual NuclearSpin objects, the dimensions of the full Hilbert
      space and the components of the overall spin operator.
    """
    def __init__(self, spins):
        """
        Constructs an instance of ManySpins.
  
        Parameters
        ----------
        - spins: list
                 List of the NuclearSpin objects which represent the spins in the system.
        
        Action
        ------
        Stores the NuclearSpin objects contained in the spins argument into the attribute
        spin, maintaining their original ordering.
  
        Initialises the attribute d with the product of each spin's dimensions d.
  
        Initialises the elements of the dictionary I from the corresponding attributes of its
        spin components by calling the method many_spin_operator.

        Returns
        -------
        The initialised ManySpins object.
        """        
        self.n_spins = len(spins)

        self.spin, self.d, self.dims = ([spins[0]], spins[0].d, spins[0].dims)
        for x in spins[1:]:
            self.spin.append(x)
            self.d = self.d*x.d
            self.dims = np.concatenate([self.dims, x.dims], axis=1)

        if type(self.dims) != list:
            self.dims = self.dims.tolist()
        self.shape = (self.d, self.d)
        self.I = {'-': self.many_spin_operator('-'), '+': self.many_spin_operator('+'),
                  'x': self.many_spin_operator('x'), 'y': self.many_spin_operator('y'),
                  'z': self.many_spin_operator('z')}

    def many_spin_operator(self, component):
        """
        Returns the specified spherical or cartesian component of the spin operator of the
        ManySpins system.
  
        Parameters
        ----------
        - component: string
                     Specifies which component of the overall spin is to be computed,
                     following the key-value correspondence of the attribute I of NuclearSpin.
        
        Returns
        -------
        If component = +, -, an Operator object representing the corresponding spherical spin
        component is returned.
        If component = x, y, z, an Observable object representing the corresponding cartesian spin
        component is returned.
        """

        many_spin_op = Qobj(np.zeros(self.shape), dims=self.dims)
        
        for i in range(self.n_spins):
            term = self.spin[i].I[component]
            for j in range(self.n_spins)[:i]:
                term = tensor(qeye(self.spin[j].d), term)
            for k in range(self.n_spins)[i+1:]:
                term = tensor(term, qeye(self.spin[k].d))
            many_spin_op += Qobj(term.full(), dims=self.dims)

        return many_spin_op