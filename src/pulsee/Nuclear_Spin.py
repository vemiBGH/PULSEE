import math
import numpy as np

from .Operators import Operator, Observable

from .Many_Body import tensor_product

class Nuclear_Spin:
    """
    An instance of the following class is to be thought as an all-round
    representation of the nuclear spin angular momentum. Indeed, it includes all
    the operators typically associated with the spin and also specific
    parameters like the spin quantum number and the spin multiplicity.
    
    Attributes
    ----------
    - quantum_number: float
                      Half-integer spin quantum number.
    - d: int
         Dimensions of the spin Hilbert space.
    - gyro_ratio_over_2pi: float
                           Gyromagnetic ratio (over 2 pi) of the nuclear spin
                           measured in units of MHz/T. The gyromagnetic ratio is
                           the constant of proportionality between the intrinsic
                           magnetic moment and the spin angular momentum of a
                           particle.             
    - I: dict
         Dictionary whose values are Operator objects representing the cartesian
         and spherical components of the spin.
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
        Constructs an instance of Nuclear_Spin.
        
        Parameters
        ----------
        - s: float
             Spin quantum number. The constructor checks if it is a
             half-integer, and raises appropriate errors in case this condition
             is not obeyed.  Default value is 1;
        - gamma_over_2pi: float
                          Gyromagnetic ratio over 2 pi (in units of MHz/T).
        
        Action
        ------
        Assigns the passed argument s to the attribute quantum_number.
        Assigns the passed argument gamma to the attribute gyromagnetic_ratio.
        Initialises the attribute d with the method multiplicity() (see below).
        Initialises the elements of the dictionary I using the methods described
        later, according to the following correspondence.
        |  I  | Method                    |
        | --- | ------------------------- |
        |  x  | cartesian_operator()[0]   |
        |  y  | cartesian_operator()[1]   |
        |  z  | cartesian_operator()[2]   |
        |  +  | raising_operator()        |
        |  -  | lowering_operator()       |

    Returns
    -------
    The initialised Nuclear_Spin object.
    
    Raises
    ------
    ValueError, when the argument s is not a half-integer number (within a
    relative tolerance of 10^(-10)).  
    """
        s = float(s)
        if not math.isclose(int(2 * s), 2 * s, rel_tol=1e-10):
            raise ValueError("The given spin quantum number is not" 
            + " a half-integer number")
        self.quantum_number = s
        self.d = self.multiplicity()
        self.I = {'-': self.lowering_operator(),
                  '+': self.raising_operator(),
                  'x': self.cartesian_operator()[0],
                  'y': self.cartesian_operator()[1],
                  'z': self.cartesian_operator()[2]}

        self.gyro_ratio_over_2pi = float(gamma_over_2pi)
    
    def multiplicity(self):
        """
        Returns the spin states' multiplicity, namely 2*quantum_number+1 (cast
        to int).  
        """
        return int((2*self.quantum_number)+1)

    def raising_operator(self):
        """
        Returns an Operator object representing the raising operator I+ of the
        spin, expressing its matrix attribute with respect to the basis of the
        eigenstates of Iz.  
        """
        I_raising = np.zeros((self.d, self.d))
        for m in range(self.d):
            for n in range(self.d):
                if n - m == 1:
                    I_raising[m, n] = math.sqrt(self.quantum_number 
                        * (self.quantum_number + 1) - (self.quantum_number-n) * 
                        (self.quantum_number-n + 1))
        return Operator(I_raising)

    def lowering_operator(self):
        """
        Returns an Operator object representing the lowering operator I- of the
        spin, expressing its matrix attribute with respect to the basis of the
        eigenstates of Iz. 
        """
        I_lowering = np.zeros((self.d, self.d))
        for m in range(self.d):
            for n in range(self.d):
                if n - m == -1:
                    I_lowering[m, n] = math.sqrt(self.quantum_number *
                     (self.quantum_number + 1) - (self.quantum_number-n) 
                     * (self.quantum_number-n - 1))
        return Operator(I_lowering)

    def cartesian_operator(self):
        """
        Returns a list of 3 Observable objects representing in the order the x,
        y and z components of the spin. The first two are built up exploiting
        their relation with the raising and lowering operators (see the formulas
        above), while the third is simply expressed in its diagonal form, since
        the chosen basis of representation is made up of its eigenstates.
        
        Returns
        -------
        - [0]: an Observable object standing for the x component of the spin;
        - [1]: an Observable object standing for the y component of the spin;
        - [2]: an Observable object standing for the z component of the spin;
        """
        I = []
        I.append(Observable(((self.raising_operator() + self.lowering_operator()) / 2).matrix))
        I.append(Observable(((self.raising_operator() - self.lowering_operator()) / (2j)).matrix))
        I.append(Observable(self.d))
        for m in range(self.d):
            I[2].matrix[m, m] = self.quantum_number - m
        return I    


class Many_Spins(Nuclear_Spin):
    """
    An instance of this class represents a system made up of many nuclear spins,
    and its attributes include the individual Nuclear_Spin objects, the
    dimensions of the full Hilbert space and the components of the overall spin
    operator.
    """
    def __init__(self, spins):
        """
        Constructs an instance of Many_Spins.
  
        Parameters
        ----------
        - spins: list
                 List of the Nuclear_Spin objects which represent the spins in
                 the system.
                 
        Action
        ------
        Stores the Nuclear_Spin objects contained in the spins argument into the
        attribute spin, maintaining their original ordering.
        
        Initialises the attribute d with the product of each spin's dimensions d.
  
        Initialises the elements of the dictionary I from the corresponding
        attributes of its spin components by calling the method
        many_spin_operator.
        
        Returns
        -------
        The initialised Many_Spins object.
        """        
        self.n_spins = len(spins)
        
        self.spin = []
                
        self.d = 1
                
        self.I = {}
        
        for x in spins:
            self.spin.append(x)
            
            self.d = self.d*x.d
            
        self.I['-'] = self.many_spin_operator('-')
        self.I['+'] = self.many_spin_operator('+')
        self.I['x'] = self.many_spin_operator('x')
        self.I['y'] = self.many_spin_operator('y')
        self.I['z'] = self.many_spin_operator('z')
    
    def many_spin_operator(self, component):
        """
        Returns the specified spherical or cartesian component of the spin
        operator of the Many_Spins system.
        
        Parameters
        ----------
        - component: string
                     Specifies which component of the overall spin is to be
                     computed, following the key-value correspondence of the
                     attribute I of Nuclear_Spin.
                     
        Returns
        -------
        If component = +, -, an Operator object representing the corresponding
        spherical spin component is returned.
        
        If component = x, y, z, an Observable object representing the
        corresponding cartesian spin component is returned.  
        """
        many_spin_op = Operator(self.d)*0
        
        for i in range(self.n_spins):
            term = self.spin[i].I[component]
            for j in range(self.n_spins)[:i]:
                term = tensor_product(Operator(self.spin[j].d), term)
            for k in range(self.n_spins)[i+1:]:
                term = tensor_product(term, Operator(self.spin[k].d))
            many_spin_op = many_spin_op + term
            
        if component in 'xyz':
            many_spin_op = many_spin_op.cast_to_observable()
        
        return many_spin_op
            
            
            
            
            