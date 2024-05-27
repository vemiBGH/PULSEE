#        |index|'frequency'|'amplitude'| 'phase' |'theta_p'|'phi_p'|'pulse_time'|
#        |-----|-----------|-----------|---------|---------|-------|------------|
#        |     | (rad/sec) |    (T)    |  (rad)  |  (rad)  | (rad) |   (mus)    |
#        |  0  |  omega_0  |    B_0    | phase_0 | theta_0 | phi_0 |   tau_0    |
#        |  1  |  omega_1  |    B_1    | phase_1 | theta_1 | phi_1 |   tau_1    |
#        | ... |    ...    |    ...    |   ...   |   ...   |  ...  |    ...     |
#        |  N  |  omega_N  |    B_N    | phase_N | theta_N | phi_N |   tau_N    |
import copy
import numpy as np

class Pulses:
    def __init__(self) -> None:
        self.size = 0
        self.frequencies = []
        self.amplitudes = []
        self.phases = []
        self.theta_p = []
        self.phi_p = []
        self.pulse_times = []
        self.shape = ""
    def copy(self):
        return copy.deepcopy(self)
    def add_pulse(self, frequency: float=0.0, amplitude: float=0.0, phase: float=0.0, theta:float=0.0, 
                  phi:float=0.0, pulse_time:float=0.0, shape: str="square") -> None:
        if not (type(frequency) == float or type(frequency) == int):
            raise TypeError("frequency must be of type float")
        if not (type(amplitude) == float or type(amplitude) == int):
            raise TypeError("amplitude must be of type float")
        if not (type(phase) == float or type(phase) == int):
            raise TypeError("phase must be of type float")
        if not (type(theta) == float or type(theta) == int):
            raise TypeError("theta must be of type float")
        if not (type(phi) == float or type(phi) == int):
            raise TypeError("phi must be of type float")
        if not (type(pulse_time) == float or type(pulse_time) == int):
            raise TypeError("pulse_time must be of type float")
        self.size += 1
        self.frequencies.append(frequency)
        self.amplitudes.append(amplitude)
        self.phases.append(phase)
        self.theta_p.append(theta)
        self.phi_p.append(phi)
        self.pulse_times.append(pulse_time)
    def clean_up(self):
        self.frequencies = np.array(self.frequencies)
        self.amplitdues = np.array(self.amplitudes)
        self.phases = np.array(self.phases)
        self.theta_p = np.array(self.theta_p)
        self.phi_p = np.array(self.phi_p)
        self.pulse_times = np.array(self.pulse_times)
    def phase_add_pi(self):
        self.phases = np.add(self.phases, np.pi)
        