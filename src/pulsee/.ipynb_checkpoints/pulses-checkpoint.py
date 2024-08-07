#        |index|'frequency'|'amplitude'| 'phase' |'theta_p'|'phi_p'|'pulse_time'|
#        |-----|-----------|-----------|---------|---------|-------|------------|
#        |     | (rad/sec) |    (T)    |  (rad)  |  (rad)  | (rad) |   (mus)    |
#        |  0  |  omega_0  |    B_0    | phase_0 | theta_0 | phi_0 |   tau_0    |
#        |  1  |  omega_1  |    B_1    | phase_1 | theta_1 | phi_1 |   tau_1    |
#        | ... |    ...    |    ...    |   ...   |   ...   |  ...  |    ...     |
#        |  N  |  omega_N  |    B_N    | phase_N | theta_N | phi_N |   tau_N    |
import copy
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Pulses:
    frequencies: list[float] = field(default_factory=lambda: [0.0])
    amplitudes: list[float] = field(default_factory=lambda: [0.0])
    phases: list[float] = field(default_factory=lambda: [0.0])
    theta_p: list[float] = field(default_factory=lambda: [0.0])
    phi_p: list[float] = field(default_factory=lambda: [0.0])
    pulse_times: list[float] = field(default_factory=lambda: [0.0])
    shape: str = "square"

    def __post_init__(self):
        self.size=len(self.frequencies)

    def copy(self):
        return copy.deepcopy(self)
    def numpify(self):
        self.frequencies = np.array(self.frequencies)
        self.amplitdues = np.array(self.amplitudes)
        self.phases = np.array(self.phases)
        self.theta_p = np.array(self.theta_p)
        self.phi_p = np.array(self.phi_p)
        self.pulse_times = np.array(self.pulse_times)
    def phase_add_pi(self):
        self.phases = np.add(self.phases, np.pi) 