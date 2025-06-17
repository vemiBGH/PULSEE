#        |index|'frequency'|'amplitude'| 'phase' |'theta_p'|'phi_p'|'pulse_time'|'sigma'|
#        |-----|-----------|-----------|---------|---------|-------|------------|-------|
#        |     | (rad/sec) |    (T)    |  (rad)  |  (rad)  | (rad) |   (mus)    | (sec) |
#        |  0  |  omega_0  |    B_0    | phase_0 | theta_0 | phi_0 |   tau_0    | sig_0 |
#        |  1  |  omega_1  |    B_1    | phase_1 | theta_1 | phi_1 |   tau_1    | sig_1 |
#        | ... |    ...    |    ...    |   ...   |   ...   |  ...  |    ...     |  ...  |
#        |  N  |  omega_N  |    B_N    | phase_N | theta_N | phi_N |   tau_N    | sig_N |
import copy
from dataclasses import dataclass, field, fields

import numpy as np


@dataclass
class Pulses:
    frequencies: list[float] = field(default_factory=lambda: [0.0])
    amplitudes: list[float] = field(default_factory=lambda: [0.0])
    phases: list[float] = field(default_factory=lambda: [0.0])
    theta_p: list[float] = field(default_factory=lambda: [0.0])
    phi_p: list[float] = field(default_factory=lambda: [0.0])
    pulse_times: list[float] = field(default_factory=lambda: [0.0])
    shape: str = "square"
    sigma: list[float] = field(default_factory=lambda: [1.0])

    def __post_init__(self):
        self.size = len(self.frequencies)

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

    def add_pulse(self, other: 'Pulses'):
        for attr in fields(self):
            attr_name = attr.name
            if attr_name == "shape":
                continue  # Skip shape
            # Only process attributes that are lists
            self_attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            if isinstance(self_attr, list) and isinstance(other_attr, list):
                self_attr.extend(other_attr)
            else:
                raise ValueError("Currently, we are only allowing pulses attributes to be lists")
        self.size = len(self.frequencies)
