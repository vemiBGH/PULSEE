import copy
from dataclasses import dataclass, field, fields

import numpy as np



@dataclass
class Pulses:
    """
    A class representing a sequence of pulses, each defined by its frequency, amplitude, phase, and other parameters.

    Attributes:
        frequencies (list[float]): List of pulse frequencies in radians per second.
        amplitudes (list[float]): List of pulse amplitudes in Tesla (T).
        phases (list[float]): List of initial phases for each pulse in radians.
        theta_p (list[float]): List of polar angles (theta) in radians.
        phi_p (list[float]): List of azimuthal angles (phi) in radians.
        pulse_times (list[float]): List of pulse durations in microseconds.
        shape (str): The shape of the pulse (default is "square").
        sigma (list[float]): List of standard deviations for Gaussian-shaped pulses (in seconds).
        size (int): The total number of pulses.
    """

    frequencies: list[float] = field(default_factory=lambda: [0.0])
    amplitudes: list[float] = field(default_factory=lambda: [0.0])
    phases: list[float] = field(default_factory=lambda: [0.0])
    theta_p: list[float] = field(default_factory=lambda: [0.0])
    phi_p: list[float] = field(default_factory=lambda: [0.0])
    pulse_times: list[float] = field(default_factory=lambda: [0.0])
    shape: str = "square"
    sigma: list[float] = field(default_factory=lambda: [1.0])

    def __post_init__(self):
        """
        Initializes additional attributes after the dataclass fields are set.
        Specifically, it calculates the number of pulses.
        """
        self.size = len(self.frequencies)

    def copy(self):
        """
        Creates a deep copy of the Pulses object.

        Returns:
            Pulses: A new instance of Pulses with identical attributes.
        """
        return copy.deepcopy(self)

    def numpify(self):
        """
        Converts all list attributes to NumPy arrays for efficient numerical operations.
        """
        self.frequencies = np.array(self.frequencies)
        self.amplitdues = np.array(self.amplitudes)
        self.phases = np.array(self.phases)
        self.theta_p = np.array(self.theta_p)
        self.phi_p = np.array(self.phi_p)
        self.pulse_times = np.array(self.pulse_times)

    def phase_add_pi(self):
        """
        Adds π (pi) to all phase values, effectively shifting all phases by 180 degrees.
        """
        self.phases = np.add(self.phases, np.pi)

    def add_pulse(self, other: 'Pulses'):
        for attr in fields(self):
            attr_name = attr.name
            if attr_name == "shape":
                continue  # Skip shape
            self_attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            # Type check
            valid_types = (list, np.ndarray)
            if not isinstance(self_attr, valid_types) or not isinstance(other_attr, valid_types):
                raise ValueError(
                    f"Both attributes must be a list or numpy array.\n"
                    f"Attribute '{attr_name}':\n"
                    f"  self_attr: {self_attr} (type: {type(self_attr)})\n"
                    f"  other_attr: {other_attr} (type: {type(other_attr)})"
                )
            # Convert both to numpy arrays and concatenate
            arr_self = np.array(self_attr)
            arr_other = np.array(other_attr)
            setattr(self, attr_name, np.concatenate([arr_self, arr_other]))
        self.size = len(self.frequencies)


