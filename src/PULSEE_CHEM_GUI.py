# Generic python imports
# Write to file import
import json
from fractions import Fraction
from functools import partial

# Generic graphics imports
import matplotlib.figure
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
# Kivy imports
from kivy.config import Config

Config.set('graphics', 'resizable', False)

from kivy.app import App

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window

from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup

try:
    from kivy.garden.matplotlib import FigureCanvasKivyAgg
except (ImportError, KeyError) as e:
    print("Locally imported FigureCanvasKivy")
    from pulsee.backend_kivyagg import FigureCanvasKivyAgg

    # catch an ImportError that happens due to some versions of kivy.
    # This shouldn't be a problem in future updates of matplotlib and kivy.

# NMR-NQRSimulationSoftware imports
from qutip import *

from pulsee.nuclear_spin import NuclearSpin

from pulsee.simulation import nuclear_system_setup, \
    evolve, plot_real_part_density_matrix, \
    FID_signal, fourier_transform_signal, \
    plot_fourier_transform, \
    fourier_phase_shift


# This class defines the object responsible of the management of the inputs and outputs of the
# simulation, mediating the interaction between the GUI and the computational core of the program.
class Simulation_Manager:
    spin_par = {'quantum number': 0.,
                'gamma/2pi': 0.}

    spin_par2 = {'quantum number': 1 / 2,
                 'gamma/2pi': 0.}

    zeem_par = {'field magnitude': 0.,
                'theta_z': 0.,
                'phi_z': 0.}

    quad_par = {'coupling constant': 0.,
                'asymmetry parameter': 0.,
                'alpha_q': 0.,
                'beta_q': 0.,
                'gamma_q': 0.,
                'order': 3}

    quad_par2 = {'coupling constant': 0.,
                 'asymmetry parameter': 0.,
                 'alpha_q': 0.,
                 'beta_q': 0.,
                 'gamma_q': 0.,
                 'order': 0}

    cs_par = {'delta_iso': 0.}

    D1_par = {'b_D': 0.,
              'theta': 0.}

    D2_par = {'b_D': 0.,
              'theta': 0.}

    hf_par = {'A': 0.,
              'B': 0.}

    J_sec_par = {'J': 0.}

    nu_q = 0

    n_pulses = 1

    pulse = np.ndarray(4, dtype=pd.DataFrame)

    for i in range(4):
        pulse[i] = pd.DataFrame([(0., 0., 0., 0., 0.),
                                 (0., 0., 0., 0., 0.)],
                                columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])

    pulse_time = np.zeros(4)

    evolution_algorithm = np.ndarray(4, dtype=object)

    for i in range(4):
        evolution_algorithm[i] = 'IP'

    RRF_par = np.ndarray(4, dtype=dict)

    for i in range(4):
        RRF_par[i] = {'nu_RRF': 0.,
                      'theta_RRF': 0.,
                      'phi_RRF': 0.}

    canonical_dm_0 = False

    temperature = 300.

    spin = NuclearSpin()

    h_unperturbed = Qobj(np.eye(1))

    decoherence_time = 100.

    dm = np.ndarray(5, dtype=Qobj)

    FID_times = np.ndarray(1)

    FID = np.ndarray(1)

    spectrum_frequencies = np.ndarray(1)

    spectrum_fourier = np.ndarray(1)

    spectrum_fourier_neg = np.ndarray(1)


def print_traceback(err, *args):
    raise err


def clear_and_write_text(text_object, text, *args):
    text_object.select_all()
    text_object.delete_selection()
    text_object.insert_text(text)


def null_string(text):
    if text == '':
        return 0
    else:
        return text


# Class of the page of the software which lists the parameters of the system
class System_Parameters(FloatLayout):
    d = 1

    dm_elements = np.empty((d, d), dtype=Widget)
    dm_elements[0, 0] = TextInput()

    manual_dm = Widget()

    error_spin_qn = Label()

    error_build_system = Label()

    tb_button = Button()

    dm_graph_box = Widget()

    nu_q_label = Label()

    dm_initial_figure = matplotlib.figure.Figure()

    ns_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nuclear_species.txt")

    # Specifies the action of the checkbox 'Canonical', i.e. to toggle the TextInput widgets associated
    # with the temperature and the density matrix to be inserted manually
    def on_canonical_active(self, *args):
        self.temperature.disabled = not self.temperature.disabled

        if self.spin_qn.text != '':
            for i in range(self.d):
                for j in range(self.d):
                    self.dm_elements[i, j].disabled = not self.dm_elements[i, j].disabled

    # Specifies the action carried out after the validation of the spin quantum number, i.e. the
    # creation of the inputs for the elements of the density matrix
    def set_dm_grid(self, y_shift, *args):
        try:
            self.remove_widget(self.error_spin_qn)
            self.remove_widget(self.manual_dm)

            self.d = int(Fraction(null_string(self.spin_qn.text)) * 2 + 1)

            if (null_string(self.spin_qn2.text) != 0.):
                self.d = self.d + int(Fraction(null_string(self.spin_qn2.text)) * 2 + 1)

            if self.d <= 8:
                self.el_width = 40
            else:
                self.el_width = 30

            # Sets the grid representing the initial density matrix to be filled manually
            self.manual_dm = GridLayout(cols=self.d, size=(self.el_width * self.d, self.el_width * self.d),
                                        size_hint=(None, None), pos=(50, y_shift + 400 - self.d * self.el_width))
            self.dm_elements = np.empty((self.d, self.d), dtype=TextInput)
            for j in range(self.d):
                for i in range(self.d):
                    self.dm_elements[i, j] = TextInput(multiline=False, disabled=False)
                    self.manual_dm.add_widget(self.dm_elements[i, j])
                    self.dm_elements[i, j].disabled = not self.temperature.disabled
            self.add_widget(self.manual_dm)

        # Prints any error raised after the validation of the spin quantum number below its TextInput
        except Exception as e:
            self.error_spin_qn = Label(text=e.args[0], pos=(50, -65), size=(200, 200), bold=True, color=(1, 0, 0, 1),
                                       font_size='15sp')
            self.add_widget(self.error_spin_qn)

    # Stores the value of nu_q among the attributes of sim_man and writes it on screen
    def store_and_write_nu_q(self, sim_man):
        sim_man.nu_q = 3 * sim_man.quad_par['coupling constant'] / \
                       (2 * sim_man.spin_par['quantum number'] * \
                        (2 * sim_man.spin_par['quantum number'] - 1)) * \
                       np.sqrt(1 + (sim_man.quad_par['asymmetry parameter'] ** 2) / 3)

        self.nu_q_label = Label(text="\N{GREEK SMALL LETTER NU}Q = " + str(round(sim_man.nu_q, 2)) + "MHz",
                                pos=(250, 140), size=(200, 200), font_size='15sp')
        self.add_widget(self.nu_q_label)

    def view_the_initial_density_matrix(self, sim_man):
        plot_real_part_density_matrix(sim_man.dm[0], show=False)

        self.dm_graph_box = BoxLayout(size=(300, 300), size_hint=(None, None), pos=(470, 95))

        self.dm_initial_figure = plt.gcf()

        self.dm_graph_box.add_widget(FigureCanvasKivyAgg(self.dm_initial_figure))

        self.add_widget(self.dm_graph_box)

    # Builds up the objects representing the nuclear system
    def build_system(self, sim_man, *args):
        try:

            self.remove_widget(self.error_build_system)

            self.remove_widget(self.tb_button)

            self.remove_widget(self.dm_graph_box)

            self.remove_widget(self.nu_q_label)

            plt.close(self.dm_initial_figure)

            # Storage of the system's input parameters
            sim_man.spin_par['quantum number'] = float(Fraction(null_string(self.spin_qn.text)))

            sim_man.spin_par2['quantum number'] = float(Fraction(null_string(self.spin_qn.text)))

            n_s = int(2 * sim_man.spin_par['quantum number'] + 1)

            sim_man.canonical_dm_0 = self.canonical_checkbox.active

            if sim_man.canonical_dm_0 == False and \
                    n_s != self.d:
                raise IndexError(
                    "The dimensions of the initial density matrix" + '\n' + "don't match the spin states' multiplicity")

            sim_man.spin_par['gamma/2pi'] = float(null_string(self.gyro.text))

            sim_man.zeem_par['field magnitude'] = float(null_string(self.field_mag.text))

            sim_man.spin_par2['gamma/2pi'] = float(null_string(self.gyro2.text))

            sim_man.zeem_par['theta_z'] = (float(null_string(self.theta_z.text)) * np.pi) / 180

            sim_man.zeem_par['phi_z'] = (float(null_string(self.phi_z.text)) * np.pi) / 180

            sim_man.quad_par['coupling constant'] = float(null_string(self.coupling.text))

            sim_man.quad_par['asymmetry parameter'] = float(null_string(self.asymmetry.text))

            sim_man.quad_par['alpha_q'] = (float(null_string(self.alpha_q.text)) * np.pi) / 180

            sim_man.quad_par['beta_q'] = (float(null_string(self.beta_q.text)) * np.pi) / 180

            sim_man.quad_par['gamma_q'] = (float(null_string(self.gamma_q.text)) * np.pi) / 180

            sim_man.quad_par2['coupling constant'] = float(null_string(self.coupling2.text))

            sim_man.quad_par2['asymmetry parameter'] = float(null_string(self.asymmetry2.text))

            sim_man.quad_par2['alpha_q'] = (float(null_string(self.alpha_q2.text)) * np.pi) / 180

            sim_man.quad_par2['beta_q'] = (float(null_string(self.beta_q2.text)) * np.pi) / 180

            sim_man.quad_par2['gamma_q'] = (float(null_string(self.gamma_q2.text)) * np.pi) / 180

            sim_man.cs_par['delta_iso'] = float(null_string(self.chemical_shift.text))

            sim_man.D1_par['b_D'] = float(null_string(self.bd1.text))

            sim_man.D1_par['theta'] = (float(null_string(self.bdtheta1.text)) * np.pi) / 180

            sim_man.D2_par['b_D'] = float(null_string(self.bd2.text))

            sim_man.D2_par['theta'] = (float(null_string(self.bdtheta2.text)) * np.pi) / 180

            sim_man.hf_par['A'] = float(null_string(self.hfA.text))

            sim_man.hf_par['B'] = float(null_string(self.hfB.text))

            sim_man.J_sec_par['J'] = float(null_string(self.Jparam.text))

            if not np.isclose(sim_man.spin_par['quantum number'], 1 / 2, rtol=1e-10):
                self.store_and_write_nu_q(sim_man)

            sim_man.decoherence_time = float(null_string(self.decoherence.text))

            sim_man.temperature = float(null_string(self.temperature.text))

            self.manual_dm_elements = np.zeros((n_s, n_s), dtype=complex)

            spin_paramters = [sim_man.spin_par, sim_man.spin_par2]
            quad_paramters = [sim_man.quad_par, sim_man.quad_par2]

            if sim_man.canonical_dm_0:
                sim_man.spin, sim_man.h_unperturbed, sim_man.dm[0] = \
                    nuclear_system_setup(spin_paramters, \
                                         quad_paramters, \
                                         sim_man.zeem_par, \
                                         None, \
                                         sim_man.cs_par, \
                                         sim_man.D1_par, \
                                         sim_man.D2_par, \
                                         sim_man.hf_par, \
                                         None, \
                                         sim_man.J_sec_par, \
                                         initial_state='canonical', \
                                         temperature=sim_man.temperature)

            else:
                for i in range(self.d):
                    for j in range(self.d):
                        self.manual_dm_elements[i, j] = complex(null_string(self.dm_elements[i, j].text))

                sim_man.spin, sim_man.h_unperturbed, sim_man.dm[0] = \
                    nuclear_system_setup(spin_paramters, \
                                         quad_paramters, \
                                         sim_man.zeem_par, \
                                         None, \
                                         sim_man.cs_par, \
                                         sim_man.D1_par, \
                                         sim_man.D2_par, \
                                         sim_man.hf_par, \
                                         None, \
                                         sim_man.J_sec_par, \
                                         initial_state=self.manual_dm_elements, \
                                         temperature=300)

            self.view_the_initial_density_matrix(sim_man)

        except Exception as e:
            self.error_build_system = Label(text=e.args[0], pos=(225, -480), size=(200, 200), bold=True,
                                            color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_build_system)

            self.tb_button = Button(text='traceback', size_hint=(0.1, 0.03), pos=(442.5, 50))
            self.tb_button.bind(on_release=partial(print_traceback, e))
            self.add_widget(self.tb_button)

    def nuclear_spin_parameters(self, x_shift, y_shift, sim_man):

        # Nuclear species dropdown list
        self.nuclear_species = Button(text='Nuclear species', size_hint=(0.15, 0.045),
                                      pos=(x_shift + 50, y_shift + 450))
        self.add_widget(self.nuclear_species)

        self.nucleus_dd = DropDown()
        # Displays the options when nuclear_species is pressed
        self.nuclear_species.bind(on_release=self.nucleus_dd.open)

        # Waits for the selection of an option in the list, then takes the text of the corresponding
        # button and assigns it to the button nuclear_species
        self.nucleus_dd.bind(on_select=lambda instance, x: setattr(self.nuclear_species, 'text', x))

        # Reads the properties of various nuclear species from a JSON file
        with open(self.ns_file_path) as ns_file:
            ns_data = json.load(ns_file)

        ns_names = ns_data.keys()

        self.species_btn = {}

        for name in ns_names:
            self.species_btn[name] = Button(text=name, size_hint_y=None, height=25)
            # When a button in the list is pressed, the on_select event is triggered and the text of
            # the chosen button is passed as the argument x in the callback launched by nucleus_dd
            self.species_btn[name].bind(on_release=lambda btn: self.nucleus_dd.select(btn.text))
            self.nucleus_dd.add_widget(self.species_btn[name])

        # Spin quantum number
        self.spin_qn_label = Label(text='Spin quantum number', size=(10, 5), pos=(x_shift - 130, y_shift - 25),
                                   font_size='15sp')
        self.add_widget(self.spin_qn_label)

        self.spin_qn = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 355, y_shift + 460))
        self.add_widget(self.spin_qn)

        # After the selection of one of the options in the dropdown list, the spin quantum number
        # takes the corresponding value
        for name in ns_names:
            self.species_btn[name].bind(on_release=partial(clear_and_write_text, self.spin_qn, \
                                                           str(ns_data[name]['spin quantum number'])))

        # Gyromagnetic ratio
        self.gyro_label = Label(text='\N{GREEK SMALL LETTER GAMMA}/2\N{GREEK SMALL LETTER PI}', size=(10, 5),
                                pos=(x_shift + 100, y_shift - 25), font_size='15sp')
        self.add_widget(self.gyro_label)

        self.gyro = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 530, y_shift + 460))
        self.add_widget(self.gyro)

        for name in ns_names:
            self.species_btn[name].bind(on_release=partial(clear_and_write_text, self.gyro, \
                                                           str(ns_data[name]['gamma/2pi'])))

        self.gyro_unit_label = Label(text='MHz/T', size=(10, 5), pos=(x_shift + 220, y_shift - 25), font_size='15sp')
        self.add_widget(self.gyro_unit_label)

    def nuclear_spin_parameters2(self, x_shift, y_shift, sim_man):

        # Nuclear species dropdown list
        self.nuclear_species2 = Button(text='Spin species', size_hint=(0.15, 0.045), pos=(x_shift + 50, y_shift + 450))
        self.add_widget(self.nuclear_species2)

        self.nucleus_dd2 = DropDown()
        # Displays the options when nuclear_species is pressed
        self.nuclear_species2.bind(on_release=self.nucleus_dd2.open)

        # Waits for the selection of an option in the list, then takes the text of the corresponding
        # button and assigns it to the button nuclear_species
        self.nucleus_dd2.bind(on_select=lambda instance, x: setattr(self.nuclear_species2, 'text', x))

        # Reads the properties of various nuclear species from a JSON file
        with open(self.ns_file_path) as ns_file:
            ns_data2 = json.load(ns_file)

        ns_names2 = ns_data2.keys()

        self.species_btn2 = {}

        for name in ns_names2:
            self.species_btn2[name] = Button(text=name, size_hint_y=None, height=25)
            # When a button in the list is pressed, the on_select event is triggered and the text of
            # the chosen button is passed as the argument x in the callback launched by nucleus_dd
            self.species_btn2[name].bind(on_release=lambda btn: self.nucleus_dd2.select(btn.text))
            self.nucleus_dd2.add_widget(self.species_btn2[name])

        # Spin quantum number
        self.spin_qn_label2 = Label(text='Spin quantum number', size=(10, 5), pos=(x_shift - 130, y_shift - 25),
                                    font_size='15sp')
        self.add_widget(self.spin_qn_label2)

        self.spin_qn2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 355, y_shift + 460))
        self.add_widget(self.spin_qn2)

        # After the selection of one of the options in the dropdown list, the spin quantum number
        # takes the corresponding value
        for name in ns_names2:
            self.species_btn2[name].bind(on_release=partial(clear_and_write_text, self.spin_qn2, \
                                                            str(ns_data2[name]['spin quantum number'])))

        # Gyromagnetic ratio
        self.gyro_label2 = Label(text='\N{GREEK SMALL LETTER GAMMA}/2\N{GREEK SMALL LETTER PI}', size=(10, 5),
                                 pos=(x_shift + 100, y_shift - 25), font_size='15sp')
        self.add_widget(self.gyro_label2)

        self.gyro2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 530, y_shift + 460))
        self.add_widget(self.gyro2)

        for name in ns_names2:
            self.species_btn2[name].bind(on_release=partial(clear_and_write_text, self.gyro2, \
                                                            str(ns_data2[name]['gamma/2pi'])))

        self.gyro_unit_label2 = Label(text='MHz/T', size=(10, 5), pos=(x_shift + 220, y_shift - 25), font_size='15sp')
        self.add_widget(self.gyro_unit_label2)

    def magnetic_field_parameters(self, x_shift, y_shift):

        self.mag_field_label = Label(text='Magnetic field', size=(10, 5), pos=(x_shift - 425, y_shift + 50),
                                     font_size='20sp')
        self.add_widget(self.mag_field_label)

        # Field magnitude
        self.field_mag_label = Label(text='Field magnitude', size=(10, 5), pos=(x_shift - 286, y_shift + 50),
                                     font_size='15sp')
        self.add_widget(self.field_mag_label)

        self.field_mag = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 170, y_shift + 535))
        self.add_widget(self.field_mag)

        self.field_mag_unit = Label(text='T', size=(10, 5), pos=(x_shift - 155, y_shift + 50), font_size='15sp')
        self.add_widget(self.field_mag_unit)

        # Polar angle
        self.theta_z_label = Label(text='\N{GREEK SMALL LETTER THETA}z', size=(10, 5),
                                   pos=(x_shift - 110, y_shift + 50), font_size='15sp')
        self.add_widget(self.theta_z_label)

        self.theta_z = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 310, y_shift + 535))
        self.add_widget(self.theta_z)

        self.theta_z_unit = Label(text='°', size=(10, 5), pos=(x_shift - 20, y_shift + 50), font_size='15sp')
        self.add_widget(self.theta_z_unit)

        # Azimuthal angle
        self.phi_z_label = Label(text='\N{GREEK SMALL LETTER PHI}z', size=(10, 50), pos=(x_shift + 45, y_shift + 50),
                                 font_size='15sp')
        self.add_widget(self.phi_z_label)

        self.phi_z = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 470, y_shift + 535))
        self.add_widget(self.phi_z)

        self.phi_z_unit = Label(text='°', size=(10, 5), pos=(x_shift + 140, y_shift + 50), font_size='15sp')
        self.add_widget(self.phi_z_unit)

    def quadrupole_parameters(self, x_shift=0, y_shift=0):
        self.quad_int = Label(text='Quadrupolar 1st', size=(10, 5), pos=(x_shift - 430, y_shift + 90), font_size='20sp')
        self.add_widget(self.quad_int)

        # Coupling constant
        self.coupling_label = Label(text='Coupling constant e\N{SUPERSCRIPT TWO}qQ', size=(10, 5),
                                    pos=(x_shift - 271, y_shift + 90), font_size='15sp')
        self.add_widget(self.coupling_label)

        self.coupling = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 220, y_shift + 575))
        self.add_widget(self.coupling)

        self.coupling_unit = Label(text='MHz', size=(10, 5), pos=(x_shift - 95, y_shift + 90), font_size='15sp')
        self.add_widget(self.coupling_unit)

        # Asymmetry parameter
        self.asymmetry_label = Label(text='Asymmetry parameter', size=(10, 5), pos=(x_shift + 25, y_shift + 90),
                                     font_size='15sp')
        self.asymmetry_label = Label(text='Asymmetry parameter', size=(10, 5), pos=(x_shift + 25, y_shift + 90),
                                     font_size='15sp')
        self.add_widget(self.asymmetry_label)

        self.asymmetry = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 515, y_shift + 575))
        self.add_widget(self.asymmetry)

        # Euler angles
        self.alpha_q_label = Label(text='\N{GREEK SMALL LETTER ALPHA}Q', size=(10, 5),
                                   pos=(x_shift - 341, y_shift + 60), font_size='15sp')
        self.add_widget(self.alpha_q_label)

        self.alpha_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 80, y_shift + 545))
        self.add_widget(self.alpha_q)

        self.alpha_q_unit = Label(text='°', size=(10, 5), pos=(x_shift - 250, y_shift + 60), font_size='15sp')
        self.add_widget(self.alpha_q_unit)

        self.beta_q_label = Label(text='\N{GREEK SMALL LETTER BETA}Q', size=(10, 5), pos=(x_shift - 180, y_shift + 60),
                                  font_size='15sp')
        self.add_widget(self.beta_q_label)

        self.beta_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 241, y_shift + 545))
        self.add_widget(self.beta_q)

        self.beta_q_unit = Label(text='°', size=(10, 5), pos=(x_shift - 89, y_shift + 60), font_size='15sp')
        self.add_widget(self.beta_q_unit)

        self.gamma_q_label = Label(text='\N{GREEK SMALL LETTER GAMMA}Q', size=(10, 5), pos=(x_shift - 19, y_shift + 60),
                                   font_size='15sp')
        self.add_widget(self.gamma_q_label)

        self.gamma_q = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 402, y_shift + 545))
        self.add_widget(self.gamma_q)

        self.gamma_q_unit = Label(text='°', size=(10, 5), pos=(x_shift + 73, y_shift + 60), font_size='15sp')
        self.add_widget(self.gamma_q_unit)

    def quadrupole_parameters1(self, x_shift=0, y_shift=0):
        self.quad_int2 = Label(text='Quadrupolar 2nd', size=(10, 5), pos=(x_shift - 430, y_shift + 90),
                               font_size='20sp')
        self.add_widget(self.quad_int2)

        # Coupling constant
        self.coupling_label2 = Label(text='Coupling constant e\N{SUPERSCRIPT TWO}qQ', size=(10, 5),
                                     pos=(x_shift - 271, y_shift + 90), font_size='15sp')
        self.add_widget(self.coupling_label2)

        self.coupling2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 220, y_shift + 575))
        self.add_widget(self.coupling2)

        self.coupling_unit2 = Label(text='MHz', size=(10, 5), pos=(x_shift - 95, y_shift + 90), font_size='15sp')
        self.add_widget(self.coupling_unit2)

        # Asymmetry parameter
        self.asymmetry_label2 = Label(text='Asymmetry parameter', size=(10, 5), pos=(x_shift + 25, y_shift + 90),
                                      font_size='15sp')
        self.asymmetry_label2 = Label(text='Asymmetry parameter', size=(10, 5), pos=(x_shift + 25, y_shift + 90),
                                      font_size='15sp')
        self.add_widget(self.asymmetry_label2)

        self.asymmetry2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 515, y_shift + 575))
        self.add_widget(self.asymmetry2)

        # Euler angles
        self.alpha_q_label2 = Label(text='\N{GREEK SMALL LETTER ALPHA}Q', size=(10, 5),
                                    pos=(x_shift - 341, y_shift + 60), font_size='15sp')
        self.add_widget(self.alpha_q_label2)

        self.alpha_q2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 80, y_shift + 545))
        self.add_widget(self.alpha_q2)

        self.alpha_q_unit2 = Label(text='°', size=(10, 5), pos=(x_shift - 250, y_shift + 60), font_size='15sp')
        self.add_widget(self.alpha_q_unit2)

        self.beta_q_label2 = Label(text='\N{GREEK SMALL LETTER BETA}Q', size=(10, 5), pos=(x_shift - 180, y_shift + 60),
                                   font_size='15sp')
        self.add_widget(self.beta_q_label2)

        self.beta_q2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 241, y_shift + 545))
        self.add_widget(self.beta_q2)

        self.beta_q_unit2 = Label(text='°', size=(10, 5), pos=(x_shift - 89, y_shift + 60), font_size='15sp')
        self.add_widget(self.beta_q_unit2)

        self.gamma_q_label2 = Label(text='\N{GREEK SMALL LETTER GAMMA}Q', size=(10, 5),
                                    pos=(x_shift - 19, y_shift + 60), font_size='15sp')
        self.add_widget(self.gamma_q_label2)

        self.gamma_q2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 402, y_shift + 545))
        self.add_widget(self.gamma_q2)

        self.gamma_q_unit2 = Label(text='°', size=(10, 5), pos=(x_shift + 73, y_shift + 60), font_size='15sp')
        self.add_widget(self.gamma_q_unit2)

    def chemical_shift_parameters(self, x_shift=0, y_shift=0):
        # Chemical shift
        # title
        self.chem_shiftLabel = Label(text='Chemical Shift', size=(10, 5), pos=(x_shift - 320, y_shift + 90),
                                     font_size='20sp')
        self.add_widget(self.chem_shiftLabel)

        self.chem_shiftLabel1 = Label(text='Isotropic chemical shift, \N{GREEK SMALL LETTER DELTA}=', size=(10, 5),
                                      pos=(x_shift - 151, y_shift + 90), font_size='15sp')
        self.add_widget(self.chem_shiftLabel1)

        self.chemical_shift = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 350, y_shift + 575))
        self.add_widget(self.chemical_shift)

        self.chemical_shiftUnit = Label(text='MHz', size=(10, 5), pos=(x_shift + 35, y_shift + 90), font_size='15sp')
        self.add_widget(self.chemical_shiftUnit)

    def b1_parameters(self, x_shift=0, y_shift=0):
        # D1
        # title
        self.dipolarLabel = Label(text='Dipolar, D1', size=(10, 5), pos=(x_shift - 320, y_shift + 90), font_size='20sp')
        self.add_widget(self.dipolarLabel)

        self.bd1label = Label(text='dipolar contant bD1', size=(10, 5), pos=(x_shift - 151, y_shift + 90),
                              font_size='15sp')
        self.add_widget(self.bd1label)

        self.bd1 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 330, y_shift + 575))
        self.add_widget(self.bd1)

        self.bd1units = Label(text='MHz', size=(10, 5), pos=(x_shift + 15, y_shift + 90), font_size='15sp')
        self.add_widget(self.bd1units)

        self.bd1label1 = Label(text='\N{GREEK SMALL LETTER THETA}', size=(10, 5), pos=(x_shift + 125, y_shift + 90),
                               font_size='15sp')
        self.add_widget(self.bd1label1)

        self.bdtheta1 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 530, y_shift + 575))
        self.add_widget(self.bdtheta1)

        self.theta1units = Label(text='°', size=(10, 5), pos=(x_shift + 200, y_shift + 90), font_size='15sp')
        self.add_widget(self.theta1units)

    def b2_parameters(self, x_shift=0, y_shift=0):
        # D2
        # title
        self.dipolar2Label = Label(text='Dipolar, D2', size=(10, 5), pos=(x_shift - 320, y_shift + 90),
                                   font_size='20sp')
        self.add_widget(self.dipolar2Label)

        self.bd2label = Label(text='dipolar contant bD2', size=(10, 5), pos=(x_shift - 151, y_shift + 90),
                              font_size='15sp')
        self.add_widget(self.bd2label)

        self.bd2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 330, y_shift + 575))
        self.add_widget(self.bd2)

        self.bd2units = Label(text='MHz', size=(10, 5), pos=(x_shift + 15, y_shift + 90), font_size='15sp')
        self.add_widget(self.bd2units)

        self.bd2label = Label(text='\N{GREEK SMALL LETTER THETA}', size=(10, 5), pos=(x_shift + 125, y_shift + 90),
                              font_size='15sp')
        self.add_widget(self.bd2label)

        self.bdtheta2 = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 530, y_shift + 575))
        self.add_widget(self.bdtheta2)

        self.theta2units = Label(text='°', size=(10, 5), pos=(x_shift + 200, y_shift + 90), font_size='15sp')
        self.add_widget(self.theta2units)

    def hf_parameters(self, x_shift=0, y_shift=0):
        # secular Hyperfine interaction
        # title
        self.hfLabel = Label(text='Hyperfine ', size=(10, 5), pos=(x_shift - 320, y_shift + 90), font_size='20sp')
        self.add_widget(self.hfLabel)

        self.Alabel = Label(text='hyperfine constant A', size=(10, 5), pos=(x_shift - 151, y_shift + 90),
                            font_size='15sp')
        self.add_widget(self.Alabel)

        self.hfA = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 330, y_shift + 575))
        self.add_widget(self.hfA)

        self.Aunits = Label(text='MHz', size=(10, 5), pos=(x_shift + 15, y_shift + 90), font_size='15sp')
        self.add_widget(self.Aunits)

        self.Blabel = Label(text='const B', size=(10, 5), pos=(x_shift + 125, y_shift + 90), font_size='15sp')
        self.add_widget(self.Blabel)

        self.hfB = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 560, y_shift + 575))
        self.add_widget(self.hfB)

        self.Bunits = Label(text='MHz', size=(10, 5), pos=(x_shift + 250, y_shift + 90), font_size='15sp')
        self.add_widget(self.Bunits)

    def j_sec_param(self, x_shift=0, y_shift=0):
        # Secular J-coupling interaction
        # title
        self.JLabel = Label(text='Secular J-coupling', size=(10, 5), pos=(x_shift - 300, y_shift + 90),
                            font_size='20sp')
        self.add_widget(self.JLabel)

        self.JLabel1 = Label(text='J=', size=(10, 5), pos=(x_shift - 61, y_shift + 90), font_size='15sp')
        self.add_widget(self.JLabel1)

        self.Jparam = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 350, y_shift + 575))
        self.add_widget(self.Jparam)

        self.JUnit = Label(text='MHz', size=(10, 5), pos=(x_shift + 35, y_shift + 90), font_size='15sp')
        self.add_widget(self.JUnit)

    def decoherence_param(self, x_shift=0, y_shift=0):
        # Decoherence time of the system
        self.decoherence_label = Label(text='Decoherence time', size=(10, 5), pos=(x_shift - 285, y_shift + 70),
                                       font_size='16sp')
        self.add_widget(self.decoherence_label)

        self.decoherence = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(x_shift + 187.5, y_shift + 555))
        self.add_widget(self.decoherence)

        self.decoherence_unit = Label(text='\N{GREEK SMALL LETTER MU}s', size=(10, 5),
                                      pos=(x_shift - 135, y_shift + 70), font_size='15sp')
        self.add_widget(self.decoherence_unit)

    def initial_dm_parameters(self, x_shift, y_shift):

        self.initial_dm = Label(text='Initial density matrix', size=(10, 5), pos=(x_shift - 260, y_shift + 30),
                                font_size='20sp')
        self.add_widget(self.initial_dm)

        self.dm_par = GridLayout(cols=5, size=(500, 35), size_hint=(None, None), pos=(x_shift + 50, y_shift + 470))

        # Checkbox to set the initial density matrix as the canonical one
        self.canonical_checkbox = CheckBox(size_hint_x=None, width=20)
        self.dm_par.add_widget(self.canonical_checkbox)

        self.canonical_label = Label(text='Canonical', font_size='15sp', size_hint_x=None, width=100)
        self.dm_par.add_widget(self.canonical_label)

        # Temperature of the system (required if the canonical checkbox is active)
        self.temperature_label = Label(text='Temperature (K)', font_size='15sp', size_hint_x=None, width=150)
        self.dm_par.add_widget(self.temperature_label)

        # The TextInput of the temperature is initially disabled
        self.temperature = TextInput(multiline=False, disabled=True, size_hint_x=None, width=65, size_hint_y=None,
                                     height=32.5)
        self.dm_par.add_widget(self.temperature)

        self.temperature_unit = Label(text='K', font_size='15sp', size_hint_x=None, width=30)
        self.dm_par.add_widget(self.temperature_unit)

        self.canonical_checkbox.bind(active=self.on_canonical_active)

        self.add_widget(self.dm_par)

        # Button to generate the grid of elements of the initial density matrix
        self.manual_dm_button = Button(text="Set density matrix element by element", font_size='15sp', size_hint_y=None,
                                       height=30, size_hint_x=None, width=280, pos=(50, y_shift + 425))

        self.add_widget(self.manual_dm_button)

        self.manual_dm_button.bind(on_release=partial(self.set_dm_grid, y_shift))

    def __init__(self, sim_man, retrieve_config_btn, retrieve_config_name, **kwargs):
        super().__init__(**kwargs)

        self.parameters = Label(text='System parameters', size=(10, 5), pos=(0, 480), font_size='30sp')
        self.add_widget(self.parameters)

        self.nuclear_spin_parameters(-40, 475, sim_man=sim_man)

        self.nuclear_spin_parameters2(-40, 435, sim_man=sim_man)

        self.magnetic_field_parameters(140, 320)

        self.quadrupole_parameters(100, 250)

        self.quadrupole_parameters1(100, 190)

        self.chemical_shift_parameters(0, 125)

        self.b1_parameters(0, 90)

        self.b2_parameters(0, 60)

        self.hf_parameters(0, 25)

        self.j_sec_param(0, -10)

        self.decoherence_param(0, -30)

        self.initial_dm_parameters(0, -25)

        self.set_up_system = Button(text='Set up the system', font_size='16sp', size_hint_y=None, height=40,
                                    size_hint_x=None, width=200, pos=(525, 50))
        self.set_up_system.bind(on_release=partial(self.build_system, sim_man))

        self.add_widget(self.set_up_system)

        # Button and TextInput which allow to retrieve all the values of the inputs of a previous simulation saved in a JSON file
        self.add_widget(retrieve_config_btn)
        self.add_widget(retrieve_config_name)


# Class of the page of the software which lists the parameters of the pulse sequence
class Pulse_Sequence(FloatLayout):
    pulse_label = np.ndarray(4, dtype=Label)

    pulse_t_label = np.ndarray(4, dtype=Label)
    pulse_times = np.ndarray(4, dtype=TextInput)
    pulse_t_unit = np.ndarray(4, dtype=Label)

    single_pulse_table = np.ndarray(4, dtype=GridLayout)

    frequency_label = np.ndarray(4, dtype=Label)
    amplitude_label = np.ndarray(4, dtype=Label)
    phase_label = np.ndarray(4, dtype=Label)
    theta_p_label = np.ndarray(4, dtype=Label)
    phi_p_label = np.ndarray(4, dtype=Label)

    frequency_unit = np.ndarray(4, dtype=Label)
    amplitude_unit = np.ndarray(4, dtype=Label)
    phase_unit = np.ndarray(4, dtype=Label)
    theta_p_unit = np.ndarray(4, dtype=Label)
    phi_p_unit = np.ndarray(4, dtype=Label)

    frequency = np.ndarray((4, 2), dtype=TextInput)
    amplitude = np.ndarray((4, 2), dtype=TextInput)
    phase = np.ndarray((4, 2), dtype=TextInput)
    theta_p = np.ndarray((4, 2), dtype=TextInput)
    phi_p = np.ndarray((4, 2), dtype=TextInput)

    n_modes = np.ones(4, dtype=int)

    more_modes_btn = np.ndarray(4, dtype=Button)
    less_modes_btn = np.ndarray(4, dtype=Button)

    n_pulses = 1

    number_pulses = TextInput()

    error_n_pulses = Label()

    error_set_up_pulse = Label()

    RRF_btn = np.ndarray(4, dtype=Button)

    RRF_frequency_label = np.ndarray(4, dtype=Label)
    RRF_theta_label = np.ndarray(4, dtype=Label)
    RRF_phi_label = np.ndarray(4, dtype=Label)

    RRF_frequency = np.ndarray(4, dtype=TextInput)
    RRF_theta = np.ndarray(4, dtype=TextInput)
    RRF_phi = np.ndarray(4, dtype=TextInput)

    RRF_frequency_unit = np.ndarray(4, dtype=Label)
    RRF_theta_unit = np.ndarray(4, dtype=Label)
    RRF_phi_unit = np.ndarray(4, dtype=Label)

    for i in range(4):
        RRF_frequency_label[i] = Label()
        RRF_theta_label[i] = Label()
        RRF_phi_label[i] = Label()
        RRF_frequency[i] = TextInput()
        RRF_theta[i] = TextInput()
        RRF_phi[i] = TextInput()
        RRF_frequency_unit[i] = Label()
        RRF_theta_unit[i] = Label()
        RRF_phi_unit[i] = Label()

    IP_btn = np.ndarray(4, dtype=Button)

    # Adds a new line of TextInputs in the table of the n-th pulse
    def add_new_mode(self, n, *args):

        if self.n_modes[n] < 2:
            self.single_pulse_table[n].size[1] = self.single_pulse_table[n].size[1] + 28
            self.single_pulse_table[n].pos[1] = self.single_pulse_table[n].pos[1] - 28

            self.frequency[n][self.n_modes[n]] = TextInput(multiline=False, size_hint=(0.5, 0.75))
            self.single_pulse_table[n].add_widget(self.frequency[n][self.n_modes[n]])

            self.amplitude[n][self.n_modes[n]] = TextInput(multiline=False, size_hint=(0.5, 0.75))
            self.single_pulse_table[n].add_widget(self.amplitude[n][self.n_modes[n]])

            self.phase[n][self.n_modes[n]] = TextInput(multiline=False, size_hint=(0.5, 0.75))
            self.single_pulse_table[n].add_widget(self.phase[n][self.n_modes[n]])

            self.theta_p[n][self.n_modes[n]] = TextInput(multiline=False, size_hint=(0.5, 0.75))
            self.single_pulse_table[n].add_widget(self.theta_p[n][self.n_modes[n]])

            self.phi_p[n][self.n_modes[n]] = TextInput(multiline=False, size_hint=(0.5, 0.75))
            self.single_pulse_table[n].add_widget(self.phi_p[n][self.n_modes[n]])

            self.n_modes[n] = self.n_modes[n] + 1

        else:
            pass

    # Removes a line of TextInputs in the table of the n-th pulse
    def remove_mode(self, n, sim_man, *args):
        if self.n_modes[n] > 1:

            self.n_modes[n] = self.n_modes[n] - 1

            self.single_pulse_table[n].remove_widget(self.frequency[n][self.n_modes[n]])
            self.single_pulse_table[n].remove_widget(self.amplitude[n][self.n_modes[n]])
            self.single_pulse_table[n].remove_widget(self.phase[n][self.n_modes[n]])
            self.single_pulse_table[n].remove_widget(self.theta_p[n][self.n_modes[n]])
            self.single_pulse_table[n].remove_widget(self.phi_p[n][self.n_modes[n]])

            sim_man.pulse[n]['frequency'][self.n_modes[n]] = 0
            sim_man.pulse[n]['amplitude'][self.n_modes[n]] = 0
            sim_man.pulse[n]['phase'][self.n_modes[n]] = 0
            sim_man.pulse[n]['theta_p'][self.n_modes[n]] = 0
            sim_man.pulse[n]['phi_p'][self.n_modes[n]] = 0

            self.single_pulse_table[n].size[1] = self.single_pulse_table[n].size[1] - 28
            self.single_pulse_table[n].pos[1] = self.single_pulse_table[n].pos[1] + 28
        else:
            pass

    # Prints on screen the controls for the parameters of the RRF
    def set_RRF_par(self, n, y_shift):
        self.RRF_frequency_label[n] = Label(text='\N{GREEK SMALL LETTER OMEGA}RRF', size=(10, 5),
                                            pos=(195, y_shift - 145), font_size='15sp')
        self.add_widget(self.RRF_frequency_label[n])
        self.RRF_frequency[n] = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(620, y_shift + 340))
        self.add_widget(self.RRF_frequency[n])
        self.RRF_frequency_unit[n] = Label(text='MHz', size=(10, 5), pos=(300, y_shift - 145), font_size='15sp')
        self.add_widget(self.RRF_frequency_unit[n])

        self.RRF_theta_label[n] = Label(text='\N{GREEK SMALL LETTER THETA}RRF', size=(10, 5), pos=(195, y_shift - 180),
                                        font_size='15sp')
        self.add_widget(self.RRF_theta_label[n])
        self.RRF_theta[n] = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(620, y_shift + 305))
        self.add_widget(self.RRF_theta[n])
        self.RRF_theta_unit[n] = Label(text='°', size=(10, 5), pos=(300, y_shift - 180), font_size='15sp')
        self.add_widget(self.RRF_theta_unit[n])

        self.RRF_phi_label[n] = Label(text='\N{GREEK SMALL LETTER PHI}RRF', size=(10, 5), pos=(195, y_shift - 215),
                                      font_size='15sp')
        self.add_widget(self.RRF_phi_label[n])
        self.RRF_phi[n] = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(620, y_shift + 270))
        self.add_widget(self.RRF_phi[n])
        self.RRF_phi_unit[n] = Label(text='°', size=(10, 5), pos=(300, y_shift - 215), font_size='15sp')
        self.add_widget(self.RRF_phi_unit[n])

    # Removes from the screen the controls of the parameters of the RRF
    def remove_RRF_par(self, n):
        self.remove_widget(self.RRF_frequency_label[n])
        self.remove_widget(self.RRF_frequency[n])
        self.remove_widget(self.RRF_frequency_unit[n])

        self.remove_widget(self.RRF_theta_label[n])
        self.remove_widget(self.RRF_theta[n])
        self.remove_widget(self.RRF_theta_unit[n])

        self.remove_widget(self.RRF_phi_label[n])
        self.remove_widget(self.RRF_phi[n])
        self.remove_widget(self.RRF_phi_unit[n])

    def set_RRF_evolution(self, n, y_shift, sim_man, *args):
        if self.RRF_btn[n].state == 'down':
            sim_man.evolution_algorithm[n] = 'RRF'
            self.IP_btn[n].state = 'normal'

            self.set_RRF_par(n, y_shift)

        else:
            self.remove_RRF_par(n)

            sim_man.evolution_algorithm[n] = 'IP'
            self.IP_btn[n].state = 'down'

    def set_IP_evolution(self, n, y_shift, sim_man, *args):
        if self.IP_btn[n].state == 'down':
            self.remove_RRF_par(n)

            sim_man.evolution_algorithm[n] = 'IP'
            self.RRF_btn[n].state = 'normal'

        else:
            sim_man.evolution_algorithm[n] = 'RRF'
            self.RRF_btn[n].state = 'down'

            self.set_RRF_par(n, y_shift)

    # Creates the set of controls associated with the parameters of a single pulse in the sequence.
    # n is an integer which labels successive pulses
    def single_pulse_par(self, n, y_shift, sim_man):

        self.pulse_label[n] = Label(text='Pulse #%r' % n, size=(10, 5), pos=(-285, y_shift), font_size='20sp')
        self.add_widget(self.pulse_label[n])

        # Duration of the pulse
        self.pulse_t_label[n] = Label(text='Time', size=(10, 5), pos=(-150, y_shift - 2.5), font_size='15sp')
        self.add_widget(self.pulse_t_label[n])

        self.pulse_times[n] = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(275, y_shift + 482.5))
        self.add_widget(self.pulse_times[n])

        self.pulse_t_unit[n] = Label(text='\N{GREEK SMALL LETTER MU}s', size=(10, 5), pos=(-50, y_shift - 2.5),
                                     font_size='15sp')
        self.add_widget(self.pulse_t_unit[n])

        # Parameters of the electromagnetic wave
        self.single_pulse_table[n] = GridLayout(cols=5, size=(400, 100), size_hint=(None, None),
                                                pos=(71, y_shift + 375))

        self.frequency_label[n] = Label(text='Frequency', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.frequency_label[n])

        self.amplitude_label[n] = Label(text='Amplitude', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.amplitude_label[n])

        self.phase_label[n] = Label(text='Phase', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.phase_label[n])

        self.theta_p_label = Label(text='\N{GREEK SMALL LETTER THETA}', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.theta_p_label)

        self.phi_p_label[n] = Label(text='\N{GREEK SMALL LETTER PHI}', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.phi_p_label[n])

        self.frequency_unit[n] = Label(text='(MHz)', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.frequency_unit[n])

        self.amplitude_unit[n] = Label(text='(T)', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.amplitude_unit[n])

        self.phase_unit[n] = Label(text='(°)', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.phase_unit[n])

        self.theta_p_unit = Label(text='(°)', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.theta_p_unit)

        self.phi_p_unit[n] = Label(text='(°)', font_size='15sp')
        self.single_pulse_table[n].add_widget(self.phi_p_unit[n])

        self.add_widget(self.single_pulse_table[n])

        self.frequency[n][0] = TextInput(multiline=False, size_hint=(0.5, 0.75))
        self.single_pulse_table[n].add_widget(self.frequency[n][0])

        self.amplitude[n][0] = TextInput(multiline=False, size_hint=(0.5, 0.75))
        self.single_pulse_table[n].add_widget(self.amplitude[n][0])

        self.phase[n][0] = TextInput(multiline=False, size_hint=(0.5, 0.75))
        self.single_pulse_table[n].add_widget(self.phase[n][0])

        self.theta_p[n, 0] = TextInput(multiline=False, size_hint=(0.5, 0.75))
        self.single_pulse_table[n].add_widget(self.theta_p[n, 0])

        self.phi_p[n][0] = TextInput(multiline=False, size_hint=(0.5, 0.75))
        self.single_pulse_table[n].add_widget(self.phi_p[n][0])

        # Button for the addition of another mode of radiation
        self.more_modes_btn[n] = Button(text='+', font_size='15sp', size_hint=(None, None), size=(30, 30),
                                        pos=(485, y_shift + 374))
        self.more_modes_btn[n].bind(on_press=partial(self.add_new_mode, n))
        self.add_widget(self.more_modes_btn[n])

        # Button for the removal of a mode of radiation
        self.less_modes_btn[n] = Button(text='-', font_size='15sp', size_hint=(None, None), size=(30, 30),
                                        pos=(517.5, y_shift + 374))
        self.less_modes_btn[n].bind(on_press=partial(self.remove_mode, n, sim_man))
        self.add_widget(self.less_modes_btn[n])

        # Buttons which specify the methods of numerical evolution of the system: RRF and IP
        self.RRF_btn[n] = ToggleButton(text='RRF', font_size='15sp', size_hint=(None, None), size=(40, 30),
                                       pos=(575, y_shift + 374))
        self.RRF_btn[n].bind(on_press=partial(self.set_RRF_evolution, n, y_shift, sim_man))
        self.add_widget(self.RRF_btn[n])

        self.IP_btn[n] = ToggleButton(text='IP', font_size='15sp', size_hint=(None, None), size=(40, 30),
                                      pos=(619, y_shift + 374))
        self.IP_btn[n].bind(on_press=partial(self.set_IP_evolution, n, y_shift, sim_man))
        self.IP_btn[n].state = 'down'
        self.add_widget(self.IP_btn[n])

    # Shows all the controls associated with the pulses in the sequence
    def set_pulse_controls(self, sim_man, *args):
        try:
            self.remove_widget(self.error_n_pulses)

            new_n_pulses = int(null_string(self.number_pulses.text))

            if new_n_pulses < 1 or new_n_pulses > 4:
                raise ValueError("The number of pulses in the sequence" + '\n' + "must fall between 1 and 4")

            if self.n_pulses > new_n_pulses:
                for i in range(new_n_pulses, self.n_pulses):
                    self.remove_widget(self.pulse_label[i])
                    self.remove_widget(self.pulse_t_label[i])
                    self.remove_widget(self.pulse_times[i])
                    self.remove_widget(self.pulse_t_unit[i])
                    self.remove_widget(self.single_pulse_table[i])
                    self.remove_widget(self.more_modes_btn[i])
                    self.remove_widget(self.less_modes_btn[i])
                    self.remove_widget(self.RRF_btn[i])
                    self.remove_widget(self.IP_btn[i])
                    if self.RRF_btn[i].state == 'down':
                        self.remove_RRF_par(i)
            else:
                for i in range(self.n_pulses, new_n_pulses):
                    self.single_pulse_par(n=i, y_shift=400 - i * 200, sim_man=sim_man)

            self.n_pulses = new_n_pulses

        except Exception as e:
            self.error_n_pulses = Label(text=e.args[0], pos=(200, 335), size=(200, 200), bold=True, color=(1, 0, 0, 1),
                                        font_size='15sp')
            self.add_widget(self.error_n_pulses)

    # Assigns the input values written on screen to the corresponding variables belonging to the
    # Simulation_Manager
    def set_up_pulse(self, sim_man, *args):
        try:
            self.remove_widget(self.error_set_up_pulse)

            for i in range(self.n_pulses):
                for j in range(self.n_modes[i]):
                    sim_man.pulse[i]['frequency'][j] = float(null_string(self.frequency[i][j].text))
                    sim_man.pulse[i]['amplitude'][j] = float(null_string(self.amplitude[i][j].text))
                    sim_man.pulse[i]['phase'][j] = (float(null_string(self.phase[i][j].text)) * np.pi) / 180
                    sim_man.pulse[i]['theta_p'][j] = (float(
                        null_string(self.theta_p[i][j].text)) * np.pi) / 180 + np.pi / 2
                    sim_man.pulse[i]['phi_p'][j] = (float(null_string(self.phi_p[i][j].text)) * np.pi) / 180
                    sim_man.pulse_time[i] = float(null_string(self.pulse_times[i].text))

                if self.RRF_btn[i].state == 'down':
                    sim_man.evolution_algorithm[i] = "RRF"
                    sim_man.RRF_par[i]['nu_RRF'] = float(null_string(self.RRF_frequency[i].text))
                    sim_man.RRF_par[i]['theta_RRF'] = (float(null_string(self.RRF_theta[i].text)) / 180) * np.pi
                    sim_man.RRF_par[i]['phi_RRF'] = (float(null_string(self.RRF_phi[i].text)) / 180) * np.pi
                else:
                    sim_man.evolution_algorithm[i] = "IP"

            sim_man.n_pulses = self.n_pulses

        except Exception as e:
            self.error_set_up_pulse = Label(text=e.args[0], pos=(0, -490), size=(200, 200), bold=True,
                                            color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_set_up_pulse)

            self.tb_button = Button(text='traceback', size_hint=(0.1, 0.03), pos=(450, 25))
            self.tb_button.bind(on_release=partial(print_traceback, e))
            self.add_widget(self.tb_button)

    def __init__(self, sim_man, **kwargs):
        super().__init__(**kwargs)

        self.pulse_sequence_label = Label(text='Pulse sequence', size=(10, 5), pos=(0, 450), font_size='30sp')
        self.add_widget(self.pulse_sequence_label)

        self.single_pulse_par(n=0, y_shift=400, sim_man=sim_man)

        # Question mark connected with the explanation of RRF and IP buttons
        self.RRF_IP_question_mark = Label(text='[ref=?]?[/ref]', markup=True, size=(20, 20), pos=(280, 290),
                                          font_size='20sp')
        self.add_widget(self.RRF_IP_question_mark)

        # Popup message which explains the meaning of the RRF and IP buttons
        explanation = 'The acronyms RRF and IP indicate two ways' + '\n' + \
                      'to derive the evolution of the system in' + '\n' + \
                      'the simulation. RRF (Rotating Reference Frame)' + '\n' + \
                      'consists in treating the dynamics of the system' + '\n' + \
                      'in the reference frame which rotates with' + '\n' + \
                      'respect to the laboratory with the specified' + '\n' + \
                      'frequency and orientation. IP (Interaction Picture)' + '\n' + \
                      'implies a change of dynamical picture which makes' + '\n' + \
                      'the evolution of the state depend only on the' + '\n' + \
                      'perturbation, i.e. the applied pulse.'
        self.RRF_IP_popup = Popup(title='RRF and IP', content=Label(text=explanation), size_hint=(0.475, 0.45),
                                  pos=(425, 500), auto_dismiss=True)

        self.RRF_IP_question_mark.bind(on_ref_press=self.RRF_IP_popup.open)

        # Number of pulses in the sequence
        self.number_pulses_label = Label(text='Number of pulses', size=(10, 5), pos=(250, 400), font_size='20sp')
        self.add_widget(self.number_pulses_label)

        self.number_pulses = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(670, 850))
        self.number_pulses.bind(on_text_validate=partial(self.set_pulse_controls, sim_man))
        self.add_widget(self.number_pulses)

        self.set_up_pulse_btn = Button(text='Set up the pulse sequence', font_size='16sp', size_hint_y=None, height=40,
                                       size_hint_x=None, width=220, pos=(535, 25))
        self.set_up_pulse_btn.bind(on_press=partial(self.set_up_pulse, sim_man))
        self.add_widget(self.set_up_pulse_btn)


# Class of the page of the software which shows the system's state throughout the various stages of
# evolution
class Evolution_Results(FloatLayout):
    error_evolution = Label()

    tb_evolution = Button()

    evolved_dm_panels = Widget()

    # Function that actually evolves the system through all the steps of the specified sequence
    def launch_evolution(self, sim_man, *args):
        try:
            self.remove_widget(self.error_evolution)
            self.remove_widget(self.tb_evolution)
            self.remove_widget(self.evolved_dm_panels)

            for i in range(sim_man.n_pulses):
                sim_man.dm[i + 1] = evolve(sim_man.spin, sim_man.h_unperturbed, sim_man.dm[i], \
                                           sim_man.pulse[i], sim_man.pulse_time[i], \
                                           picture=sim_man.evolution_algorithm[i], \
                                           RRF_par=sim_man.RRF_par[i])

            # Recap of the input parameters

            print("Field magnitude = " + str(sim_man.zeem_par['field magnitude']) + " T")
            print("theta_z = " + str(sim_man.zeem_par['theta_z']) + " rad")
            print("phi_z = " + str(sim_man.zeem_par['phi_z']) + " rad")

            print('\n')

            print("First Spin:")

            print("Spin quantum number = " + str(sim_man.spin_par['quantum number']))
            print("gamma/2pi = " + str(sim_man.spin_par['gamma/2pi']) + " MHz")

            print("e2qQ = " + str(sim_man.quad_par['coupling constant']) + " MHz")
            print("Asymmetry = " + str(sim_man.quad_par['asymmetry parameter']))
            print("alpha_q = " + str(sim_man.quad_par['alpha_q']) + " rad")
            print("beta_q = " + str(sim_man.quad_par['beta_q']) + " rad")
            print("gamma_q = " + str(sim_man.quad_par['gamma_q']) + " rad")

            print('\n')

            print("*******")

            print("Second Spin:")
            print("Spin quantum number = " + str(sim_man.spin_par2['quantum number']))
            print("gamma/2pi = " + str(sim_man.spin_par2['gamma/2pi']) + " MHz")

            print("e2qQ = " + str(sim_man.quad_par2['coupling constant']) + " MHz")
            print("Asymmetry = " + str(sim_man.quad_par2['asymmetry parameter']))
            print("alpha_q = " + str(sim_man.quad_par2['alpha_q']) + " rad")
            print("beta_q = " + str(sim_man.quad_par2['beta_q']) + " rad")
            print("gamma_q = " + str(sim_man.quad_par2['gamma_q']) + " rad")

            print('\n')

            print("chemical_shift, delta_iso = " + str(sim_man.cs_par['delta_iso']) + " MHz")

            print('\n')

            print("b_D1 = " + str(sim_man.D1_par['b_D']) + " MHz")
            print("theta_bD1 = " + str(sim_man.D1_par['theta']) + " rad")

            print('\n')

            print("b_D2 = " + str(sim_man.D2_par['b_D']) + " MHz")
            print("theta_bD2 = " + str(sim_man.D2_par['theta']) + " rad")

            print('\n')

            print("A_hf = " + str(sim_man.hf_par['A']) + " MHz")
            print("B_hf = " + str(sim_man.hf_par['B']) + " rad")

            print('\n')

            print("J_secular = " + str(sim_man.J_sec_par['J']) + " MHz")

            print('\n')

            print("# pulses = " + str(sim_man.n_pulses))
            for i in range(sim_man.n_pulses):
                for j in range(2):
                    print("frequency (pulse #" + str(i) + " mode #" + str(j) + ") = " + str(
                        sim_man.pulse[i]['frequency'][j]) + " MHz")
                    print("amplitude (pulse #" + str(i) + " mode #" + str(j) + ") = " + str(
                        sim_man.pulse[i]['amplitude'][j]) + " T")
                    print("phase (pulse #" + str(i) + " mode #" + str(j) + ")" + " = " + str(
                        sim_man.pulse[i]['phase'][j]) + " rad")
                    print("theta_p (pulse #" + str(i) + " mode #" + str(j) + ")" + " = " + str(
                        sim_man.pulse[i]['theta_p'][j]) + " rad")
                    print("phi_p (pulse #" + str(i) + " mode #" + str(j) + ")" + " = " + str(
                        sim_man.pulse[i]['phi_p'][j]) + " rad")

                print("Pulse time (pulse #" + str(i) + ") = " + str(sim_man.pulse_time[i]) + " us")
                print("Evolution algorithm (pulse #" + str(i) + ") = " + str(sim_man.evolution_algorithm[i]))
                print("nu_RRF (pulse #" + str(i) + ") = " + str(sim_man.RRF_par[i]['nu_RRF']) + " MHz")
                print("theta_RRF (pulse #" + str(i) + ") = " + str(sim_man.RRF_par[i]['theta_RRF']) + " rad")
                print("phi_RRF (pulse #" + str(i) + ") = " + str(sim_man.RRF_par[i]['phi_RRF']) + " rad")
                print('\n')

            print("Temperature = " + str(sim_man.temperature) + " K")
            print("Initial density matrix = " + str(sim_man.dm[0]))
            print("Decoherence time = " + str(sim_man.decoherence_time) + " us")

            # Panels showing the diagram of the density matrix evolved at each stage of the pulse
            # sequence
            self.evolved_dm_panels = Evolved_Density_Matrices(size_hint=(0.8, 0.6), pos=(80, 275), do_default_tab=False,
                                                              tab_width=150, tab_pos='top_mid', sim_man=sim_man)
            self.add_widget(self.evolved_dm_panels)

        except Exception as e:
            self.error_evolution = Label(text=e.args[0], pos=(210, 360), size=(200, 180), bold=True, color=(1, 0, 0, 1),
                                         font_size='15sp')
            self.add_widget(self.error_evolution)

            self.tb_evolution = Button(text='traceback', size_hint=(0.1, 0.03), pos=(640, 815))
            self.tb_evolution.bind(on_release=partial(print_traceback, e))
            self.add_widget(self.tb_evolution)

    def __init__(self, sim_man, **kwargs):
        super().__init__(**kwargs)

        self.evolved_state_label = Label(text='Evolved states', size=(10, 5), pos=(0, 450), font_size='30sp')
        self.add_widget(self.evolved_state_label)

        # Button which launches the evolution
        self.launch_evo_btn = Button(text='Launch evolution', font_size='16sp', bold=True, background_normal='',
                                     background_color=(0, 0.2, 1, 1), size_hint_y=None, height=35, size_hint_x=None,
                                     width=160, pos=(560, 875))
        self.launch_evo_btn.bind(on_press=partial(self.launch_evolution, sim_man))
        self.add_widget(self.launch_evo_btn)


# Class for the panels showing the density matrix evolved at each stage of the pulse (to be embedded
# inside the Evolve main panel)
class Evolved_Density_Matrices(TabbedPanel):

    def __init__(self, sim_man, **kwargs):
        super().__init__(**kwargs)

        self.pulse_tab = np.ndarray(sim_man.n_pulses, dtype=TabbedPanelItem)
        self.evolved_dm_box = np.ndarray(sim_man.n_pulses, dtype=BoxLayout)
        self.evolved_dm_figure = np.ndarray(4, dtype=matplotlib.figure.Figure)

        for i in range(sim_man.n_pulses):
            plt.close(self.evolved_dm_figure[i])

            self.pulse_tab[i] = TabbedPanelItem(text='Pulse ' + str(i + 1))
            self.evolved_dm_box[i] = BoxLayout()

            self.evolved_dm_figure[i] = plot_real_part_density_matrix(sim_man.dm[i + 1], show=False)[0]

            self.evolved_dm_box[i].add_widget(FigureCanvasKivyAgg(self.evolved_dm_figure[i]))
            self.pulse_tab[i].add_widget(self.evolved_dm_box[i])
            self.add_widget(self.pulse_tab[i])


# Class of the page of the software dedicated to the acquisition and analysis of the FID signal
class NMR_Spectrum(FloatLayout):
    error_FID = Label()
    tb_FID = Button()

    FID = Widget()
    FID_figure = matplotlib.figure.Figure()

    error_fourier = Label()
    tb_fourier = Button()

    fourier_spectrum = Widget()
    fourier_spectrum_figure = matplotlib.figure.Figure()

    error_adj_phase = Label()
    tb_adj_phase = Button()

    # Assigns the values of the inputs connected with the acquisition of the FID to the corresponding
    # variables of the Simulation_Manager and generates the FID signal
    def generate_FID(self, sim_man, y_shift, *args):
        try:
            self.remove_widget(self.error_FID)
            self.remove_widget(self.tb_FID)

            input_theta = (float(null_string(self.coil_theta.text)) / 180) * np.pi

            input_phi = (float(null_string(self.coil_phi.text)) / 180) * np.pi

            input_time_aq = float(null_string(self.time_aq.text))

            input_n_points = float(null_string(self.sample_points.text))

            sim_man.FID_times, sim_man.FID = FID_signal(sim_man.spin, sim_man.h_unperturbed, \
                                                        sim_man.dm[sim_man.n_pulses], \
                                                        acquisition_time=input_time_aq, \
                                                        T2=sim_man.decoherence_time, \
                                                        theta=input_theta, \
                                                        phi=input_phi, \
                                                        n_points=input_n_points)

        except Exception as e:
            self.error_FID = Label(text=e.args[0], pos=(175, y_shift + 360), size=(200, 205), bold=True,
                                   color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_FID)

            self.tb_FID = Button(text='traceback', size_hint=(0.1, 0.03), pos=(670, y_shift + 815))
            self.tb_FID.bind(on_release=partial(print_traceback, e))
            self.add_widget(self.tb_FID)

    def plot_fourier(self, sim_man):
        self.remove_widget(self.fourier_spectrum)
        plt.close(self.fourier_spectrum_figure)

        if self.input_opposite_frequency == False:
            plot_fourier_transform(sim_man.spectrum_frequencies, sim_man.spectrum_fourier, \
                                   square_modulus=self.input_square_modulus, show=False)
        else:
            plot_fourier_transform(sim_man.spectrum_frequencies, sim_man.spectrum_fourier, \
                                   sim_man.spectrum_fourier_neg, \
                                   square_modulus=self.input_square_modulus, show=False)

        self.fourier_spectrum = BoxLayout(size_hint=(0.9, 0.5), pos=(40, 0))

        self.fourier_spectrum_figure = plt.gcf()

        self.fourier_spectrum.add_widget(FigureCanvasKivyAgg(self.fourier_spectrum_figure))
        self.add_widget(self.fourier_spectrum)

    # Assigns the values of the inputs connected with the analysis of the FID to the corresponding
    # variables of the Simulation_Manager and generates its Fourier spectrum
    def generate_fourier(self, sim_man, y_shift, *args):
        try:
            self.remove_widget(self.error_fourier)
            self.remove_widget(self.tb_fourier)

            self.input_opposite_frequency = self.flip_negative_freq_checkbox.active
            self.input_square_modulus = self.sq_mod_checkbox.active

            self.input_frequency_left_bound = float(null_string(self.frequency_left_bound.text))
            self.input_frequency_right_bound = float(null_string(self.frequency_right_bound.text))

            if self.input_opposite_frequency == False:
                sim_man.spectrum_frequencies, \
                    sim_man.spectrum_fourier = fourier_transform_signal(sim_man.FID_times, \
                                                                        sim_man.FID, \
                                                                        frequency_start=self.input_frequency_left_bound, \
                                                                        frequency_stop=self.input_frequency_right_bound)
            else:
                sim_man.spectrum_frequencies, \
                    sim_man.spectrum_fourier, \
                    sim_man.spectrum_fourier_neg = fourier_transform_signal(sim_man.FID_times, \
                                                                            sim_man.FID, \
                                                                            frequency_start=self.input_frequency_left_bound, \
                                                                            frequency_stop=self.input_frequency_right_bound, \
                                                                            opposite_frequency=self.input_opposite_frequency)

            self.plot_fourier(sim_man)

        except Exception as e:
            self.error_fourier = Label(text=e.args[0], pos=(220, y_shift + 200), size=(200, 205), bold=True,
                                       color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_fourier)

            self.tb_fourier = Button(text='traceback', size_hint=(0.1, 0.03), pos=(670, y_shift + 655))
            self.tb_fourier.bind(on_release=partial(print_traceback, e))
            self.add_widget(self.tb_fourier)

    # Corrects the phase of the NMR spectrum at the given peak frequency
    def adjust_phase(self, sim_man, y_shift, *args):
        try:
            self.remove_widget(self.error_adj_phase)
            self.remove_widget(self.tb_adj_phase)

            peak_frequency = float(null_string(self.peak_frequency.text))

            search_window = float(null_string(self.int_domain_width.text))

            if not self.flip_negative_freq_checkbox.active:
                phi = fourier_phase_shift(sim_man.spectrum_frequencies, sim_man.spectrum_fourier, \
                                          peak_frequency=peak_frequency, \
                                          int_domain_width=search_window)
            else:
                phi = fourier_phase_shift(sim_man.spectrum_frequencies, sim_man.spectrum_fourier, \
                                          sim_man.spectrum_fourier_neg, \
                                          peak_frequency, search_window)

            sim_man.spectrum_fourier = np.exp(1j * phi) * sim_man.spectrum_fourier

            self.plot_fourier(sim_man)

        except Exception as e:
            self.error_adj_phase = Label(text=e.args[0], pos=(225, y_shift + 45), size=(100, 100), bold=True,
                                         color=(1, 0, 0, 1), font_size='15sp')
            self.add_widget(self.error_adj_phase)

            self.tb_adj_phase = Button(text='traceback', size_hint=(0.1, 0.03), pos=(670, y_shift + 500))
            self.tb_adj_phase.bind(on_release=partial(print_traceback, e))
            self.add_widget(self.tb_adj_phase)

    def FID_parameters(self, sim_man, y_shift):

        self.FID_parameters_label = Label(text='Acquisition of the FID', size=(10, 5), pos=(-250, y_shift + 400),
                                          font_size='20sp')
        self.add_widget(self.FID_parameters_label)

        # Orientation of the detection coils

        self.coil_orientation_label = Label(text='Normal to the plane of the detection coils', size=(10, 5),
                                            pos=(-207.5, y_shift + 360), font_size='15sp')
        self.add_widget(self.coil_orientation_label)

        self.coil_theta_label = Label(text='\N{GREEK SMALL LETTER THETA}', size=(10, 5), pos=(-342.5, y_shift + 320),
                                      font_size='15sp')
        self.add_widget(self.coil_theta_label)

        self.coil_theta = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(65, y_shift + 805))
        self.add_widget(self.coil_theta)

        self.coil_theta_unit = Label(text='°', size=(10, 5), pos=(-265, y_shift + 320), font_size='15sp')
        self.add_widget(self.coil_theta_unit)

        self.coil_phi_label = Label(text='\N{GREEK SMALL LETTER PHI}', size=(10, 5), pos=(-230, y_shift + 320),
                                    font_size='15sp')
        self.add_widget(self.coil_phi_label)

        self.coil_phi = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(180, y_shift + 805))
        self.add_widget(self.coil_phi)

        self.coil_phi_unit = Label(text='°', size=(10, 5), pos=(-152.5, y_shift + 320), font_size='15sp')
        self.add_widget(self.coil_phi_unit)

        # Time of acquisition of the FID Signal

        self.time_aq_label = Label(text='Time of acquisition of the FID signal', size=(10, 5),
                                   pos=(-227.5, y_shift + 280), font_size='15sp')
        self.add_widget(self.time_aq_label)

        self.time_aq = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(305, y_shift + 765))
        self.add_widget(self.time_aq)

        self.time_aq_unit = Label(text='\N{GREEK SMALL LETTER MU}s', size=(10, 5), pos=(-17.5, y_shift + 280),
                                  font_size='15sp')
        self.add_widget(self.time_aq_unit)

        # Number of sampling points

        self.sample_points_label = Label(text='#points/\N{GREEK SMALL LETTER MU}s', size=(10, 5),
                                         pos=(50, y_shift + 280), font_size='15sp')
        self.add_widget(self.sample_points_label)

        self.sample_points = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(495, y_shift + 765))
        self.sample_points.text = '10'
        self.add_widget(self.sample_points)

        # Question mark connected with the explanation of the input right above
        self.sample_question_mark = Label(text='[ref=?]?[/ref]', markup=True, size=(20, 20), pos=(175, y_shift + 280),
                                          font_size='20sp')
        self.add_widget(self.sample_question_mark)

        # Popup message which explains the meaning of the input above
        explanation = "The number of points per microsecond in which " + '\n' + \
                      "the time interval of acquisition is to be sampled " + '\n' + \
                      "is a fundamental parameter for the correct " + '\n' + \
                      "reproduction of the FID and its Fourier spectrum." + '\n' + \
                      "Indeed, according to Nyquist sampling theorem, the " + '\n' + \
                      "sampling frequency must be at least twice the " + '\n' + \
                      "maximum resonance frequency in the spectrum in " + '\n' + \
                      "order for the continuous FID signal to be sampled " + '\n' + \
                      "correctly. For this reason, few sampling points may " + '\n' + \
                      "lead to erroneous peaks in the Fourier spectrum."
        self.sample_popup = Popup(title='Nyquist theorem', content=Label(text=explanation), size_hint=(0.475, 0.45),
                                  pos=(425, 500), auto_dismiss=True)

        self.sample_question_mark.bind(on_ref_press=self.sample_popup.open)

        self.generate_FID_btn = Button(text='Acquire FID signal', font_size='16sp', size_hint_y=None, height=35,
                                       size_hint_x=None, width=200, pos=(550, 875 + y_shift))
        self.generate_FID_btn.bind(on_press=partial(self.generate_FID, sim_man, y_shift))
        self.add_widget(self.generate_FID_btn)

    def fourier_parameters(self, sim_man, y_shift):

        self.fourier_parameters_label = Label(text='Fourier analysis', size=(10, 5), pos=(-280, y_shift + 235),
                                              font_size='20sp')
        self.add_widget(self.fourier_parameters_label)

        # Checkbox which specifies if the generated NMR spectrum shows the frequencies in both the
        # negative and positive real axis of the same graph or in two separate graphs corresponding
        # to the clockwise and counter-clockwise precession

        self.flip_negative_freq_space = GridLayout(cols=2, size=(1000, 35), size_hint=(None, None),
                                                   pos=(50, y_shift + 675))
        self.flip_negative_freq_checkbox = CheckBox(size_hint_x=None, width=20)
        self.flip_negative_freq_space.add_widget(self.flip_negative_freq_checkbox)

        self.flip_negative_freq_label = Label(text='Separate plots for counter-rotating Fourier components',
                                              font_size='15sp', size_hint_x=None, width=395)
        self.flip_negative_freq_space.add_widget(self.flip_negative_freq_label)

        self.add_widget(self.flip_negative_freq_space)

        # Checkbox which specifies if the generated NMR spectrum displays the separate
        # real and imaginary parts or the square modulus of the complex signal

        self.sq_mod_space = GridLayout(cols=2, size=(1000, 35), size_hint=(None, None), pos=(50, y_shift + 640))
        self.sq_mod_checkbox = CheckBox(size_hint_x=None, width=20)
        self.sq_mod_space.add_widget(self.sq_mod_checkbox)

        self.sq_mod_label = Label(text='Square modulus of the NMR/NQR spectrum', font_size='15sp', size_hint_x=None,
                                  width=310)
        self.sq_mod_space.add_widget(self.sq_mod_label)

        self.add_widget(self.sq_mod_space)

        # Controls for the left and right bounds of the frequency domain where the NMR spectrum is to 
        # be plotted

        self.NMR_domain = Label(text='Frequency domain of the NMR/NQR spectrum', size=(10, 5),
                                pos=(-195, y_shift + 120), font_size='15sp')
        self.add_widget(self.NMR_domain)

        self.frequency_left_bound = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(365, 605 + y_shift))
        self.add_widget(self.frequency_left_bound)

        self.left_to_right = Label(text='-', size=(10, 5), pos=(30, y_shift + 120), font_size='15sp')
        self.add_widget(self.left_to_right)

        self.frequency_right_bound = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(435, 605 + y_shift))
        self.add_widget(self.frequency_right_bound)

        self.frequency_bounds_unit = Label(text='MHz', size=(10, 5), pos=(115, y_shift + 120), font_size='15sp')
        self.add_widget(self.frequency_bounds_unit)

        self.generate_fourier_btn = Button(text='Perform Fourier analysis', font_size='16sp', size_hint_y=None,
                                           height=35, size_hint_x=None, width=225, pos=(525, 715 + y_shift))
        self.generate_fourier_btn.bind(on_press=partial(self.generate_fourier, sim_man, y_shift))
        self.add_widget(self.generate_fourier_btn)

    # Set of controls for the phase adjustment of the NMR spectrum
    def adjustment_controls(self, sim_man, y_shift):

        self.phase_adjustment_label = Label(text='Phase adjustment', size=(10, 5), pos=(-270, y_shift + 85),
                                            font_size='20sp')
        self.add_widget(self.phase_adjustment_label)

        self.peak_frequency_label = Label(text='Peak frequency', size=(10, 5), pos=(-296, y_shift + 50),
                                          font_size='15sp')
        self.add_widget(self.peak_frequency_label)

        self.peak_frequency = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(165, y_shift + 535))
        self.add_widget(self.peak_frequency)

        self.peak_frequency_unit = Label(text='MHz', size=(10, 5), pos=(-150, y_shift + 50), font_size='15sp')
        self.add_widget(self.peak_frequency_unit)

        self.int_domain_width_label = Label(text='Width of the domain of integration', size=(10, 5),
                                            pos=(0, y_shift + 50), font_size='15sp')
        self.add_widget(self.int_domain_width_label)

        self.int_domain_width = TextInput(multiline=False, size_hint=(0.075, 0.03), pos=(525, y_shift + 535))
        self.add_widget(self.int_domain_width)

        self.int_domain_width_unit = Label(text='MHz', size=(10, 5), pos=(210, y_shift + 50), font_size='15sp')
        self.add_widget(self.int_domain_width_unit)

        self.phase_adj_btn = Button(text='Adjust phase', font_size='16sp', size_hint_y=None, height=35,
                                    size_hint_x=None, width=110, pos=(640, y_shift + 560))
        self.phase_adj_btn.bind(on_press=partial(self.adjust_phase, sim_man, y_shift))
        self.add_widget(self.phase_adj_btn)

    def __init__(self, save_config_btn, save_config_name, sim_man, **kwargs):
        super().__init__(**kwargs)

        self.NMR_spectrum_label = Label(text='NMR/NQR spectrum', size=(10, 5), pos=(0, 450), font_size='30sp')
        self.add_widget(self.NMR_spectrum_label)

        # Button and TextInput which allow to save all the inputs written throughout the program in a JSON file
        self.add_widget(save_config_btn)
        self.add_widget(save_config_name)

        self.FID_parameters(sim_man, 0)

        self.fourier_parameters(sim_man, 7.5)

        self.adjustment_controls(sim_man, 0)


# Class of the object on top of the individual panels
class Panels(TabbedPanel):
    error_retrieve_config = Label()
    error_save_config = Label()

    def retrieve_config(self, sim_man, *args):

        p1 = self.sys_par

        p2 = self.pulse_par

        p4 = self.spectrum_page

        try:

            p1.remove_widget(self.error_retrieve_config)

            with open(self.retrieve_config_name.text) as config_file:
                configuration = json.load(config_file)

            p1.spin_qn.text = configuration['spin_par']['quantum number']

            p1.gyro.text = configuration['spin_par']['gamma/2pi']

            p1.spin_qn2.text = configuration['spin_par2']['quantum number']

            p1.gyro2.text = configuration['spin_par2']['gamma/2pi']

            p1.field_mag.text = configuration['zeem_par']['field magnitude']

            p1.theta_z.text = configuration['zeem_par']['theta_z']

            p1.phi_z.text = configuration['zeem_par']['phi_z']

            p1.coupling.text = configuration['quad_par']['coupling constant']

            p1.asymmetry.text = configuration['quad_par']['asymmetry parameter']

            p1.alpha_q.text = configuration['quad_par']['alpha_q']

            p1.beta_q.text = configuration['quad_par']['beta_q']

            p1.gamma_q.text = configuration['quad_par']['gamma_q']

            p1.coupling2.text = configuration['quad_par2']['coupling constant']

            p1.asymmetry2.text = configuration['quad_par2']['asymmetry parameter']

            p1.alpha_q2.text = configuration['quad_par2']['alpha_q']

            p1.beta_q2.text = configuration['quad_par2']['beta_q']

            p1.gamma_q2.text = configuration['quad_par2']['gamma_q']

            p1.chemical_shift.text = configuration['cs_par']['delta_iso']

            p1.bd1.text = configuration['D1_par']['b_D']

            p1.bdtheta1.text = configuration['D1_par']['theta']

            p1.bd2.text = configuration['D2_par']['b_D']

            p1.bdtheta2.text = configuration['D2_par']['theta']

            p1.hfA.text = configuration['hf_par']['A']

            p1.hfB.text = configuration['hf_par']['B']

            p1.Jparam.text = configuration['J_sec_par']['J']

            p1.remove_widget(p1.nu_q_label)

            p1.decoherence.text = configuration['decoherence time']

            p1.temperature.text = configuration['temperature']

            p1.canonical_checkbox.active = configuration['initial state at equilibrium']

            if len(configuration['manual initial density matrix']) > 1:

                p1.set_dm_grid(-5)

                for i in range(p1.d):
                    for j in range(p1.d):
                        p1.dm_elements[i, j].text = configuration['manual initial density matrix'][str(i) + str(j)]

            p2.number_pulses.text = configuration['n_pulses']

            p2.set_pulse_controls(self, sim_man, *args)

            for n in range(int(null_string(p2.number_pulses.text))):

                p2.pulse_times[n].text = configuration['pulse #' + str(n + 1) + ' time']

                if p2.n_modes[n] < int(configuration['n_modes #' + str(n + 1)]):
                    p2.add_new_mode(n)
                elif p2.n_modes[n] > int(configuration['n_modes #' + str(n + 1)]):
                    p2.remove_mode(n, sim_man)

                p2.n_modes[n] = configuration['n_modes #' + str(n + 1)]

                for m in range(p2.n_modes[n]):
                    p2.frequency[n][m].text = str(
                        configuration['pulse #' + str(n + 1)]['mode #' + str(m + 1)]['frequency'])

                    p2.amplitude[n][m].text = str(
                        configuration['pulse #' + str(n + 1)]['mode #' + str(m + 1)]['amplitude'])

                    p2.phase[n][m].text = str(configuration['pulse #' + str(n + 1)]['mode #' + str(m + 1)]['phase'])

                    p2.theta_p[n][m].text = str(configuration['pulse #' + str(n + 1)]['mode #' + str(m + 1)]['theta_p'])

                    p2.phi_p[n][m].text = str(configuration['pulse #' + str(n + 1)]['mode #' + str(m + 1)]['phi_p'])

                if configuration['pulse #' + str(n + 1) + ' evolution algorithm'] == "RRF":

                    p2.RRF_btn[n].state = 'down'

                    p2.set_RRF_evolution(n, y_shift=400 - n * 200, sim_man=sim_man)

                    p2.RRF_frequency[n].text = str(configuration['pulse #' + str(n + 1) + ' RRF parameters']['nu_RRF'])
                    p2.RRF_theta[n].text = str(configuration['pulse #' + str(n + 1) + ' RRF parameters']['theta_RRF'])
                    p2.RRF_phi[n].text = str(configuration['pulse #' + str(n + 1) + ' RRF parameters']['phi_RRF'])

                elif configuration['pulse #' + str(n + 1) + ' evolution algorithm'] == "IP":

                    p2.IP_btn[n].state = 'down'

                    p2.set_IP_evolution(n, y_shift=400 - n * 200, sim_man=sim_man)

            p4.coil_theta.text = configuration['coil_theta']

            p4.coil_phi.text = configuration['coil_phi']

            p4.time_aq.text = configuration['acquisition time']

            p4.sample_points.text = configuration['#points/us']

            p4.flip_negative_freq_checkbox.active = configuration['plot for opposite frequencies']

            p4.sq_mod_checkbox.active = configuration['plot square modulus']

            p4.frequency_left_bound.text = configuration['frequency domain left bound']
            p4.frequency_right_bound.text = configuration['frequency domain right bound']

            p4.peak_frequency.text = configuration['peak frequency to be adjusted']

            p4.int_domain_width.text = configuration['integration domain width']

        except Exception as e:

            kind_of_error = len(e.args) - 1

            self.error_retrieve_config = Label(text=str(e.args[kind_of_error]), pos=(254, 405), size=(100, 100),
                                               bold=True, color=(1, 0, 0, 1), font_size='15sp')
            p1.add_widget(self.error_retrieve_config)

    def save_config(self, sim_man, *args):

        p1 = self.sys_par

        p2 = self.pulse_par

        p4 = self.spectrum_page

        try:

            p4.remove_widget(self.error_save_config)

            configuration = {}

            configuration['spin_par'] = {'quantum number': p1.spin_qn.text, \
                                         'gamma/2pi': p1.gyro.text}

            configuration['spin_par2'] = {'quantum number': p1.spin_qn2.text, \
                                          'gamma/2pi': p1.gyro2.text}

            configuration['zeem_par'] = {'field magnitude': p1.field_mag.text,
                                         'theta_z': p1.theta_z.text,
                                         'phi_z': p1.phi_z.text}

            configuration['quad_par'] = {'coupling constant': p1.coupling.text,
                                         'asymmetry parameter': p1.asymmetry.text,
                                         'alpha_q': p1.alpha_q.text,
                                         'beta_q': p1.beta_q.text,
                                         'gamma_q': p1.gamma_q.text}

            configuration['quad_par2'] = {'coupling constant': p1.coupling2.text,
                                          'asymmetry parameter': p1.asymmetry2.text,
                                          'alpha_q': p1.alpha_q2.text,
                                          'beta_q': p1.beta_q2.text,
                                          'gamma_q': p1.gamma_q2.text}

            configuration['cs_par'] = {'delta_iso': p1.chemical_shift.text}

            configuration['D1_par'] = {'b_D': p1.bd1.text,
                                       'theta': p1.bdtheta1.text}

            configuration['D2_par'] = {'b_D': p1.bd2.text,
                                       'theta': p1.bdtheta2.text}

            configuration['hf_par'] = {'A': p1.hfA.text,
                                       'B': p1.hfB.text}

            configuration['J_sec_par'] = {'J': p1.Jparam.text}

            configuration['decoherence time'] = p1.decoherence.text

            configuration['initial state at equilibrium'] = sim_man.canonical_dm_0

            configuration['temperature'] = p1.temperature.text

            configuration['manual initial density matrix'] = {}

            for i in range(p1.dm_elements.shape[0]):
                for j in range(p1.dm_elements.shape[1]):
                    configuration['manual initial density matrix'][str(i) + str(j)] = p1.dm_elements[i, j].text

            if len(configuration['manual initial density matrix']) > 1:
                if p1.d != int(2 * float(configuration['spin_par']['quantum number']) + 1):
                    raise IndexError(
                        "The dimensions of the grid for the initial state" + '\n' + "don't match the spin states' multiplicity")

            configuration['n_pulses'] = p2.number_pulses.text

            for n in range(int(null_string(p2.number_pulses.text))):

                configuration['pulse #' + str(n + 1) + ' time'] = p2.pulse_times[n].text

                configuration['n_modes #' + str(n + 1)] = str(p2.n_modes[n])

                configuration['pulse #' + str(n + 1)] = {}

                for m in range(p2.n_modes[n]):
                    configuration['pulse #' + str(n + 1)]['mode #' + str(m + 1)] = \
                        {'frequency': p2.frequency[n][m].text,
                         'amplitude': p2.amplitude[n][m].text,
                         'phase': p2.phase[n][m].text,
                         'theta_p': p2.theta_p[n][m].text,
                         'phi_p': p2.phi_p[n][m].text}

                if p2.RRF_btn[n].state == 'down':
                    configuration['pulse #' + str(n + 1) + ' evolution algorithm'] = "RRF"

                    configuration['pulse #' + str(n + 1) + ' RRF parameters'] = \
                        {'nu_RRF': p2.RRF_frequency[n].text, \
                         'theta_RRF': p2.RRF_theta[n].text, \
                         'phi_RRF': p2.RRF_phi[n].text}

                else:
                    configuration['pulse #' + str(n + 1) + ' evolution algorithm'] = "IP"

            configuration['coil_theta'] = p4.coil_theta.text

            configuration['coil_phi'] = p4.coil_phi.text

            configuration['acquisition time'] = p4.time_aq.text

            configuration['#points/us'] = p4.sample_points.text

            configuration['plot for opposite frequencies'] = p4.flip_negative_freq_checkbox.active

            configuration['plot square modulus'] = p4.sq_mod_checkbox.active

            configuration['frequency domain left bound'] = p4.frequency_left_bound.text

            configuration['frequency domain right bound'] = p4.frequency_right_bound.text

            configuration['peak frequency to be adjusted'] = p4.peak_frequency.text

            configuration['integration domain width'] = p4.int_domain_width.text

            with open(self.save_config_name.text, 'w') as config_file:
                json.dump(configuration, config_file)

        except Exception as e:

            kind_of_error = len(e.args) - 1

            self.error_save_config = Label(text=str(e.args[kind_of_error]), pos=(254, 405), size=(100, 100), bold=True,
                                           color=(1, 0, 0, 1), font_size='15sp')
            p4.add_widget(self.error_save_config)

    def __init__(self, sim_man, **kwargs):
        super().__init__(**kwargs)

        self.retrieve_config_btn = Button(text='Retrieve configuration', size_hint=(0.23, 0.03), pos=(610, 965),
                                          bold=True, background_color=(2.07, 0, 0.15, 1), font_size='15')

        self.retrieve_config_btn.bind(on_release=partial(self.retrieve_config, sim_man))

        self.retrieve_config_name = TextInput(multiline=False, size_hint=(0.23, 0.03), pos=(610, 935))

        self.tab_sys_par = TabbedPanelItem(text='System')
        self.scroll_window = ScrollView(size_hint=(1, None), size=(Window.width, 500))
        self.sys_par = System_Parameters(size_hint=(1, None), size=(Window.width, 1000), sim_man=sim_man,
                                         retrieve_config_btn=self.retrieve_config_btn,
                                         retrieve_config_name=self.retrieve_config_name)

        self.scroll_window.add_widget(self.sys_par)
        self.tab_sys_par.add_widget(self.scroll_window)
        self.add_widget(self.tab_sys_par)

        self.tab_pulse_par = TabbedPanelItem(text='Pulse')
        self.scroll_window = ScrollView(size_hint=(1, None), size=(Window.width, 500))
        self.pulse_par = Pulse_Sequence(size_hint=(1, None), size=(Window.width, 1000), sim_man=sim_man)

        self.scroll_window.add_widget(self.pulse_par)
        self.tab_pulse_par.add_widget(self.scroll_window)
        self.add_widget(self.tab_pulse_par)

        self.tab_evolve = TabbedPanelItem(text='Evolve')
        self.scroll_window = ScrollView(size_hint=(1, None), size=(Window.width, 500))
        self.evo_res_page = Evolution_Results(size_hint=(1, None), size=(Window.width, 1000), sim_man=sim_man)
        self.scroll_window.add_widget(self.evo_res_page)
        self.tab_evolve.add_widget(self.scroll_window)
        self.add_widget(self.tab_evolve)

        self.save_config_btn = Button(text='Save configuration', size_hint=(0.23, 0.03), pos=(565, 945), bold=True,
                                      background_color=(2.07, 0, 0.15, 1), font_size='15')
        self.save_config_btn.bind(on_release=partial(self.save_config, sim_man))

        self.save_config_name = TextInput(multiline=False, size_hint=(0.23, 0.03), pos=(565, 915))

        self.tab_spectrum = TabbedPanelItem(text='Spectrum')
        self.scroll_window = ScrollView(size_hint=(1, None), size=(Window.width, 500))
        self.spectrum_page = NMR_Spectrum(size_hint=(1, None), size=(Window.width, 1000),
                                          save_config_btn=self.save_config_btn, save_config_name=self.save_config_name,
                                          sim_man=sim_man)
        self.scroll_window.add_widget(self.spectrum_page)
        self.tab_spectrum.add_widget(self.scroll_window)
        self.add_widget(self.tab_spectrum)


# Class of the application and main class of the program
class PULSEE_CHEM(App):
    sim_man = Simulation_Manager()

    def build(self):
        return Panels(size=(500, 500), do_default_tab=False, tab_pos='top_mid', sim_man=self.sim_man)


if __name__ == '__main__':
    PULSEE_CHEM().run()
