- [ ] Add zero-field NQR example
- [ ] Fix ipyparallel for Jupyter notebook

Jiwoo: 
- [ ] simulate CNOT on ADP
- [ ] Test, QA

Ilija
- [ ] Check magnus higher order integral approx
- [ ] Add notebook selective pulse

Alex
- [ ] Arbitrary pulse shape (not just square pulse)

### General Features
- [ ] Gradient pulses & Arbitrary shape pulses
- [ ] GRAPE
- [ ] Automatically find the \pi pulse
- [ ] Simulate dissipation?
- [ ] Noise spectroscopy  
- [ ] Add a function that generates CNOT gate based on some interaction? (Is it possible)? Not theoretically, but using pulses...
- [ ] Pulse diagram after pulses are sent to sim.evolve
- [ ] Fix GUI with most recent changes
- [ ] FID parallelizable

### Project status
- [x] Replace all use of the plotting functions with pulsee.plot instead of pulsee.simulation
- [x] Update `plot_complex_density_matrix` and `plot_real_part_density_matrix` (fix element labels & optional arguments to rotate the plot)
- [x] Replace pd dataframe (mode) with a new `Pulses` class. Should simplify the code and remove dependency on pandas package
- [x] Add type hints to simulation.py
- [x] Update poetry files for package management
- [x] Replace the use of matplotlib.pylab (strongly discouraged by matplotlib documentation) with matplotlib.pyplot
- [x] Time dependent Hamiltonians to simulate FID and add option to add pulses during FID acquisition
- [x] Add notebook simple 1/2 case
- [x] Check the demo for hyperfine (small errors) 
- [x] Add xlim and ylim for the plot_fourier_transform function
- [x] Add normalize for plot_fourier_transform function
- [x] Show option of for plot_fourier_transform function does not do anything (it works perfectly fine for me - Lucas)
- [x] (hyperfine demos use `plot_fourier_transform` and the plots in `sim_vesna.py` cannot be made with `plot_fourier_transform` - Lucas) Change the demo plots to use plot_fourier_transform 
- [x] Fix the initial demo on the main page (initially made by Davide)
- [x] Fix demos (3 methods should agree) & there are errors
- [x] Don't return every state ed_evolve, return only final state unless wanted
- [x] IMPORTANT Write examples of how to pass $T_2$ decay function
- For the direct diagonalization ed_evolve:
  - [x] option to only return expected value of operator of ed_evolve
  - [x] returns twice the fid?
  - [x] no T2 in the FID

- [x] CRITICAL GUI
