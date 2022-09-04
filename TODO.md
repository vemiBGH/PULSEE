### Project status

- [ ] Check the demo for hyperfine (small errors) 
- [x] Add xlim and ylim for the plot_fourier_transform function
- [x] Add normalize for plot_fourier_transform function
- [x] Show option of for plot_fourier_transform function does not do anything (it works perfectly fine for me - Lucas)
- [x] (hyperfine demos use `plot_fourier_transform` and the plots in `sim_vesna.py` cannot be made with `plot_fourier_transform` - Lucas) Change the demo plots to use plot_fourier_transform 
- [ ] Fix the initial demo on the main page (initially made by Davide)
- [x] Fix demos (3 methods should agree) & there are errors
- [x] Don't return every state ed_evolve, return only final state unless wanted
- [x] IMPORTANT Write examples of how to pass $T_2$ decay function
- For the direct diagonalization ed_evolve:
  - [x] option to only return expected value of operator of ed_evolve
  - [x] returns twice the fid?
  - [x] no T2 in the FID

- [ ] CRITICAL GUI qt error
QObject::moveToThread: Current thread (0x239c69f27d0) is not the object's thread (0x239c69f3f30).
Cannot move to target thread (0x239c69f27d0) 
qt.qpa.plugin: Could not load the Qt platform plugin "windows" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
