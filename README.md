# PULSEE (Program for the simULation of nuclear Spin Ensemble Evolution)
## Development branch for Quantum Computing Module
Authors: Davide Candoli (Università di Bologna) and Lucas Brito (Brown University)

## Installation
To use this development version of PULSEE, first install the dependencies as 
outlined by `requirements.txt`: 
```
>>> pip install -r requirements.txt
```

or, with Anaconda: 
```
conda install --file conda-requirements.txt
```

Navigate to the directory containing `setup.py` (this should simply be PULSEE), 
and run `pip install -e .` to perform a development installation of the package. 
This lets your environment know that you will continue to edit the source code 
of the package, so changes made to any files will be trickled down to any 
imports. Enjoy! 

## Demos
See `/demos` for some demonstrations of using PULSEE (as Python scripts and 
Jupyter notebooks).