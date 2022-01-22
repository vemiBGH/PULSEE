import os 
import argparse
import json 
import numpy as np 

TEX_FILENAME = 'gen_slides.tex'
SIM_DIR = './test/'

INITIAL_DM_FILENAME = 'initial_dm.csv'
EVOLVED_DM_FILENAME = 'evolved_dm.csv'
INITIAL_DM_FIG_FILENAME = 'InitialRealPartDensityMatrix.png'
EVOLVED_DM_FIG_FILENAME = 'EvolvedRealPartDensityMatrix.png'
FID_FILENAME = 'FIDSignal.png'
FT_FILENAME = 'FTSignal.png'
PARAMS_FILENAME = 'params.json'

TEX_PREAMBLE = '''
\\documentclass[10pt]{beamer}

\\title{Simulation Results}
\\author{Ilija Nikolov, Lucas Z. Brito}
\\institute{}
\date{2021}
\setlength{\\arraycolsep}{1pt}
\setbeamersize{text margin left=0.4cm, text margin right=0.4cm}
\\begin{document}

\\frame{\\titlepage}
'''

def np2tex(a, env='pmatrix'): 
	assert len(np.shape(a)) == 2
	
	tex_array = f"\\begin{'{' + env + '}'}\n" 
	for row in a[:-1]: 
		for entry in row[:-1]: 
			tex_array += str(entry) + ' & '
		tex_array += str(row[-1]) + ' \\\\\n'
	for entry in a[-1][:-1]:
		tex_array += str(entry) + ' & '
	tex_array += str(a[-1][-1]) + f"\n\\end{'{' + env + '}'}"
	return tex_array

def mat_preproc(a): 
	return np.round(np.real(a), 3)

def make_slide(dir): 
		params_file = open(os.path.join(dir, PARAMS_FILENAME))
		params = json.load(params_file)
		params['B_0'] = 10 # hard coded, fix this when using updated sim files 

		# Process hamiltonians and convert to TeX
		H = '\\text{I}_{z}'
		addtn_args = ''
		if 'D2_param' in params['hamiltonian_args'].keys(): 
			H += '+ b_D \\big(3\\cos^2\\theta-1\\big)I_{1z}I_{2z}.'
			h_params = params['hamiltonian_args']['D2_param']
			for k in h_params.keys(): 
				addtn_args += f',${k}\\approx{np.round(h_params[k], 3)}$'
		
		if 'h_tensor_inter' in params['hamiltonian_args'].keys(): 
			H += '+ \\hat{S}\\tilde{A}\\hat{I}'
			h_params = params['hamiltonian_args']['h_tensor_inter']
			addtn_args += f'$,A={h_params}$,'
		

		params_str = f"Spin: {params['quantum_numbers']}," \
			+ f"$B_0= {params['B_0']}$, $\\gamma/2\\pi = {params['gamma_2pis']}$,"\
			+ f"$\\mathcal{{H}}={H}$" + addtn_args

		initial_dm_file = os.path.join(dir, INITIAL_DM_FILENAME)
		initial_dm = np.loadtxt(initial_dm_file, dtype=complex)
		initial_dm = mat_preproc(initial_dm)
		tex_initial_dm = np2tex(initial_dm)

		evolved_dm_file = os.path.join(dir, EVOLVED_DM_FILENAME)
		evolved_dm = np.loadtxt(evolved_dm_file, dtype=complex)
		evolved_dm = mat_preproc(evolved_dm)
		tex_evolved_dm = np2tex(evolved_dm)

		return f'''\\begin{{frame}}
\\frametitle{{{os.path.basename(dir).replace('_', ' ')}}}
{params_str}
\\begin{{columns}}[T]
\\begin{{column}}{{.5\\textwidth}}
\\tiny\\begin{{align*}}
\\rho_{{\\text{{initial}}}}\\doteq
{tex_initial_dm}
\\\\
\\rho_{{\\text{{final}}}}\\doteq
{tex_evolved_dm}
\\end{{align*}}
\\begin{{column}}{{.5\\textwidth}}
\\includegraphics[width=1.3\\textwidth]{{{os.path.join(dir, INITIAL_DM_FIG_FILENAME)}}}
\\end{{column}}
\\begin{{column}}{{.5\\textwidth}}
\\includegraphics[width=1.3\\textwidth]{{{os.path.join(dir, EVOLVED_DM_FIG_FILENAME)}}}
\\end{{column}}
\\end{{column}}
\\begin{{column}}{{.5\\textwidth}}
\includegraphics[width=\\textwidth]{{{os.path.join(dir, FID_FILENAME)}}}
\includegraphics[width=\\textwidth]{{{os.path.join(dir, FT_FILENAME)}}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}
'''

def main(sim_dir, tex_filename):
	tex = TEX_PREAMBLE
	for dir in os.listdir(sim_dir): 
		if not os.path.isdir(os.path.join(sim_dir, dir)):
			continue  
		
		tex += make_slide(os.path.join(sim_dir, dir))
	
	tex += '\n\\end{document}'
	f = open(tex_filename, 'w')
	f.write(tex)
	f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert the given directory' \
										+ 'of simulation results into a TeX' \
										+ 'Beamer file.')

	parser.add_argument('dir')
	parser.add_argument('--name', default=TEX_FILENAME)
	args = parser.parse_args()
	main(args.dir, args.name)
		