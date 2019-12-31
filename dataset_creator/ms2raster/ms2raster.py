from subprocess import Popen, PIPE
import numpy as np
import argparse as ap
import os


def get_full_matrix(ms_out, length1):
	data = []
	new_data = []
	for line in ms_out:
		line = str(line.rstrip()).split("'")[1]
		#print(line)
		if line.startswith('/') or line == "":
			continue
		elif line.startswith('seg'):
			segsites = int(line.split()[1])
		elif line.startswith('pos'):
			positions = line.split()[1:segsites+1]
			positions = [round(float(i)*length1) for i in positions]
		elif line[0].isdigit() and ' ' in line:
			continue
		elif line[0].isdigit():
			data.append(list(line))
	if segsites == 0:
		return(0,0)

	else:
		transposed = zip(*data)
		for i in transposed:
			new_data.append(''.join(list(i)))
	
		samples = len(new_data[0])	
		full_list = []
		point = 0
		for i in range(0,len(positions)):
			pos = positions[i]
			for x in range(1,pos - point):
				full_list.append('0'*samples)
			if pos == point:
				point = pos+1
			else: 
				point = pos
			full_list.append(new_data[i])
		for x in range(1,length1-point+1):
			full_list.append('0'*samples)
	
		return full_list, positions


def writing_region(alist, asim):
	out = open(asim,'w')
	for y in alist:
		for i in y:
			out.write(str(i))
		out.write('\n')
	out.close()			

#######################################################
#to be fully argparsed#################################
#######################################################

parser = ap.ArgumentParser()
#parser.add_argument('-Ne', '--eff_pop', help='Effective population size', required=True, type=int)
parser.add_argument('-bp', '--length', help='Length of the locus to simulate in bp', required=True, type=int)
#parser.add_argument('-mu', '--mut_rate', help='Mutation rate', required=True, type=float)
#parser.add_argument('-ro', '--rec_rate', help='Recombination rate', required=True, type=float)
parser.add_argument('-s', '--simulations', help='Simulations to be performed', required=True, type=int)
parser.add_argument('-l', '--label', help='Label for these simulations: neutral, selection', required=True, type=str)
parser.add_argument('-selstr', '--selstrength', help='Selection strength', required=True, type=float)
parser.add_argument('-p', '--path', help='Path to the ms and mssel folder with all executables: e.g. /opt/software/ms_sim_software', required=True, type=str)
parser.add_argument('-i', '--individuals', help='Number of diploid individuals to simulate', required=True, type=int)

args = parser.parse_args()

simulations = args.simulations
label = args.label
path = args.path

sel = args.selstrength

Ne = 80000
Ne_mu = 0.0008
bp = args.length
Ne_ro = 0.0008
ind = args.individuals

ms_t = 4*Ne_mu*bp
ms_r = 4*Ne_ro*bp

segsites = []
for i in range(0,simulations):
	if label == 'neutral':
		pipe = Popen(path+"ms "+str(ind*2)+" 1 -t "+str(ms_t)+" -p "+str(int(np.log10(bp)))+" -r "+str(ms_r)+" "+str(bp+1), shell=True, stdout=PIPE)

	elif label == 'selection':
		os.system(path+"trajfixconst 1 "+str(sel)+" "+str(Ne)+" 135757 | "+path+"lastp0 .2 .02 | "+path+"stepftn >tp.out")
		pipe = Popen(path+"mssel "+str(ind*2)+" 1 0 48 tp.out 5000 -t "+str(ms_t)+" -p "+str(int(np.log10(bp)))+" -r "+str(ms_r)+" "+str(bp+1), shell=True, stdout=PIPE)
	else:
		print('Label is not among those allowed: neutral or selection')
	
	sim_data = get_full_matrix(pipe.stdout, bp)

	if sim_data[0] == 0:
		print('No segregation sites in simulation # ', str(i),': skipped')
	else:
		writing_region(sim_data[0], str(i+1)+'.'+label+'.sim')
		segsites.append(len(sim_data[1]))
		print('Simulation # ', str(i),'had one or more segregation sites and has been used')

		if len(sim_data[0]) != bp:
			print('Something went wrong with simulation: ', str(i), ' as the number of bp is not ', str(bp))

if len(segsites) == 0:
	print('None of the simulation had segregating sites. Try to increase bp or decrease selstrength')
else:
	print('Average segregating sites: ', np.mean(segsites))

