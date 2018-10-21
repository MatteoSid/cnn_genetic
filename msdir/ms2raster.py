from subprocess import Popen, PIPE
import numpy as np
import argparse as ap


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
# to be fully argparsed ###############################
#######################################################

parser = ap.ArgumentParser()
#parser.add_argument('-Ne', '--eff_pop', help='Effective population size', required=True, type=int)
parser.add_argument('-bp', '--length', help='Length of the locus to simulate in bp', required=True, type=int)
#parser.add_argument('-mu', '--mut_rate', help='Mutation rate', required=True, type=float)
#parser.add_argument('-ro', '--rec_rate', help='Recombination rate', required=True, type=float)
parser.add_argument('-s', '--simulations', help='Simulations to be performed', required=True, type=int)
parser.add_argument('-l', '--label', help='Label for these simulations', required=True, type=str)
parser.add_argument('-p', '--path', help='Path to the ms executable. E.g. ~/software/ms.folder/msdir/ms', required=True, type=str)
parser.add_argument('-i', '--individuals', help='Number of diploid individuals to simulate', required=True, type=int)

args = parser.parse_args()

simulations = args.simulations
label = args.label
ms = args.path

Ne_mu = 0.0008
bp = args.length
Ne_ro = 0.0008
ind = args.individuals

ms_t = 4*Ne_mu*bp
ms_r = 4*Ne_ro*bp

segsites = []
for i in range(0,simulations):
	pipe = Popen(ms+" "+str(ind*2)+" 1 -t "+str(ms_t)+" -p "+str(int(np.log10(bp)))+" -r "+str(ms_r)+" "+str(bp+1), shell=True, stdout=PIPE)
	sim_data = get_full_matrix(pipe.stdout, bp)
	writing_region(sim_data[0], str(i+1)+'.'+label+'.sim')
	segsites.append(len(sim_data[1]))
	if len(sim_data[0]) != bp:
		print('Something went wrong with simulation: ', str(i), ' as the number of bp is not ', str(bp))
print('Average segregating sites: ', np.mean(segsites))





