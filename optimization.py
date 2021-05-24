from basics import Basics
import numpy as np
import os
from scipy import optimize
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--amplitude", type=float, default=0.4)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--maxiter", type=int, default=2000)

args = parser.parse_args()
if args.amplitude<1:
    blim = 2
else:
    blim = 4
basi = Basics(amplitude = args.amplitude, epsilon = 0.01)
fun = [basi.success_probability_1L, basi.success_probability_2L, basi.success_probability_3L][args.layers-1]

path="/data/uab-giq/scratch/matias/compound/brute_force/"
# path="compound/brute_force/"
name = path+"{}L{}a".format(args.layers, args.amplitude)
os.makedirs(name, exist_ok=True)

optimized_object=optimize.dual_annealing(fun,([(-blim,blim)])*3, maxiter=args.maxiter,no_local_search=True )
np.save(name+"/psuc", -optimized_object.fun)
np.save(name+"/optvals", optimized_object.x)
