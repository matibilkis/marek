import os
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--amplitude", type=float, default=0.4)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--maxiter", type=int, default=2000)
args = parser.parse_args()

string = "python3 optimization.py --amplitude {} --layers {} --maxiter {}".format(args.amplitude, args.layers, args.maxiter)
os.system(string)
