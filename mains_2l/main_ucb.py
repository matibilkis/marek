import training
import gc
from showing import ploting
import pickle
#In this program we compute the learning curves for the bandit problem in the Kennedy case, taking Dolinar's guessing rule.

#Each arm is a possible displacement.

def save_obj(obj, name ):
    with open('dicts/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



dict={}
method = "ucb"

for ucbm in ["ucb1", "ucb2", "ucb3"]:

    exper = training.Experiment(searching_method = method, layers=2, ucb_method=ucbm ,resolution=0.1, bound_displacements=1, states_wasted=10**2, guessing_rule="None")
    exper.train(4)

    with open(str(exper.layers)+"L"+str(exper.number_phases)+"PH"+str(exper.resolution)+"R/number_rune.txt", "r") as f:
        c = f.readlines()[0]
        f.close()

    dict["run_"+str(c)] = {}
    dict["run_"+str(c)]["label"] = ucbm
    dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs]
    dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
    dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]

save_obj(dict, "2l_ucb")
ploting(dict, mode="minimax")
ploting(dict, mode="stds")
