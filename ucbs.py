import training
import gc
from showing import ploting
import pickle
from misc import save_obj, filter_keys
from main import MegaFrontEnd
import numpy as np

mega = MegaFrontEnd(layers=2, guessing_rule="None")
mega.ucbs(total_episodes=10**5, bob=48)
# mega.ucb_single(10**2, bob=1)
# mega.run_darkcounts(total_episodes=10**5, bobs=48)

# bob=48
# # #
# exper = training.Experiment(searching_method = "ucb", layers=2, ucb_method="ucb1" , resolution=.1, bound_displacements=1,  states_wasted=10**4, guessing_rule="None", efficient_time=False)
# exper.train(bob)
# #
# exper = training.Experiment(searching_method = "ucb", layers=2, ucb_method="ucb2" , resolution=.1, bound_displacements=1,  states_wasted=10**4, guessing_rule="None", efficient_time=False)
# exper.train(bob)
#
# exper = training.Experiment(searching_method = method, layers=self.layers, ucb_method=ucbm , resolution=self.resolution, bound_displacements=self.bound_displacements,  states_wasted=total_episodes, guessing_rule=self.guessing_rule, efficient_time=False)

# exper = training.Experiment(searching_method = "ep-greedy", layers=2, ucb_method="ucb2" , resolution=.1, bound_displacements=1,  states_wasted=1000, guessing_rule="None", efficient_time=False)
# exper.train(bob)
