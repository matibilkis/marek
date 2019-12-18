import training
import gc

# methods = ["ep-greedy", "ucb","thompson-sampling"]
# soft_ucbs = [0.01,0.1,1,10]
# ucb_methods = ["ucb1", "ucb2", "ucb3"]
# efficiencies = [0.01, 0.1]
# pflips = [0.01, 0.3]
#
# for sm in methods:
#     for sfts in soft_ucbs:
#         for ucbm in ucb_methods:
#             for eff in efficiencies:
#                 for pflip in pflips:
#                     for mg in methods:
#
#
#                         a = training.Experiment(layers=2,number_phases=2,states_wasted=10**2, searching_method = sm, ucb_method=ucbm, method_guess=mg, pflip = pflip, efficiency = eff, soft_ts = sfts)
#                         a.train(2)
#                         del a
#                         gc.collect()

a = training.Experiment(searching_method = "ep-greedy", ep_method="exp-decay", time_tau=2000, min_ep=0.000001, method_guess="ucb", ucb_method="ucb3")
a.train(2)
