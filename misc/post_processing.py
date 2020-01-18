from showing import ploting
from misc import load_obj
# dict = load_obj("all_favourite_methods_x1000", resolution=0.7)
# dict = load_obj("exp-ep-greedy-Dolinar_x500", resolution=0.7)
name = "all_methods_x12_ep100"
dict = load_obj(name, resolution=0.1, layers=2)

# dict = load_obj("ep-greedy-Dolinar_x1", resolution=0.33)
# # for i in dict.keys():
# #     print(i, dict[i]["label"])
# interesting = ["run_10","run_16", "run_2"]
# interesting = ["run_1","run_2", "run_3", "run_4","run_5"]
interesting = ["run_1","run_2", "run_3","run_4","run_5"]
# interesting = dict.keys()
# #
dict_plot = {}
for i in interesting:
    dict_plot[i] = dict[i]
ploting(dict_plot,mode_log="off",save=True,show=True, particular_name=name,mode="stds")
