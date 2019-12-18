import os

if not os.path.exists("plots"):
    os.makedirs("plots")

for i in [ "plot_enh.py"]:
    st = "python3 "+i
    os.system(st)
