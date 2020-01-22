import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
import matplotlib
plt.figure(figsize=(20,20), dpi=50)
matplotlib.rc('font', serif='cm10')
matplotlib.rc('text', usetex=True)

bets = pd.read_csv("solobetas.csv")
bets = bets.to_numpy()

betsatts = pd.read_csv("betas_atts.csv")
betsatts = betsatts.to_numpy()

colors = {"2":"black", "3": "red", "4": "green", "5":"purple", "6": "red", "7":"blue", "8":"pink", "9":"yellow"}

for layer in range(2,10):
    plt.plot(np.square(bets[:,0]), bets[:,1] - bets[:,layer], color=colors[str(layer)],linewidth=6, alpha=.85,label="L = "+str(layer-1))
    plt.plot(np.square(betsatts[:,0]), betsatts[:,1] - betsatts[:,11-layer], '--',linewidth=6, alpha=.85, color=colors[str(layer)])

colors={"0":"red","1":"blue","2":"green"}
labels={"1":"UCB-1", "2":"max(0.01,"+r'$e^{-\frac{t}{\tau}}$'+")-greedy", "0":"TS"}
eff=0.1


def helstrom(alpha):
    return (1+np.sqrt(1-np.exp(-4*alpha**2)))/2


# size=2000
# for r in range(1,46):
#     lc = np.load("2L2PH0.1R/run_"+str(r)+"/learning_curves.npy", allow_pickle=True)
#     cumrefin = lc[1][40]/lc[0][40] #at 5 10**5
#     prosucgre = lc[2][40]
#     if r<4:
#         # plt.scatter(eff**2, helstrom(eff)-cumrefin , s=size, color=colors[str(r%3)],alpha=0.7, label=labels[str(r%3)])
#         plt.scatter(eff**2, helstrom(eff)-prosucgre, s=size, color=colors[str(r%3)],alpha=0.7, label=labels[str(r%3)])
#     else:
#         # plt.scatter(eff**2, helstrom(eff)-cumrefin, s=size, alpha=0.7,color=colors[str(r%3)])
#         plt.scatter(eff**2, helstrom(eff)- prosucgre, s=size, alpha=0.7,color=colors[str(r%3)])
#
#     if (r%3==0):
#         eff+=0.1

plt.xlabel(r'$|\alpha|^{2}$',size=50)
plt.ylabel(r'$P_s^{hel} - P_*^{(L)}$',size=50)

plt.legend(prop={"size":50})
plt.savefig("to_john.png")
plt.show()
