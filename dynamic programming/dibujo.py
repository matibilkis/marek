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

plt.xlabel(r'$|\alpha|^{2}$',size=50)
plt.ylabel(r'$P_s^{hel} - P_*^{(L)}$',size=50)

plt.legend(prop={"size":50})
plt.savefig("dp_results_notation.png")
plt.show()
