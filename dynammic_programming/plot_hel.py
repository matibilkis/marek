import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 50})

plt.figure(figsize=(30,25))

bets = pd.read_csv("solobetas.csv")
bets = bets.to_numpy()

betsatts = pd.read_csv("betas_atts.csv")
betsatts = betsatts.to_numpy()

colors = {"2":"black", "3": "red", "4": "green", "5":"purple", "6": "red", "7":"blue", "8":"pink", "9":"yellow"}

# for layer in range(2,10):
plt.plot(np.square(bets[:,0]), bets[:,1], color="red", linewidth=10, alpha=.6, label="Helstrom bound")# = \n"+ r'$\frac{1 - \sqrt{1-e^{-4 |\alpha|^{2}}}}{2}$')
plt.plot(np.square(bets[:,0]), np.exp(-4*np.square(bets[:,0])), '-.',color="blue", linewidth=10, alpha=.6, label= r'$|< - \alpha | \alpha > |^{2} = e^{-4 |\alpha|^{2}}$')# = \n"+ r'$\frac{1 - \sqrt{1-e^{-4 |\alpha|^{2}}}}{2}$')

    # plt.plot(np.square(betsatts[:,0]), betsatts[:,1] - betsatts[:,11-layer], '--',linewidth=6, alpha=.85, color=colors[str(layer)])

plt.xlabel(r'$|\alpha|^{2}$',size=50)
# plt.ylabel("Optimal probability of success",size=50)
plt.yticks(np.arange(0,1.1,.1))
# plt.xticks(size=30)
plt.title(r'$P_{Helstrom} = \frac{1 + \sqrt{1 - |< - \alpha | \alpha > |^{2}}}{2}$', size=70)
plt.legend(prop={"size":60})
# plt.savefig("hel.png")
plt.show()
