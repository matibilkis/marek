from basics import Basics
import numpy as np
import matplotlib.pyplot as plt

b = Basics(layers=1,bound_displacements=0.7, resolution=0.7)
b.define_actions()
dol_prob= [b.err_dolinar(disp) for disp in b.actions]
max_prob = 1-b.err_kennedy(b.actions)
optimal = max(dol_prob)
pos_optimal = np.where(dol_prob == optimal)[0][0]
plt.plot(b.actions, dol_prob, '.', label="Dolinar")
# plt.plot(b.actions, max_prob, '.',label="Maximum-likelihood")
plt.plot([b.actions[pos_optimal]]*2, [min(dol_prob),optimal],'--')

plt.legend()
# plt.savefig("kennedy_probs_4.png")
plt.show()
