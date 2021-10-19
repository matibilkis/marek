import numpy as np
import pandas as pd

bets = pd.read_csv("solobetas.csv")
bets = bets.to_numpy()

betsatts = pd.read_csv("betas_atts.csv")
betsatts = betsatts.to_numpy()

print(np.where(bets[:,1] - bets[:,2] == np.max(bets[:,1] - bets[:,2]) ))
print(np.where(bets[:,1] - bets[:,2] == np.max(bets[:,1] - bets[:,2]) ))
print(np.where(bets[:,1] - bets[:,2] == np.max(bets[:,1] - bets[:,2]) ))
print(np.where(bets[:,1] - bets[:,2] == np.max(bets[:,1] - bets[:,2]) ))

import basics
b = basics.Basics(layers=1)
