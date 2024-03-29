"""
グラフの中で，後半部分の変化率を調査して，収束性を形式的に確認する．
"""

import numpy as np
from env.gym_barfem import BarFemOutputGym
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

history_path = "//ZUIHO/share/user/knakamur/Metamaterial/seminar_data/修論用補足データ/Annealing_results_η3"


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
maxes = []

for i in range(5):
    with open(os.path.join(history_path, "{}/history.pkl".format(i)), 'rb') as web:
        history = pickle.load(web)
    history = np.array(history["result_efficiency"]).reshape((-1, 1))
    prior_history = history[:-1, :]
    pro_history = history[1:, :]
    rc = pro_history / prior_history  # 変化率
    print(np.max(rc[2500:, :]) - 1)
