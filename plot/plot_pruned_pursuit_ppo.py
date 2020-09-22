import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

METHOD = "ppo"

# FORMAT = "pgf"  # "png"
FORMAT = "pgf"
OUTPUT_FILENAME = "pursuitGraph"

if FORMAT == "pgf":
    matplotlib.use("pgf")
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 9,
        "legend.fontsize": 9,
        "text.usetex": True,
        "pgf.rcfonts": False
    });
    # plt.figure(figsize=(2.65, 1.5))
    plt.figure(figsize=(5, 2.8))

else:
    plt.figure()

data_path = "data/pursuit/"
data_path_tweak = "data/pursuitTweak/"

scale_factor = 8. if METHOD in ["DQN", "RDQN"] else 1.

linestyles = ["-","-","-","-"]

colors = ['#1B4332','#40916C','#74C69D','#B7E4C7']
data_mean = np.array([np.nan])
for i in range(1,5):
    df = pd.read_csv(os.path.join(data_path, 'unpruned_'+METHOD+'_'+str(i)+'.csv'))
    df = df[['episodes_total', "episode_reward_mean"]]
    data = df.to_numpy()
    data_mean = np.append(data_mean, df["episode_reward_mean"].max())
    window = int(len(data[:, 1])/30)
    filtered = signal.savgol_filter(data[:, 1],window+1 if window%2 == 0 else window,5)
    plt.plot(data[:, 0], filtered/scale_factor, label='Unpruned '+str(i), linestyle=linestyles[i-1], linewidth=0.75, color=colors[i-1])

print('Unpruned: ',np.nanmean(data_mean))

colors = ['#023E8A','#0096C7','#48CAE4','#ADE8F4']
data_mean = np.array([np.nan])
for i in range(1,5):
    df = pd.read_csv(os.path.join(data_path_tweak, 'pruned_'+METHOD+'_'+str(i)+'.csv'))
    df = df[['episodes_total', "episode_reward_mean"]]
    data = df.to_numpy()
    data_mean = np.append(data_mean, df["episode_reward_mean"].max())
    window = int(len(data[:, 1])/30)
    filtered = signal.savgol_filter(data[:, 1],window+1 if window%2 == 0 else window,5)
    plt.plot(data[:, 0], filtered/scale_factor, label='Pruned '+str(i), linestyle=linestyles[i-1], linewidth=0.75, color=colors[i-1])

print('Pruned: ',np.nanmean(data_mean))

plt.xlabel('Episode', labelpad=1)
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Pursuit')
plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
plt.xlim(0, 60000)
plt.yticks(ticks=[80,90,100,110,120],labels=['80','90','100','110','120'])
plt.ylim(75, 125)
plt.tight_layout()
plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("pursuit_"+METHOD+".pgf", bbox_inches = 'tight',pad_inches = .025)
plt.savefig("pursuit_"+METHOD+".png", bbox_inches = 'tight',pad_inches = .025, dpi = 600)
