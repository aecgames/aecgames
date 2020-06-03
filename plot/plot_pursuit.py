import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

# FORMAT = "pgf"  # "png"
FORMAT = "png"
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
FILENAME = OUTPUT_FILENAME + "." + FORMAT
data_path = "data/pursuit/"

df = pd.read_csv(os.path.join(data_path, 'ppo_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], filtered, label='Unpruned 1', linewidth=0.75, linestyle='--')

df = pd.read_csv(os.path.join(data_path, 'ppo_2.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], filtered, label='Unpruned 2', linewidth=0.75, linestyle='--')

df = pd.read_csv(os.path.join(data_path, 'ppo_3.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], filtered, label='Unpruned 3', linewidth=0.75, linestyle='--')

df = pd.read_csv(os.path.join(data_path, 'ppo_4.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], filtered, label='Unpruned 4', linewidth=0.75, linestyle='--')

df = pd.read_csv(os.path.join(data_path, 'ppo_tweak1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], filtered, label='Pruned 1', linewidth=0.75)

df = pd.read_csv(os.path.join(data_path, 'ppo_tweak2.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], filtered, label='Pruned 2', linewidth=0.75)
 
df = pd.read_csv(os.path.join(data_path, 'ppo_tweak3.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], filtered, label='Pruned 3', linewidth=0.75)
 
df = pd.read_csv(os.path.join(data_path, 'ppo_tweak4.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], filtered, label='Pruned 4', linewidth=0.75)
 
# plt.plot(np.array([0,60000]),np.array([31.03,31.03]), label='Random', linewidth=0.75)

plt.xlabel('Episode', labelpad=1)
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Pursuit')
plt.xticks(ticks=[0,10000,20000,30000,40000,50000,60000],labels=['0','10k','20k','30k','40k','50k','60k'])
plt.tight_layout()
plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
# plt.legend(loc='lower right', ncol=1, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig(FILENAME, bbox_inches = 'tight',pad_inches = .025)
