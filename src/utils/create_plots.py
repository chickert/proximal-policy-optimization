import pandas as pd
import matplotlib.pyplot as plt
from os import listdir

def plot_all(df, name, outdir):
    plt.plot(df["iteration"], df["0"], label="0")
    plt.plot(df["iteration"], df["1"], label="1")
    plt.plot(df["iteration"], df["2"], label="2")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Rewards over time, all seeds")
    plt.legend()

    outpath = f"{outdir}\\{name}_ALL.png"
    plt.savefig(outpath)
    plt.close()

def plot_mean(df, name, outdir):
    dfseeds = df.copy()
    dfseeds = dfseeds.drop(columns=["iteration"])
    df["MeanRewards"] = dfseeds.mean(axis=1)
    plt.plot(df["iteration"], df["MeanRewards"])
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Mean Rewards over time")

    outpath = f"{outdir}\\{name}_MEAN.png"
    plt.savefig(outpath)
    plt.close()

def plot_best(df, name, outdir):
    dfseeds = df.copy()
    dfseeds = dfseeds.drop(columns=["iteration"])
    total_rewards = dfseeds.sum(axis=0)
    index = total_rewards.idxmax()
    plt.plot(df["iteration"], df[index])
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Rewards over time, best seed")

    outpath = f"{outdir}\\{name}_BEST.png"
    plt.savefig(outpath)
    plt.close()

def plot_worst(df, name, outdir):
    dfseeds = df.copy()
    dfseeds = dfseeds.drop(columns=["iteration"])
    total_rewards = dfseeds.sum(axis=0)
    index = total_rewards.idxmin()
    plt.plot(df["iteration"], df[index])
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Rewards over time, worst seed")

    outpath = f"{outdir}\\{name}_WORST.png"
    plt.savefig(outpath)
    plt.close()


filedir = "None"
outdir = "None"

for file in listdir(filedir):
    filepath = f"{filedir}\\{file}"
    name = "_".join(file.split(".")[:-1])
    df = pd.read_csv(filepath)
    plot_all(df, name, outdir)
    plot_mean(df, name, outdir)
    plot_best(df, name, outdir)
    plot_worst(df, name, outdir)






