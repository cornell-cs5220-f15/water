#!/share/apps/python/anaconda/bin/python

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_plot(runs):
    "Plot results of timing trials"
    for arg in runs:
        df = pd.read_csv("timing-{0}.csv".format(arg))
        df = df.sort('threads')
        nrm = df['time'][0]
        plt.plot(df['threads'], df['time']/nrm, label=arg)
    plt.xlabel('Threads')
    plt.ylabel('Time (normalized)')

def show(runs):
    "Show plot of timing runs (for interactive use)"
    make_plot(runs)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def main(runs):
    "Show plot of timing runs (non-interactive)"
    make_plot(runs)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('thread_timing.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
    main(sys.argv[1:])
