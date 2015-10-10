import csv
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sys

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

file_name = None

if len(sys.argv) == 1:
    file_name = 'timings'       
elif len(sys.argv) == 2:
    file_name = sys.argv[1]

t = []

with open(file_name + '.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        t.append(float(row[0]))

plt.plot(t)
plt.savefig(file_name + '.png')
plt.clf()
plt.cla()
