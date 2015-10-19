import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sys

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def get_last_row(csv_filename):
    with open(csv_filename,'rb') as f:
        reader = csv.reader(f)
        lastline = reader.next()
        for line in reader:
            lastline = line
        return lastline

file_name_template = 'average_t%d.csv'

threads = [1,2,3,4,5,6,7,8,9,10]
t = [0] * len(threads)
i = 0

for th in threads:
    with open(file_name_template%th, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            t[i] = float(row[0])
    i += 1

t_f = t[0]
for i in range(len(t)):
    t[i] = t_f/t[i]

plt.plot(threads, t)
plt.xlim([1,25])
plt.xlabel("Number of OMP threads")
plt.ylabel("Speedup over the serial implementation")
plt.savefig('weak_scaling.png', dpi=1000)
plt.clf()
plt.cla()
