import csv
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

t = []

with open('timings.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        t.append(float(row[0]))

plt.plot(t)
plt.savefig('timings.png')
plt.clf()
plt.cla()
