import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = np.genfromtxt('results.csv', delimiter=',', names=True)
baseline = np.genfromtxt('baseline.csv', delimiter=',', names=True)
baseline.sort(order=['n'])
N = baseline['n']
baseline_y = baseline['time']
colors = ['ro-', 'g+-', 'b*-', 'y^-', 'ks-', 'm*-']

# strong scaling
for i, n in enumerate(N):
    pts = results[results['n']==n]
    pts.sort(order=['n'])
    x = pts['p']
    y = baseline_y[i] / pts['time']
    plt.plot(x,y, colors[i], label=str(n))
plt.legend(loc=4)
plt.savefig('strong.pdf')
