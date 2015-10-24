import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def average_results(data, baseline=False):
    output = []
    for n in np.unique(data['n']):
        temp = data[data['n'] == n]
        if not baseline:
            for p in np.unique(temp['p']):
                temp2 = temp[temp['p'] == p]
                output.append((n, p, np.mean(temp2['time']))) #TODO: remove 1
        else:
            output.append((n, np.mean(temp['time'])))
    if baseline:
        return np.array(output, dtype={'names':['n', 'time'], 'formats':['f4', 'f4']})
    else:
        return np.array(output, dtype={'names':['n', 'p', 'time'], 'formats':['f4', 'f4', 'f4']})



results = np.genfromtxt('results.csv', delimiter=',', names=True)
results = average_results(results)
baseline = np.genfromtxt('baseline_360.csv', delimiter=',', names=True)
baseline = average_results(baseline, baseline=True)
baseline.sort(order=['n'])
N = baseline['n']
baseline_y = baseline['time']
# colors = ['ro-', 'g+-', 'b*-', 'y^-', 'ks-', 'm*-', 'c']
colors = ['red', 'green', 'blue', 'yellow', 'black', 'magenta', 'cyan', 'brown', 'orange', 'gray', 'purple', 'pink']
# strong scaling
for i, n in enumerate(N):
    pts = results[results['n']==n]
    pts.sort(order=['n'])
    x = pts['p']
    y = baseline_y[i] / pts['time']
    plt.plot(x,y, color=colors[i], label=str(n))
plt.legend(loc=4)
plt.savefig('strong.pdf')
plt.close()

# weak scaling
for i, n in enumerate(N):
    pts = results[(results['n']/np.sqrt(results['p']))==n]
    print n
    print pts
    pts.sort(order=['n'])
    x = pts['p']
    y = baseline_y[i] / pts['time']
    plt.plot(x,y, color=colors[i], label=str(n))
plt.legend(loc=4)
plt.savefig('weak.pdf')

