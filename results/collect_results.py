import os, re, csv

with open('results.csv', 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['n', 'p', 'b', 'time'])
    for fname in os.listdir(os.getcwd()):
        if re.match('.*\.o[0-9]+', fname):
            with open(fname) as f:
                x = f.read().split('\n')
                params = x[0].split()
                n = params[3]
                p = params[-2]
                b = params[-1]
                time = float(x[-2].split(': ')[1])
                writer.writerow([n, p, b, time])
