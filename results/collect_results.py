import os, re, csv

with open('results.csv', 'w') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['n', 'p', 'b', 'time'])
    for fname in os.listdir(os.getcwd()):
        if re.match('.*\.o[0-9]+', fname):
            with open(fname) as f:
                print fname
                try:
                    # x = f.read().split('\n')
                    # params = x[0].split()
                    # n = params[3]
                    # p = params[-3]
                    # b = params[-2]
                    # mic = int(params[-1])
                    # time = float(x[-2].split(': ')[1])
                    # writer.writerow([n, p, b, time])
                    found = False
                    for line in f:
                        par = line.split()
                        if not found and par[0] == '2' and par[1] == '2':
                            found=True
                            params = par
                            n = params[3]
                            p = params[-3]
                            b = params[-2]
                            print n, p
                        elif found and par[0] == 'Total':
                            time = float(line.split(': ')[1])
                            found=False
                            writer.writerow([n, p, b, time])
                except:
                    print fname
