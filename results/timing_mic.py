import os
import time

n= [1440]
procs= [1, 4, 9, 16, 25, 36, 64, 81, 100, 144, 225]
rounds= [2]

indx = 0

fname = "{}.pbs".format(indx)
out = open(fname,'w')
s = '''#!/bin/sh -l
#PBS -l nodes=1:ppn=24
#PBS -l walltime=5:00:00
#PBS -N lshallow
#PBS -j oe
module load cs5220
cd $PBS_O_WORKDIR
'''
out.write(s)
for np in procs:
  for nr in rounds:
    for ni in n:

      line = "./../lshallow ../tests.lua dam {} {} {} 1\n".format(ni, np, nr)
      out.write(line)
out.close()
os.system("qsub {}".format(fname))
