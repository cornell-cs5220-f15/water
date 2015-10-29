import os
import time

n= range(2160,2900,360)
procs= [1]
rounds= [1]

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

      line = "./../lshallow ../tests.lua dam {} {} {} 0\n".format(ni, np, nr)
      out.write(line)
out.close()
os.system("qsub {}".format(fname))
