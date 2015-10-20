import os
import time

n= [300,600,900,1200,1500,2400,3600]
procs= [1,4,9,16,25,36]
rounds= [1]
dirp = "results"

indx = 0

for np in procs:
  for nr in rounds:
    for ni in n:
      fname = "{}/{}.pbs".format(dirp,indx)
      out = open(fname,'w')
      s = '''#!/bin/sh -l
#PBS -l nodes=1:ppn=24
#PBS -l walltime=0:30:00
#PBS -N lshallow
#PBS -j oe
module load cs5220
cd $PBS_O_WORKDIR'''
      out.write(s)

      line = "./lshallow tests.lua dam {} {} {}\n".format(ni, np, nr)
      out.write(line)
      indx = indx + 1
      out.close()
      os.system("qsub {}".format(fname))
      time.sleep(10)        
