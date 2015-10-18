#!/share/apps/python/anaconda/bin/python


import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
from optparse import OptionParser, Option

def option_parse():
  usage = "usage: %prog [options]"
  parser = OptionParser(usage=usage)
  parser.add_option("-t", "--types",
                  default="1",
                  dest="types",
                  help="1 - cells/seconds vs nx, 2 - seconds vs nx "
                       "[default: %default]")
  return parser 

def average(l):
  return float(sum(l))/len(l)

def parse(filename, regex, start=None, end=None):
  """  
  Run the regex passed on each line in filename. Values for group(1) of the
  regex are stored in a list and returned. start specifies a regex to search
  for before starting to save data.
  """
  f = open(filename, 'r') 
  data = [] 
  if start:
    enabled = False
  else:
    enabled = True 
  lines=0
  for line in f:
    lines+=1
    # Not enabled, check start condition
    if not enabled:
      if lines==start: 
        enabled = True 
    # Enabled, store data
    else:
      if lines==end+1:
        break 
      res = re.search(regex, line)
      if res: 
        data.append(res.group(1))
  f.close()
  return data 

def parse_int(filename, regex, start=None, end=None):
  data = parse(filename, regex, start, end)
  data = map(int, data)
  return data 

def parse_float(filename, regex, start=None, end=None):
  data = parse(filename, regex, start, end)
  data = map(float, data)
  return data 

def parse_qsub_result(runs):
  "Plot results of timing trials"
  # Make initila csv file
  args=[]
  for arg in runs:
    if not str(arg.split('.o')[0]) in args:
      args.append(str(arg.split('.o')[0]))
      file_name="timing-"+str(arg.split('.o')[0])+".csv"
      f=open(file_name, 'w') 
  f.close()

  # Fill csv file and sort
  for arg in runs:
    parse_lines=[]
    lines=0
    f=open(arg, 'r')
    for line in f.readlines():
      lines+=1
      if re.search('^parse_line', line):
        parse_lines.append(lines)
    parse_lines.append(lines)
    f.close()
    print parse_lines

    file_name="timing-"+str(arg.split('.o')[0])+".csv"
    f=open(file_name, 'a')
    for i in xrange(0, len(parse_lines)-1):
      times = parse_float(arg, "Time: (-?[\d.]+(?:e-?\d+)?)",
          start=parse_lines[i], end=parse_lines[i+1])
      #frames = parse_int(arg, "frames: ([0-9\.]+)",
      #    start=parse_lines[i], end=parse_lines[i+1])
      #assert(len(times)==frames[0])
      nx = parse_int(arg, "nx: ([0-9\.]+)",
          start=parse_lines[i], end=parse_lines[i+1])
      # Write data 
      # x axis : nx (number of cells per side)
      # y axis : cells / seconds 
      if options.types == "1":
        f.write(str(nx[0])+','+str(nx[0]*nx[0]*nx[0]/times[0])+'\n') 
      elif options.types == "2":
        f.write(str(nx[0])+','+str(times[0])+'\n') 
  f.close()

  # Insert header into first line
  args=[]
  for arg in runs:
    if not str(arg.split('.o')[0]) in args:
      args.append(str(arg.split('.o')[0]))
      file_name="timing-"+str(arg.split('.o')[0])+".csv"
      os.system("sort -n "+file_name+" -o "+file_name)
      with open(file_name, 'r') as original: data = original.read()
      with open(file_name, 'w') as modified: modified.write("nx,time\n"+ data)

def make_plot(runs):
  "Plot results of timing trials"
  args=[]
  for arg in runs:
    if not str(arg.split('.o')[0]) in args:
      args.append(str(arg.split('.o')[0]))
      file_name=arg.split('.o')[0]
      df = pd.read_csv("timing-"+file_name+".csv")
      plt.plot(df['nx'], df['time'], label=file_name)
  plt.xlabel('nx')
  if options.types == "1":
    plt.ylabel('Cells / Seconds')
  elif options.types == "2":
    plt.ylabel('Seconds')

def show(runs):
  "Show plot of timing runs (for interactive use)"
  make_plot(runs)
  lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.show()

def main(runs):
  print options.types 
  "Parse qsub result and make csv file"
  parse_qsub_result(runs)
  "Show plot of timing runs (non-interactive)"
  make_plot(runs)
  lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('timing%s.pdf'%options.types, bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
  (options, args) = option_parse().parse_args()
  main(sys.argv[3:])
