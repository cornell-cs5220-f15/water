#=========================================================================
# plot-example.py
#=========================================================================
# An example python plotting script.

import matplotlib.pyplot as plt
import math
import sys
import os.path

import numpy as np

#-------------------------------------------------------------------------
# Calculate figure size
#-------------------------------------------------------------------------
# We determine the fig_width_pt by using \showthe\columnwidth in LaTeX
# and copying the result into the script. Change the aspect ratio as
# necessary.

fig_width_pt  = 244.0
inches_per_pt = 1.0/72.27                     # convert pt to inch
aspect_ratio  = ( math.sqrt(5) - 1.0 ) / 2.0  # aesthetic golden mean

fig_width     = fig_width_pt * inches_per_pt  # width in inches
fig_height    = fig_width * aspect_ratio      # height in inches
fig_size      = [ fig_width, fig_height ]

#-------------------------------------------------------------------------
# Configure matplotlib
#-------------------------------------------------------------------------

plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.size']          = 7
plt.rcParams['font.family']        = 'serif'
plt.rcParams['font.serif']         = ['Times']
plt.rcParams['figure.figsize']     = fig_size

#-------------------------------------------------------------------------
# Create plot
#-------------------------------------------------------------------------

compute_limit = 120
mem_bandwitdh = 59
max_ai = 3

x1 = np.linspace(0,compute_limit/mem_bandwitdh)
x2 = np.linspace(compute_limit/mem_bandwitdh, max_ai)
plt.ylim(0, compute_limit + 30)
plt.ylabel("GFlops/s")
plt.xlabel("Arithmetic Intensity")
plt.xlim(0, max_ai)
plt.hlines(compute_limit, 0, compute_limit/mem_bandwitdh, linestyles=u'dashed')
plt.hlines(compute_limit, compute_limit/mem_bandwitdh, max_ai, linestyles=u'solid')
plt.plot(x1, mem_bandwitdh*x1, 'b')
plt.plot(x2, mem_bandwitdh*x2, 'b--')
plt.axvspan(0.11,0.27, ymin=0, ymax=1, alpha=0.5, color='r')

#-------------------------------------------------------------------------
# Generate PDF
#-------------------------------------------------------------------------

input_basename = os.path.splitext( os.path.basename(sys.argv[0]) )[0]
output_filename = input_basename + '.py.pdf'
plt.savefig( output_filename, bbox_inches='tight' )