#=========================================================================
# fig-parallel-node-results.py
#=========================================================================

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

aspect_ratio  = 0.70

fig_width     = 6.5                           # width in inches
fig_height    = fig_width * aspect_ratio      # height in inches
fig_size      = [ fig_width, fig_height ]

#-------------------------------------------------------------------------
# Configure matplotlib
#-------------------------------------------------------------------------

plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.size']          = 16
plt.rcParams['font.family']        = 'serif'
plt.rcParams['font.serif']         = ['Times']
plt.rcParams['figure.figsize']     = fig_size

#-------------------------------------------------------------------------
# Raw data
#-------------------------------------------------------------------------

# Benchmarks

bmarks = [
  'dam_break',
  'pond',
  'river',
  'wave',
]

num_bmarks = len( bmarks )

# Configurations

configs = [
  '1',
  '2',
  '4',
  '6',
  '8',
  '12',
  '16',
  '18',
  '20',
  '22',
  '24'
]

num_configs = len( configs )

# Results (execution time in seconds)

perf_data = [

  # Dam      Pond     River    Wave

  # serial

  [
    # 4.43e-2, 2.91e-2, 3.60e-2, 4.40e-2,
    1.62980008, 1.38199294, 1.5994134, 1.72687586
  ],

  # parallel (2 threads)

  [
    # 2.94e-2, 2.34e-2, 2.93e-2, 3.54e-2,
    1.56364356, 1.06897982, 1.48551, 1.57664108
  ],

  # parallel (4 threads)

  [
    # 1.32e-2, 8.67e-3, 1.10e-2, 1.31e-2,
    1.48042942, 1.03875882, 1.41134254, 1.5308339
  ],

  # parallel (6 threads)

  [
    # 1.03e-2, 8.33e-3, 1.06e-2, 1.04e-2,
    1.43045718, 1.05831446, 1.436788, 1.53733296
  ],

  # parallel (8 threads)

  [
    # 8.21e-3, 6.44e-3, 8.15e-3, 9.70e-3,
    1.52202828, 1.07554644, 1.45426624, 1.56909764
  ],

  # parallel (12 threads)

  [
    # 1.06e-2, 4.97e-3, 1.07e-2, 7.33e-3,
    1.45071744, 1.07890796, 1.46010786, 1.58024256
  ],

  # parallel (16 threads)

  [
    # 7.22e-3, 5.66e-3, 6.98e-3, 8.52e-3,
    1.52923774, 1.129831942, 1.5471821, 1.49500626
  ],

  # parallel (18 threads)
  [
    1.37822696, 1.0329229, 1.47621496, 1.5879974
  ],

  # parallel (20 threads)
  [
    1.62591866, 1.04607394, 1.49237058, 1.54658156
  ],

  # parallel (22 threads)
  [
    1.52086766, 1.1537411, 1.47274562, 1.61659738
  ],

  # parallel (24 threads)
  [
    1.44664722, 1.1829096, 1.43610872, 1.53214778
  ]

]

perf_data = [ np.array( perf_data[0] ) / np.array( data ) for data in perf_data ]

#-------------------------------------------------------------------------
# Plot parameters
#-------------------------------------------------------------------------

# Setup x-axis

ind = np.arange( num_bmarks )
mid = num_configs / 2.0

# Bar widths

width = 0.10

# Colors

colors = [
  '#FFCCCC',
  '#FF9999',
  '#FF6666',
  '#FF3333',
  '#FF0000',
  '#CC0000',
  '#990000',
  '#660000',
  '#ff6666',
  '#66cccc',
  '#ff9966',
  '#33cc99',
  '#ff99cc',
  '#66cc99',
  '#ff99cc',
  '#66cc99',
  '#ff99cc',
  '#ffff99',
  '#ffff99',
  '#ffff99',
]

#-------------------------------------------------------------------------
# Create plot
#-------------------------------------------------------------------------

# Initialize figure

fig = plt.figure()
ax  = fig.add_subplot(111)

# Plot formatting

ax.set_xticks( ind+mid*width+width )
ax.set_xticklabels( bmarks )

ax.set_xlabel( 'Initial Conditions', fontsize=16 )
ax.set_ylabel( 'Speedup',            fontsize=16 )

ax.grid(True)

# Set axis limits

plt.axis( xmax=num_bmarks-1+(num_configs+2)*width, ymax=1.5)#6.5 )

# Add bars for each configuration

rects = []

for i, perf in enumerate( perf_data ):
  rects.append( ax.bar( ind+width*i+width, perf, width, color=colors[i] ) )

# Set tick positions

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Add horizontal line for baseline

plt.axhline( y=1, color='k', linewidth=1.5 )

# Legend

ax.legend( rects, configs, loc=8, bbox_to_anchor=(0.01,1.02,0.98,0.1),
           ncol=4, borderaxespad=0, prop={'size':12}, frameon=False )

# Pretty layout

plt.tight_layout()

# Turn off top and right border

ax.xaxis.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#-------------------------------------------------------------------------
# Generate PDF
#-------------------------------------------------------------------------

input_basename = os.path.splitext( os.path.basename(sys.argv[0]) )[0]
output_filename = input_basename + '.py.pdf'
plt.savefig( output_filename, bbox_inches='tight' )

