#=========================================================================
# fig-parallel-node-weak-results.py
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
]

num_configs = len( configs )

# Results (execution time in seconds)

perf_data = [

  # serial

  [
    4.43e-2, 2.91e-2, 3.60e-2, 4.40e-2,
  ],

  # parallel (2 threads)

  [
    5.80e-2, 4.92e-2, 5.78e-2, 6.58e-2,
  ],

  # parallel (4 threads)

  [
    1.02e-1, 5.87e-2, 1.00e-1, 1.11e-1,
  ],

  # parallel (6 threads)

  [
    1.08e-1, 8.76e-2, 1.14e-1, 1.27e-1,
  ],

  # parallel (8 threads)

  [
    2.47e-1, 1.70e-1, 2.21e-1, 2.44e-1,
  ],

  # parallel (12 threads)

  [
    5.42e-1, 4.46e-1, 6.54e-1, 6.10e-1,
  ],

  # parallel (16 threads)

  [
    9.82e-1, 6.86e-1, 9.40e-1, 1.10e0,
  ],

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

plt.axis( xmax=num_bmarks-1+(num_configs+2)*width )

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

ax.legend( rects, configs, loc=8, bbox_to_anchor=(0.01,1.02,0.98,0.1), ncol=4,
           borderaxespad=0, prop={'size':12}, frameon=False )

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