#=========================================================================
# plot-example.py
#=========================================================================
# An example python plotting script.

import matplotlib.pyplot as plt
import math
import sys
import os.path

import numpy

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
plt.rcParams['font.size']          = 9
plt.rcParams['font.family']        = 'serif'
plt.rcParams['font.serif']         = ['Times']
plt.rcParams['figure.figsize']     = fig_size

#-------------------------------------------------------------------------
# Get data
#-------------------------------------------------------------------------

x = 100 + 15 * numpy.random.randn(10000)

#-------------------------------------------------------------------------
# Create plot
#-------------------------------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist( x, 50, normed=1, facecolor='g', alpha=0.75 )
ax.set_xlabel('Foo')
ax.set_ylabel('Probability')
ax.set_xlim( 40, 160 )
ax.set_ylim( 0, 0.03 )

# Turn off top and right border

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

