##############################################################################
#########################         USER INPUT       ###########################
##############################################################################
#
# GLOSSARY:
# ---------
# sfile [srting]         : name of the output SAC file from STEP1
# min_freq [Hz]          : minimum desired frequency
# max_freq [Hz]          : maximum desired frequency
# Vmin     [m/s]         : minimum group velocity to plot the dispersion map
# Vmax     [m/s]         : maximum group velocity to plot the dispersion map
# dist     [m]           : source-receiver separation
# alpha_fact [numerical] : bandwidth of Gaussian filters such that
#                          bandwidth = (max_freq - min_freq)/alpha_fact                               
# dt [s]                 : sampling period of [sfile]. i.e.  1/[samples/sec]. e.g. if sac file is recorded with 8000 samples/second, dt = 1/8000 = .000125
# levels [percentage]    : contours for percentage signal strength (0%-100%)
#                          enter this as an array (e.g. [70, 80, 90])
#
# NOTES:
# ------
# alpha_fact
#      The larger the alpha_fact, the narrower the bandwidth and more noisier
#      the dispersion map can get. On the other hand, the smaller the
#      alpha_fact, the broader the bandwidth is, which smears the dispersion map
#      at longer periods (lower frequencies). To minimise the tradeoff,
#      define an alpha_fact by trial and error for the dataset of interest such
#      that a relatively sharper dispersion map can be generated for the entire
#      frequency range of interest.
#      Suggestion: try bandwidths between 5 and 20 for a few seismograms in the
#                  dataset and explore if both the higher and lower end of
#                  periods have resolvable dispersion characteristics.
#      
#
# USAGE:
# -----
# Dispersion Curve Picking
#      The dispersion map shows the distribution of normalised spectral
#      signal strength as a function of filter periods and group velocity.
#         - ensure that Vmin and Vmax are suitably defined.
#         - when the map pops up, use the Zoom button in the plot window
#           to pan into the area of interest.
#         - points on the map are picked by bringing the cursor to the
#           point of interest and hitting the space bar. 
#         - first pick the origin, x-axis maximum and y-axis maximum.
#         - Now pick your dispersion curve points.
#         - use contours to guide the dispersion curve picking.
#           e.g. pick the dispersion curve from the center of the area
#           enclosed by the 90% contour. This method assumes that most
#           energy is transmitted in the fundamental mode across all periods. 
#
# Deleting Clicks
#      Hit backspace. The first three picks register the origin, max x, and
#      max y. 
#
# Completing Curve Picking
#       Hit the escape key and the plot for the next seismogram will pop up.
#       Repeat the above procedure.
#
# Resume From Where You Exited
#       The current program will resume from where you exited in previous runs.
#       It creates two files (sacfiles.txt and locfile.txt), which enables
#       this option. These files are created in the first instance this program
#       is run and will be updated automatically in subsequent runs.
#       sacfiles.txt holds a list of all SAC files in the working
#       directory and locfile.txt is indexed to the last file that was
#       completely operated on in the last run. The program will output the
#       number of SAC files that are in the working directory and the number
#       of files that were operated on in previous runs. You may run this program
#       as many times as you like and it will pick up from where you left off.
#       For this option to work as intended, ensure that there are no other
#       files in the working directory with .sac/.SAC extension. The user may
#       decide to delete sacfiles.txt and locfile.txt and start from the beginning.
#
#
# VERSION HISTORY:
# ---------------
# 19/01/2024    Original code FTANos acquired
#               https://github.com/thecraigoneill/FTANos
#               Dr. Craig O'Neill, thecraigoneill@gmail.com
#  
# 23/01/2024    Modified as a standalone dispersion curve picking routine
#               with flexible user input, visualisation, and pick guiding
#               Dr. Januka Attanayake (januka.attanayake@ghd.com)
#
# 29/01/2024    Added loops to read multiple sac files.
#               Added code to resume picking from where one exited.
#               Modified code to pick points with cursor and keyboard.
#               Dr. Januka Attanayake (januka.attanayake@ghd.com)
#
# 02/02/2024    Enabled saving user clicks with the dispersion figure.
#               Re-structured defs.
#               Dr. Januka Attanayake (januka.attanayake@ghd.com)
#
# 06/02/2024    Added filename (for tracking) to the plot window
#               Dr. Januka Attanayake (januka.attanayake@ghd.com)

wcard      = "*.sac"
min_freq   = 6
max_freq   = 126
Vmin       = 0
Vmax       = 1500
dist       = 21
alpha_fact = 12
dt         = 0.000125 
levels     = [90, 95] # levels for plotting contours

##############################################################################
#########################    USER INPUT END       ############################
##############################################################################

# IMPORT LIBRARIES
import sys, glob, os
from itertools import islice
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from numba import jit
from obspy import read
from obspy.io.sac.sactrace import SACTrace
from scipy.optimize import minimize
from scipy.interpolate import interp1d, griddata
from scipy.signal import find_peaks



# FTAN MAP COMPUTATION
class FTANos(object):
	def __init__(self,
				 fre1 = min_freq,
				 fre2 = max_freq,
				 vel1 = Vmin,
				 vel2 = Vmax,
				 dist = dist,
				 alpha = (max_freq - min_freq)/alpha_fact,
				 dt = dt,
				 filename = None
					):

		self.fre1 = fre1
		self.fre2 = fre2
		self.vel1 = vel1
		self.vel2 = vel2
		self.alpha = alpha     
		self.dt = dt
		self.dist = dist
		self.filename = filename
		self.st = read(self.filename, format="SAC")
		self.tr = self.st[0].detrend()
		self.x = self.tr.data
		self.dom = 1/(len(self.x)*self.dt)
		# else: 
		# 	raise ValueError('The file format can not be recognized!')
		

	def times(self,):
		t = np.arange(1e-12, np.size(self.x)*self.dt, self.dt)
		return t

	def periods(self,):
		self.T1 = 1.0/self.fre2
		self.T2 = 1.0/self.fre1
		p = np.linspace(self.T1, self.T2, 40)
		return p

	def FTAN_a(self,):
        # FTAN routine. Some parts cannibalised from CPS/AFTAN under BSD licence. Largely rewritten. 
        # Some python inspired by other FTAN distros but again largely rewritten. 
        # Inherits structure "self" containing time-series data x, and sample rate dt, periods.
        # The FTAN also needs the band filter width alpha, which is predefined (or use the default)
        # Ammplitude map scaled as per CPS and FTAN. 
        # Returns an array of amplitude for plotting.
		amplitude = np.zeros(shape=(len(self.periods()), len(self.x)))
		#apply Fourier transformation
		xi = fft(self.x)
		# array of frequencies
		freq = fftfreq(len(xi), d=self.dt)
		freq_n = np.arange(0, len(self.x), 1)*self.dom
		# filter signal if needed
		# xi[freq < 0] = 0.0
		# xi[freq > 0] *= 2.0

		for iperiod, T0 in enumerate(self.periods()):
			f0 = 1.0/T0
			xi_f0 = xi*np.exp(-self.alpha*((freq-f0)/f0)**2)
			#apply Fourier transformation back to time domain
			xi_f1 = ifft(xi_f0)/len(self.x)
			xi_f2 = np.copy(xi_f0)
			#filling amplitude and phase of column
			amplitude[iperiod, :] = 60.0* np.log10(np.abs(xi_f1)/(np.max(np.abs(xi_f1))-np.min(np.abs(xi_f0))))
			amax = -1.0e10
			for amp in amplitude[iperiod, :]:
				if amp > amax:
					amax = amp
			amplitude[iperiod, :] = amplitude[iperiod, :] + 100.0 - amax
			i = 0
			for amp in amplitude[iperiod, :]:
				if amp < 40.0:
					amplitude[iperiod, i] = 40.0
				i += 1
			if iperiod == (len(self.periods())-1):
				print(np.c_[freq, amplitude[iperiod, :]])

			#phase[iperiod, :] = np.angle(xi_f1)
		return amplitude

	def plot_FTAN(self,filename):
        # Function to create an FTAN plot
        # Regrids data onto a finer grid for presentation
        # Inherits structure "self", which has periods and v range, as well as amplitude from FTAN routine.
        # Creates a png FTAN map named " self.filename+".png" ", with filename being inherited from input.
		amp = self.FTAN_a()
		v = self.dist/self.times()
		X, Y = np.meshgrid(self.periods(), self.times())
		V = self.dist/Y
		X_new, Y_new = np.meshgrid(self.periods(), np.linspace(self.vel1, self.vel2, 4000))
		A_new = griddata((X.ravel(), V.ravel()), amp.T.ravel(), (X_new, Y_new), method='nearest')
		np.set_printoptions(threshold=sys.maxsize)

		ax1    = plt.subplot2grid((1,4), (0,0), colspan=3)
		extent = [self.T1,self.T2,self.vel1,self.vel2]
		cont   = ax1.contour(A_new, levels=levels, extent=extent, origin='lower', linewidths=0.5, colors='k')
		im1    = ax1.imshow(A_new, cmap='winter', origin='lower', extent=extent, aspect='auto')
		ax1.clabel(cont, cont.levels, inline=True, fontsize=8)
		ax1.set_xlabel("Period (s)")
		ax1.set_ylabel("Group Velocity (m/s)")
		ax1.set_title(filename)
		plt.colorbar(im1, orientation="horizontal")

		ax2 = plt.subplot2grid((1,4),(0,3))
		ax2.plot(self.x, self.times(), color="xkcd:deep purple", linewidth=2)
		ax2.set(xticklabels=[])
		ax2.set_ylim(np.min(self.times()), np.max(self.times()))
		plt.tight_layout()

		# start picking the dispersion curve
		print("Hit the Zoom button and pan to the area of interest")                
		print("position the cursor and hit the space bar to select points")                
		print("First click the origin, maximum of x axis, and maximum of y-axis, in that order.")
		print("All clicks after the first three clicks are treated as data.")
		print("Hit backspace to delete clicks")
		print("Hit escape to exit")

		x = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_add=None, mouse_pop=None, mouse_stop=None)

		print("Coordinates of the clicked points", np.array(x))
		x = np.asarray(x)
		xminP = x[0,0]
		yminP = x[0,1]
		xmaxP = x[1,0]
		ymaxP = x[2,1]
		points_x = x[3:,0]
		points_y = x[3:,1]
		# #########normalize the coordinates in the scale of periods and velocity##########
		X = ((points_x - xminP)/(xmaxP - xminP))*(self.T2 - self.T1) + self.T1
		Y = ((points_y - yminP)/(ymaxP - yminP))*(self.vel2 - self.vel1) + self.vel1

		# disp_file = "{0:s}_{1:s}.disp".format(self.tr.stats.station, self.tr.stats.channel)
		disp_file = self.filename+".disp"
		np.savetxt(disp_file, np.c_[X,Y])

		# put picked points on the figure for saving
		ax1.scatter(points_x, points_y, marker='x', s=10, color='r')
		
		fig_file = self.filename+".png"
		plt.savefig(fig_file, dpi=300, bbox_inches='tight')
		plt.close()
		
####################################################
################## READ FILES ######################
####################################################

# check and create sacfiles.txt and locfile.txt
# start from first line or resume from last file

if os.path.exists("sacfiles.txt"):
        print("sacfiles.txt exists")
        sacfiles  = open("sacfiles.txt","r")

else:
        sacfiles = open("sacfiles.txt","w")
        for filename in glob.glob(wcard):
                print(filename)
                sacfiles.write("%s\n" % filename)
        sacfiles.close()
        sacfiles  = open("sacfiles.txt","r")

num_files = len(sacfiles.readlines())
print("Number of SAC files:",num_files)
sacfiles.seek(0)
 
if os.path.exists("locfile.txt"):
        print("locfile.txt exists")
        if os.stat("locfile.txt").st_size == 0:
                line_no = 0
                locfile   = open("locfile.txt",'w')
        else:
                locfile   = open("locfile.txt",'r+')
                line_no   = int(locfile.readline())
                print("# of files read in the previous runs:", line_no)

else:
        locfile   = open("locfile.txt","w+")
        locfile.write("0")
        line_no = 0

if num_files == line_no:
        print("\n-----All SAC files have been read previously-----\n")
        exit()

input("Hit ENTER to continue\n")        



for lines in range(line_no):
        sacfiles.readline()        

for sacfilename in sacfiles:
        print("WORKING ON:", sacfilename)

        sfile = sacfilename.strip()
        outfile = FTANos(filename=sfile)
        outfile.plot_FTAN(sfile)
        figname = sfile +".disp"

        line_no = line_no + 1
        locfile.seek(0)
        locfile.truncate()
        locfile.write("%s" % line_no)
        
        if num_files == line_no:
                print("\n-----All SAC files have been read-----\n")

sacfiles.close()
locfile.close()
