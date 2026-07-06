import sys, glob, os
import numpy as np
from scipy.interpolate import interp1d
import statistics, datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools
from pathlib import Path

from evodcinv import EarthModel, Layer, Curve, factory
from disba import PhaseDispersion, depthplot

# USER INPUT

# 1. file path
filedir = "./"            # path of the directory containing dispersion curves relative to the working directory  
wcard   = "*_.sac.disp"    # wildcard with preceding asterix to list dispersion curve files (should be unique in $filedir) 



# 2. dispersion curve interpolation
int_npts                = 50 # number of points to which measured dispersion curve is interpolated
                             # exclude first and/or last n measured periods for stable inversion results
exclude_first_n_periods = 0  # excluded the first n measured dispersion periods from the inversion; 0 = none excluded 
exclude_last_n_periods  = 0  # excluded the  last n measured dispersion periods from the inversion; 0 = none excluded



# 3. inversion parameters
############################################
# 1. SELECT STOCHASTIC OPTIMISER ALGORITHM
#     (a) Competitive Particle Swarm Optimization (cpso)
#     (b) Covariance Matrix Adaptation - Evolution Strategy (cmaes)
#     (c) Differential Evolution (de)
#     (d) Neighborhood Algorithm (na)
#     (e) Particle Swarm Optimization (pso)
#     (f) vdcma
optmz="de"

############################################
# 2. SELECT MISFIT FUNCTION
msft="rmse"                # options: rmse, norm1, norm2

############################################
# 3. OPTIMIZER ARGUMENTS
population_size = 200       # 5-10 x number of model layers 
num_iterations  = 500       # number of iterations 
num_cores       = -1        # number of CPU cores to be used
seed_value      = 0         # initalizing value between 0 and (2**32-1)
runs            = 2         # number of inversion runs with population_size and num_iterations
                            # higher number of runs improves accuracy but at an additional computational cost
                            # results from the run containing the minimum misfit is taken

############################################
# 4. MODEL SEARCH RANGE
# Initialize model
model = EarthModel()

# Set up model search range assuming uniform search ranges
    # search ranges for each layer must be set up using the format model.add(Layer([1.0, 10.0], [0.1, 2.0]))
    # First argument: layer thickness [km] range; e.g. [1, 10] for 1 km to 10 km search range
    # Declare layer thickness search range using thick_min and thick_max
    #                         thick_min=thick_max for constant layer thickness models
    # Second argument: S velocity [km/s] range; e.g. [0.1, 2.0] for 0.1 km/s 2.0 km/s search 
    # Declare vs search range using vs_min and vs_max
    # num_layers is the desired number of layers in the model

# Alternatively, if the model layers are non-uniform, comment out the for loop and insert layers using 
# the following format 
    #        model.add(Layer([0.0005, 0.0020], [0.1, 0.5]))
    #        model.add(Layer([0.0002, 0.0050], [0.5, 1.0]))
    #        model.add(Layer([0.0010, 0.0020], [1.0, 2.0]))
    #        model.add(Layer([0.0020, 0.0050], [2.0, 3.0]))
    #                     .... and so on ....
    
# Tip: Consider the utility of the model before constructing the model. e.g. For computing Vs30, no need to 
# construct models deeper than about 40 m. 

thick_min   = 0.001
thick_max   = 0.010
vs_min      = 0.010
vs_max      = 3.300
poisson_min = 0.10        # Quartz (Brocher, 2005)
poisson_max = 0.40         
num_layers  = 30

for _ in range(num_layers):
    model.add(Layer([thick_min, thick_max], [vs_min, vs_max], [poisson_min, poisson_max]))
    
    
############################################
# 5. INVERSION PARAMETERS/FLAGS
wave_type           = "rayleigh"      # options: "rayleigh", "love"
mode                = 0               # options: 0,1,2, 0 - fundamental mode
velocity_type       = "group"         # options: "phase" "group"
increasing_vel_flag = "True"          # Set inversion to unconstrained or constrained. options: True, False
                                      # False - unconstrained inversion, velocity jumps across layers are unconstrained
                                      # True - constrained inversion, velocity always increases with depth and across layers
max_depth           = 0.040           # maximum depth of expected investigation [kilometers]
                                      # There is no need to exceed 0.050 km usually and 0.040 km is best for Vs30 estimates
                                      # only used in constrained inversion
smooth_fact         = 1.0e-3          # Factor for smoothing the velocity between layers, i.e. drastic velocity perturbations
                                      # usually do not occur in reality if the model is finely parameterised. Smoothing_fact
                                      # penalizes models with sharp deviations of velocity across layer interfaces. 
                                      # This is a "trial-and-error" input. Start with 1.0e-3. If minimum misfit values of 
                                      # multiple inversion runs get truncated at smooth_fact, then reduce it and re-run inversions.
                                      # both max_depth and smooth_fact are needed for constrained inversion

############################################
# 6. VISUALIZATION PARAMETERS
max_vis_depth        = max_depth    # maximum depth for Vs profile figure and <= max_depth  
resolution           = 300          # dots-per-inch (dpi) resolution of figures, for publication quality; dpi > 300
threshold_percentile = 20           # This is the percentile of misfit values used for selecting the model ensemble
                                    # to make figures and compute standard deviation.
                                    # options: any number (percentile) between 0 and 100
                                    # e.g. if 25 is used, models with misfit values that are in the lowest 25% 
                                    # of misfit values are selected.
                                    # tip: select a reasonably large percentile (larger misfit) if the inversion 
                                    # converges faster (look up the misfit vs. iterations figure).

# Start Time
starttime = datetime.datetime.now()
print('INVERSION STARTED AT:', datetime.datetime.now())


# create/check dispersion file list and the log file

dispfile_path = (filedir + wcard)
print("LOOKING IN DIR:", dispfile_path,"\n")
#time.sleep(10)

# set-up file list
if os.path.exists("dispfiles.txt"):
        print("dispfiles.txt exists")
        
        if os.stat("dispfiles.txt").st_size == 0:
            dispfiles = open("dispfiles.txt","w")
            for filename in glob.glob(dispfile_path):
                print(filename)
                dispfiles.write("%s\n" % filename)
            dispfiles.close()
        
        dispfiles  = open("dispfiles.txt","r")

else:
        dispfiles = open("dispfiles.txt","w")
        for filename in glob.glob(dispfile_path):
                print(filename)
                dispfiles.write("%s\n" % filename)
        dispfiles.close()
        dispfiles  = open("dispfiles.txt","r")

num_files = len(dispfiles.readlines())
print("Number of dispersion curves:",num_files)
dispfiles.seek(0)

# set-up log file for tracking progress
if os.path.exists("disp_locfile.txt"):
        print("disp_locfile.txt exists")
        if os.stat("disp_locfile.txt").st_size == 0:
                line_no   = 0
                locfile   = open("disp_locfile.txt",'w')
                locfile.write("0")
                print("Reading dispersion files from the start")
        else:
                locfile   = open("disp_locfile.txt",'r+')
                line_no   = int(locfile.readline())
                print("# of dispersion files inverted in the previous runs:", line_no)

else:
        locfile   = open("disp_locfile.txt","w+")
        locfile.write("0")
        line_no   = 0
        print("Reading dispersion files from the start")

if num_files == line_no:
        print("\n-----All dispersion curves have been inverted previously-----\n")
        exit()

input("\nHit ENTER to continue\n") 


# configure inversion set-up
if increasing_vel_flag == "True":
    model.configure(
        optimizer      = optmz,  
        misfit         = msft,  
        density        = lambda vs: 1.4 + 0.67 * np.sqrt(vs),
        dt             = 0.001,
        dc             = 0.001,
        optimizer_args = { "popsize": population_size,  # Population size
                            "maxiter": num_iterations,  # Number of iterations
                            "workers": num_cores,  # Number of cores
                            "seed": seed_value, },
        increasing_velocity=increasing_vel_flag,
        extra_terms=[ lambda x: factory.smooth(x, alpha=smooth_fact),
                      lambda x: factory.prior(x, [0.0, max_depth], [vs_min, vs_max], alpha=smooth_fact)],
                    )

else: 
        model.configure(
        optimizer      = optmz,  
        misfit         = msft,  
        density        = lambda vs: 1.4 + 0.67 * np.sqrt(vs),
        dt             = 0.001,
        dc             = 0.001,
        optimizer_args = { "popsize": population_size,  # Population size
                            "maxiter": num_iterations,  # Number of iterations
                            "workers": num_cores,  # Number of cores
                            "seed": seed_value, },
                        )


for lines in range(line_no):
        dispfiles.readline()        

print('Inversion uses the interpolated dispersion curve\n\n')

# loop through inversions        
for dispfilename in dispfiles:
    print("\nWORKING ON:", dispfilename, "File No:",line_no+1)
    
    sacfname = Path(dispfilename.strip()).stem  # common string for saved file names
    
    
    # STEP1: DISPERSION CURVE INTERPOLATION
    # -------------------------------------------------------------------------------------
    
    data          = np.loadtxt(dispfilename.strip())           # remove \n with strip()
    
    # period exclusion switch
    # obs_period [s],  obs_grp_vel[km/s]
    if ( (exclude_first_n_periods > 0) and (exclude_last_n_periods == 0) ) :
        obs_period    = data[exclude_first_n_periods:,0]           
        obs_grp_vel   = data[exclude_first_n_periods:,1]/1e3       
        
    elif ((exclude_last_n_periods > 0) and (exclude_first_n_periods == 0)):
        obs_period    = data[:-exclude_last_n_periods,0]           
        obs_grp_vel   = data[:-exclude_last_n_periods,1]/1e3       
        
    elif ( (exclude_first_n_periods > 0) and (exclude_last_n_periods > 0) ):
        obs_period    = data[exclude_first_n_periods:-exclude_last_n_periods,0]           
        obs_grp_vel   = data[exclude_first_n_periods:-exclude_last_n_periods,1]/1e3        
        
    else:
        obs_period    = data[::,0]           
        obs_grp_vel   = data[::,1]/1e3           
        

    plt.semilogx(obs_period,obs_grp_vel,'r+',label='Measured')

    # interpolate
    int_disp_coeff = interp1d(obs_period,obs_grp_vel,fill_value="extrapolate",kind='slinear')
    t1             = np.log10(np.min(obs_period))
    t2             = np.log10(np.max(obs_period))
    int_disp_t     = np.logspace(t1,t2,int_npts)
    int_grp_vel    = int_disp_coeff(int_disp_t)

    plt.semilogx(int_disp_t, int_grp_vel,'b+', label='Interpolated')
    plt.xlabel('period (s)')
    plt.ylabel('group velocity (km/s)')
    plt.legend()
    
    # save dispersion curve as a png figure
    disp_curve_fig = sacfname + "_dispcurve.png"
    fig1           = plt.savefig(disp_curve_fig, bbox_inches='tight', dpi=resolution)
    plt.close("all") 
    
    
    # STEP 2. INVERSION
    # -------------------------------------------------------------------------------------
    period   = int_disp_t   # interpolated period
    velocity = int_grp_vel  # interpolated velocity
    T_lim1   = min(period)  # lowest period to be plotted [seconds]
    T_lim2   = max(period)  # highest period to be plotted [seconds]
    
    curves = [Curve(period, velocity, mode, wave_type, velocity_type)]

    # Run Inversion
    res = model.invert(curves, maxrun=runs, split_results=True)
    res = min(res, key=lambda x: x.misfit)  # Get the results for the inversions with the lowest misfit


    # End Time
    print('INVERSION ENDED AT:', datetime.datetime.now())
    
    
    # STEP 3. VISUALISATION AND SAVE MODELS + FIGS
    # -------------------------------------------------------------------------------------
    print(res)

    thresh_val = np.percentile(res.misfits[np.isfinite(res.misfits)], threshold_percentile)
    best_res   = res.threshold(thresh_val)
    print('# of models below the misfit threshold:',len(best_res.models))

    
    # ------------------------------------------------------------
    # Vs30 calculation (thickness in km, Vs in km/s)
    # ------------------------------------------------------------
    bestmodel = best_res.model

    thk_km = bestmodel[:, 0]
    vs_kms = bestmodel[:, 2]

    dVs30_km = 0.030  # 30 m
    cum_km   = np.cumsum(thk_km)

    if cum_km[-1] < dVs30_km:
        raise ValueError(
        f"Model depth {cum_km[-1]*1000:.1f} m < 30 m – cannot compute Vs30"
        )

    # first layer that exceeds 30 m
    idx = np.searchsorted(cum_km, dVs30_km)

    # travel time through full layers above
    time_s = np.sum(thk_km[:idx] / vs_kms[:idx])

    # partial layer to exactly 30 m
    depth_before = cum_km[idx-1] if idx > 0 else 0.0
    partial_km   = dVs30_km - depth_before
    time_s      += partial_km / vs_kms[idx]

    vs30_inv = (dVs30_km / time_s) * 1000.0  # m/s

    print(f"Vs30 Inverted: {vs30_inv:.0f} m/s")

    
    
    ###########################################################################
    # Plot results
    t = period
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(25,10), width_ratios=[1,3.2,1])
    font = {'size':15,
            'weight':'normal',
            'family':'Tahoma'}
    plt.rc('font',**font)

    for a in [ax0,ax1,ax2]:
        a.grid(True, linestyle=":")

    zmax = max_vis_depth
    cmap = "viridis_r"

    # Velocity model
    #best_res.plot_model(
    #                    "vs",
    #                    zmax=zmax,
    #                    show="all",
    #                    ax=ax0,
    #                    plot_args={"cmap": cmap},
    #                    )

    best_res.plot_model(
                        "vs",
                        zmax=zmax,
                        show="best",
                        ax=ax0,
                        plot_args={
                                    "color": "red",
                                    "linestyle": "--",
                                    "label": "Best",
                                   },
                        )

    ax0.set_xlabel('Vs (km/s)', fontsize=20, fontweight='bold')
    ax0.set_ylabel('Depth (km)', fontsize=20, fontweight='bold')
    ax0.legend(loc=1, frameon=False)


    ######################################################
    # Dispersion curve
    #best_res.plot_curve(
    #                    t, 0, "rayleigh", "group",
    #                    show="all",
    #                    ax=ax1,
    #                    plot_args={
    #                                "type": "semilogx",
    #                                "xaxis": "period",
    #                                "cmap": cmap,
    #                               },
    #                    )
    ax1.semilogx(
                    period, velocity,
                    color="black",
                    linewidth=2,
                    label="Measured",
                )

    best_res.plot_curve(
                        t, 0, "rayleigh", "group",
                        show="best",
                        ax=ax1,
                        plot_args={
                                    "type": "semilogx",
                                    "xaxis": "period",
                                    "color": "red",
                                    "linestyle": "--",
                                    "label": "Best",
                                   },
                        )

    ax1.set_xlabel('Period(s)', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Group Velocity (km/s)', fontsize=20, fontweight='bold')
    ax1.set_xlim(T_lim1, T_lim2)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_minor_formatter(ScalarFormatter())
    ax1.legend(loc=1, frameon=False)


    ######################################################
    # Misfit - need all misfits, not only within threshold
    res.plot_misfit(ax=ax2)
    ax2.set_xlabel('Iteration', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Misfit', fontsize=20, fontweight='bold')

    # Colorbar
    #norm = Normalize(vmin=res.misfits.min(), vmax=res.misfits.max())
    #smap = ScalarMappable(norm=norm, cmap=cmap)
    #axins = inset_axes(
    #                    ax1,
    #                    width="150%",
    #                    height="6%",
    #                    loc="lower center",
    #                    borderpad=-6.0,
    #                   )

    #cb = plt.colorbar(smap, cax=axins, orientation="horizontal")
    #cb.set_label("Misfit value")

    ######################################################
    # Write out results + save figures
    # text file
    sacfname = Path(dispfilename.strip()).stem
    modfname = sacfname + ".txt"

    with open(modfname, "w", encoding="utf-8") as f:
        f.write(str(res))
        f.write("\n")
        f.write(f"Vs30(m/s)\t{vs30_inv:.0f}\n")


    # png figure
    disp_fig = sacfname + "_inv.png"
    fig2     = plt.savefig(disp_fig, bbox_inches='tight', dpi=resolution)
    plt.close(fig)

    # End Time
    
    endtime = datetime.datetime.now()
    print('PLOTTING ENDED AT:', datetime.datetime.now())
    
    # update locfile
    line_no = line_no + 1
    locfile.seek(0)
    locfile.truncate()
    locfile.write("%d" % line_no)
    locfile.flush()
        
    if num_files == line_no:
            print("\n-----All dispersion curves have been inverted-----\n")

dispfiles.close()
locfile.close()

endtime = datetime.datetime.now()
total_hrs = (endtime - starttime)
print("Total Inversion Time:",total_hrs, "hours")
