# VERSION HISTORY
# ----------------
# 28 January 2025   Added comments to explain pre-existing code   
#                   Moved main functional code into separate defined functions
#                       - User is now prompted whether or not to generate BLN and DAT files   
#                   DAT file now includes a datapoint at surface (i.e. depth = 0)
#                   Added timestamp to output filenames
#                   Changed the filename construction to use the folder name and user-defined prefix
#                       - Assumes the folder name is the line number or name
#
#                   Ben Patterson (ben.patterson@ghd.com)
#
# 3 February 2025   Added timestamp to both output filenames
#                   Changed the filename construction = [project_prefix] + a line prefix
#                       - project_prefix is a user defined string variable
#                       - line prefix is entered via a user prompt
#                   Added error handling for bad/missing sac.txt files during DAT file creation
#
#                   Ben Patterson (ben.patterson@ghd.com)

import numpy as np, pandas as pd, tkinter as tk, os, glob
from scipy.interpolate import interp1d
from datetime import datetime
from tkinter import filedialog, messagebox

## User defined variables   ***CHANGE AS REQUIRED***
########################################
project_prefix      = "Rentails_"     ## Start string of all output files
spreadsheet_name    = "ShotData.xlsx"   ## Ignored if line below (SpecifySpreadsheet) = True
SpecifySpreadsheet  = False             ## If set to True, user is prompted to specify the ShotData.xlsx file.
BLN_depth_cutoff    = 30                ## Lower bound of BLN file
BLN_max_DepthIsRL   = False             ## False = BLN lower bound is depth below min Z. True = BLN lower bound is elevation in RL.
########################################

## Generated variables
current_time          = datetime.now().strftime("%H%M%p")
source_dir            = filedialog.askdirectory()
print(f'\nSource directory = {source_dir}')
line_prefix           = project_prefix+tk.simpledialog.askstring("Line name", "Enter the line number, name or other prefix below:")   
num_input_files       = len(glob.glob(os.path.join(source_dir, '*.sac.txt'))) 
if SpecifySpreadsheet:  shot_info_spreadsheet = filedialog.askopenfilename(title="Shot data spreadsheet", filetypes=([("Excel XLSX","*.xlsx")]))
else:                   shot_info_spreadsheet = source_dir+"/"+spreadsheet_name
num_ShotData_records  = pd.read_excel(shot_info_spreadsheet, sheet_name="ShotData").index.size
print(f'\nNum sac.txt files    = {num_input_files}')
print(f'Num ShotData records = {num_ShotData_records}')
if messagebox.askyesno("Output directory", "Save outputs to the source directory?"): outfile = source_dir+"/"+line_prefix+"_"+current_time
else: outfile = filedialog.askdirectory()+"/"+line_prefix+"_"+current_time

##########################################
########## READ LOCATION DATA   ##########
##########################################
df2         = pd.read_excel(shot_info_spreadsheet, sheet_name="Positioning")    ## Read in data from the 'Positioning' sheet in the Excel spreadsheet
distance    = df2['Distance'].to_numpy()                                        ## Assign data to a Numpy array
N           = df2['Northing'].to_numpy()                                        ## "                           "  
E           = df2['Easting'].to_numpy()                                         ## "                           "
Z           = df2['Elevation'].to_numpy()                                       ## "                           "
f           = interp1d(distance, Z, fill_value="extrapolate")                   ## Expression to interpolate elevation values 
print(f'\nNum XYZ points = {distance.size}')
print(f'XYZ point sep  = {distance[1]-distance[0]} m')

##################################################
########## CREATE SURFER BLANKING FILE  ##########
##################################################
def MakeBLNfile():
    if BLN_max_DepthIsRL: 
        zMaxDepth = BLN_depth_cutoff                    ## Set lower bound in elevation RL
    else:                                               ##  OR
        zMaxDepth = np.min(Z) - BLN_depth_cutoff        ## Set lower bound as depth below minimum 'z' value 

    blnX = np.insert(distance, [0], distance.size+3)    ## Copy the 'distance' array and insert the number of verticies at start of the array           (BLN file requirement)
    blnY = np.insert(Z, [0], '0')                       ## Copy the 'Z' array and insert a 0 to specify that NoData to be assigned outside the polygon  (BLN file requirement)
    blnX = np.append(blnX, np.max(distance))            ## Set bottom right chainage  (i.e. Xmax Ymin). Third last row col 1
    blnY = np.append(blnY, zMaxDepth)                   ## Set bottom right elevation (i.e. Xmax Ymin). Third last row col 2
    blnX = np.append(blnX, np.min(distance))            ## Set bottom left chainage   (i.e. Xmin Ymin). Second last row col 1
    blnY = np.append(blnY, zMaxDepth)                   ## Set bottom left elevation  (i.e. Xmin Ymin). Second last row col 2
    blnX = np.append(blnX, np.min(distance))            ## Set the final chainage to same as first to close polygon. Last row col 1
    blnY = np.append(blnY, blnY[1])                     ## Set the final elevation to same as first to close polygon. Last row col 2

    np.savetxt(outfile+".BLN", np.c_[blnX,blnY], delimiter=",", fmt="%.3f")      ## Save the BLN file. Numbers rounded to 3 decimal places.
    print(f"\nSaved BLN to file: {outfile}.BLN\n")

###########################################################
########## CREATE DAT FILE FROM sac.txt FILES    ##########
###########################################################
def MakeDATfile():
    df = pd.read_excel(shot_info_spreadsheet, sheet_name="ShotData")    ## Read in data from the 'ShotData' sheet in the Excel spreadsheet
    DEPTH   = np.array([])                                  ## Create empty Numpy arrays
    DIST    = np.array([])                                  ## "                        "
    VS      = np.array([])                                  ## "                        "
    VP      = np.array([])                                  ## "                        "
    RHO     = np.array([])                                  ## "                        "

    for index, row in df.iterrows():    ## Iterate through the rows listed in the 'ShotData' sheet in the Excel spreadsheet
        try:                                ## Check that the file exists in the source directory
            file     = str(row[7])                                                      ## Extract the value in column 7 (i.e. column H) which is the 'Sac File' column
            midpoint = row[5]                                                           ## Extract the value in column 5 (i.e. column F) which is the 'Midpoint (m)' column
            Vs_file  = source_dir+"/"+file+".txt"                                       
            d1  = np.genfromtxt(Vs_file, skip_header=8, skip_footer=7, usecols=0)*1e3   ## Read in layer thickness column and convert km to m    (Negative down from surface for RL)
            d2  = np.cumsum(d1)                                                         ## Sum the layer thickness to calculate the max depth
            Vs  = np.genfromtxt(Vs_file, skip_header=8, skip_footer=7, usecols=2)*1e3   ## Read in the P-wave velocities for each layer and convert km/s to m/s.
            Vp  = np.genfromtxt(Vs_file, skip_header=8, skip_footer=7, usecols=1)*1e3   ## Read in the S-wave velocities for each layer and convert km/s to m/s.
            rho = np.genfromtxt(Vs_file, skip_header=8, skip_footer=7, usecols=3)*1e3   ## Read in Rho for each layer and convert g/cm^3 to kg/m^3
            di  = f(midpoint)                                                           ## Interpolate elevation data at the chainage of the midpoint
            d   = di-d2                         ## Calculate the elevation of the lowest layer
            x   = np.ones_like(Vs)*midpoint     ## Create a chainage array by copying the shape of the Vs array and then mutiplying the midpoint value by 1

            DIST   = np.append(DIST, x[0]);     DIST   = np.append(DIST, x)      ## Add a point at the surface for each shot point then fill the arrays
            DEPTH  = np.append(DEPTH, di);      DEPTH  = np.append(DEPTH, d)     ## "                                                                  "
            VS     = np.append(VS, Vs[0]);      VS     = np.append(VS, Vs)       ## "                                                                  "
            VP     = np.append(VP, Vp[0]);      VP     = np.append(VP, Vp)       ## "                                                                  "
            RHO    = np.append(RHO, rho[0]);    RHO    = np.append(RHO, rho)     ## "                                                                  "

            print(f"Processed file {index+1}/{num_ShotData_records}: {file}")
        except FileNotFoundError:
            messagebox.showwarning("DAT file generator error",(f"The following file could not be found and has been skipped:\n\n{file}.txt"))
            continue
        except IndexError:
            messagebox.showwarning("DAT file generator error",(f"The following file could not be processed and has been skipped:\n\n{file}.txt"))
            continue
 
    np.savetxt(outfile+".DAT", np.c_[DIST,DEPTH,VS,VP,RHO], fmt='%.2f')          ## Save the extracted data to an ASCII file. Numbers rounded to 2 decimal places.
    print(f"\nSaved DAT to file: {outfile}.DAT\n")

############################################################
########## CHECK INPUT DATA AND CALL FUNCTIONS    ##########
############################################################
if num_input_files != num_ShotData_records:
    messagebox.showwarning("Data checker", "The number of '.sac.txt' files does not match the number of records in the ShotData.xlsx spreadsheet.\n\n\Find the missing '.sac.txt' files or delete the relevant rows from the spreadsheet.")
    if messagebox.askyesno("BLN File", f"{num_ShotData_records} 'ShotData' records in spreadsheet.\n\nDo you want to create a BLN file?"):
        MakeBLNfile()
    else: print("No BLN file")
    if messagebox.askyesno("DAT File", f"{num_input_files} '.sac.txt' files found in folder.\n\nDo you want to create a DAT file?"):
        MakeDATfile() 
    else: print("No DAT file")
elif num_input_files == num_ShotData_records:
    if messagebox.askyesno("BLN File", f"{num_ShotData_records} 'ShotData' records in spreadsheet.\n\nDo you want to create a BLN file?"):
        MakeBLNfile()
    else: print("No BLN file")
    if messagebox.askyesno("DAT File", f"{num_input_files} '.sac.txt' files found in folder.\n\nDo you want to create a DAT file?"):
        MakeDATfile() 
    else: print("No DAT file")