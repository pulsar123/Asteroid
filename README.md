Code Asteroid was written to model the light curve of the first interstellar visitor, `Oumuamua (https://en.wikipedia.org/wiki/%CA%BBOumuamua), but it can also be used for modeling other minor bodies (asteroids an comets) if they are tumbling and/or experience a fixed torque. The code is described in the paper Mashchenko (2019), https://arxiv.org/abs/1906.03696 . It is written in C/CUDA, and runs on Tesla GPUs starting from 2.0 capability. It is optimized for NVIDIA P100 GPUs. (For some reason, performance is worse on newer V100 GPUs.)


Instructions for the `Oumuamua paper

0) Environment used in the paper
 - CentOS Linux release 7.5.1804
 - CUDA V10.0.130
 - GCC 5.4.0

1) The light curve file
 - Format (each line corresponds to one observation):
```
filter  MJD  V  sgm  Aux
``` 
Here 
 - filter: identifier for a homogeneous set of observations (for example, a specific filter, or uncalibrated observations done homogeneously).
           special value W: presumes that the light curve was already converted to absolute magnitudes and to asteroidal time (light travel corrected)
           Data will be grouped by the filter value, and these separate groups will be fitted separately. Make sure parameter N_FILTERS in asteroid.h is as large
           as the number of different filters in the light curve data
 - MJD: observational time (light travel corrected for "W filter" data points)
 - V: visual magnitude (absolute magnitude for "W filter" data points)
 - sgm: std for the brightness measurement, mag
 - Aux: any character (currently not used, but needs to be present)

2) Three ephemeris files (should be present in the directory where the code is executed). At least three moments of time have to be present, bracketing the
light curve time span. If INTERP macro parameter is used (needed for light curves >490 points), exactly three moments of time have to be present. The positions
of the asteroid, Sun, and Earth will be (second order) interpolated to specific observed times, using the three (or more) ephemeris points.
 - asteroid.eph
 - sun.eph
 - earth.eph
 
 The ephemeris files are generated using the online NASA tool HORIZONS (https://ssd.jpl.nasa.gov/horizons.cgi).
  - Ephemeris Type : VECTORS
  - Target Body: Earth (Geocenter), or Sun, or Asteroid 'Oumuamua (A/2017 U1)  (search for "A/2017 U1")
  - Coordinate Origin :	Solar System Barycenter (SSB)
  - Time Span: Start=JD 2452566.5, Stop=JD 2452621.5, Intervals=2
  - Table settings: Type 1 (x,y,z only), Labels: No
  - Display/Output: plain text

Press Generate Ephemeris, then Ctrl-S to save the text file as one of the three *.eph files

3) The fiducial model from the paper: self-consistent LS brightness ellipsoid with torque, small light curve dataset (<490 points). 
 * To do a relaxed brightness ellipsoid, add "-DBC" option in makefile. 
 * To do a relaxed brightness ellipsoid without torque, add "-DBC" option, and remove the "-DTORQUE" option in makefile. 
 * To do a black-and-white ball with torque, add "-DBW_BALL" in makefile.

 - makefile:
```
 OPT=--ptxas-options=-v -arch=$(ARCH) -DP_PSI -DTORQUE
``` 
  - asteroid.h :
```
const int N_FILTERS = 1;
const double TIME_STEP = 1e-2;  // Change to 5e-3 if used for >5 days integration (e.g. for TD60 asteroid)
```
  - asteroid.c :
```
// Angular momentum L value, radians/day; if P is period in hours, L=48*pi/P
hLimits[0][T_L] = 48.0*PI / 10; // 8.5
hLimits[1][T_L] = 48.0*PI / 0.1; // 0.4    

// Maximum amplitude for torque parameters (units are rad/day)
double Tmax = 10.0;

// c_tumb (physical (tumbling) value of the axis c size; always smallest)
hLimits[0][T_c_tumb] = log(0.01);
hLimits[1][T_c_tumb] = log(1.0);                
    
// b_tumb (physical (tumbling) value of the axis b size; always intermediate), in relative to c_tumb units (between 0: 1, and 1: log(c_tumb))
// For a symmetric cigar / disk, freeze the limits to 1 / 0
hLimits[0][T_b_tumb] = 0;
hLimits[1][T_b_tumb] = 1;
```
 
 - To compile:
```
 make clean; make
```
 - Stage One (random search) run (if used with SLURM scheduler; in other cases, replace $SLURM_JOB_ID with a suitable choice of a unique integer number), 8 instances using 8 GPUs:
```
 ./asteroid  -Nstages 2 -seed $SLURM_JOB_ID -keep  -i light_curve_data  -o output_file  -Ppsi 2 4800
```
 - Stage Two (reoptimization) run ($i is the job number in the GPU farm, or another suitable unique integer number; par1... is a model from the Stage One), 8 instances using 8 GPUs:
```
 ./asteroid -t -N 20 -reopt -best -seed $i -i light_curve_data  -o output_file  -m par1 par2 par3 ...
```
 - Stage Three (fine-tuning; optional). Requires recompiling the code.

 -- makefile (INTERP is only needed if used with >490 points dataset):
```
 OPT=--ptxas-options=-v -arch=$(ARCH) -DP_PSI -DTORQUE  -DACC  -DNUDGE  -DINTERP
```
 -- Create text file in the same directory, observed.min . Each line correspond to an observed minimum. Format:
```
    MJD_min  V_min
```
 Here MJD_min is the time of the minimum, and V_min is the magnitude at the minimum. Can have between 1 and 10 minima, without modifying params in asteroid.h

 -- Execution (par1...  is a model from Stage One or Two):
```
 ./asteroid -t -reopt -best -seed $i -i light_curve_data  -o output_file  -m par1 par2 par3 ...
``` 
 
4) Format of the output file. Each line contains one model, with the following parameters:
```
chi^2  delta_V1  [delta_V2  ...]  par1  par2  par3 ...
```

The number of the fitting parameters delta_V is equal to the number of different filters in the light_curve_data file. The order of the free model parameters par1, par2, ...
is the same as in the table "Property[N_PARAMS][N_COLUMNS]" (at the top of asteroid.c file), taking into account the macro parameters set in the makefile. For example,
self-consistent LS ellipsoid model with torque will have the following 11 parameters:
```
theta_M  phi_M  phi_0  Tb  Tc  Ta  c  b  E'  L  psi_0
```
The units for the parameters are physical (where Ia=1 and a=1, time unit is a day; see paper).

5) The code can be used to create light curve plots for a given model (no need to recompile):
```
./asteroid  -i light_curve_data  -plot  -m par1 par2 par3 ...
```
This will create model.dat file with 20,000 (NPLOT in asteroid.h) model brightness points, using the delta_V1 offset parameter.

6) The code can be used to compute confidence intervals for a given model - either constrained ones (varying one parameter at a time, while keeping the rest at 
the initial values), or unconstrained ones (varying all the free parameters at the same time; dramatically more computationally expensive).

 - Constrained confidence intervals. Requires recompiling the code.
 
 -- makefile: add three more switches:
```
  OPT= ... -DSPHERICAL_K  -DRMSD  -DPROFILES
```
The SPHERICAL_K switch is to convert the torque vector from Cartesian normalized components, T_{b,c,a} to spherical components, K, theta_K, and phi_K, 
which are much more useful for confidence interval calculations.

 -- Execution:
```
 ./asteroid -dx $DX -plot -seed $i -i light_curve_data  -o output_file  -m par1 par2 par3 ...
``` 
Here $DX is the size of the half-interval (in dimensionless units; full interval is 0...1) for each parameter. The code will create multiple new files, 
lines_X.dat, one for each free model parameter. The ordering of the parameters is the one from the table Property[N_PARAMS][N_COLUMNS] described in asteroid.c.
In these files, the initial model is at the middle line (so if there are 2560 lines, the initial model is at the line 1280). There are two columns:
dimensional parameter value, and RMSD value. The allowed interval of RMSD values, as described in the paper Mashchenko (2019), can be used to find the
confidence interval for this parameter.

 - Unconstrained confidence intervals. Requires recompiling the code.
 
 -- makefile: add two more switches:
```
  OPT= ... -DSPHERICAL_K  -DRMSD
```
 -- Execution: a few runs with varying values of the search radius (in scale-free units) $DX: 0.003, 0.01, 0.03, 0.1, 0.3. Each instance runs for 3 hours on P100 GPU.
```
 ./asteroid -dx $DX -Nstages 10000 -reopt -seed $SLURM_JOB_ID -i light_curve_data  -o output_file  -m par1 par2 par3 ...
``` 
Output file contains two columns for each free parameter; parameters are separated by a semicolon. The results are cumulative, so only the last line should be used
for analysis. The two columns are the smallest and largest value of the parameter (dimensional units) corresponding to models with good RMSD values; in other
words, the unconstrained confidence interval for this parameter. One has to find the globally smallest/largest parameter values for each parameter, across
files produced with different search radii ($DX).

7) The code can be used to produce a sequence of PNG images visualizing the asteroid (as seen by observer on Earth), for a specific model.

 -- makefile: add one more switch:
```
  OPT= ... -DANIMATE
```

 -- Execution:
```
 ./asteroid -plot -i light_curve_data  -o output_file  -m par1 par2 par3 ...
``` 
This will create a sequence of NPLOT (see asteroid.h) images, covering the whole simulation period, starting at index 0. One can use optional command line switches
"-i1" and "-i2" to generate a subset of snapshots. By default it will create colored images (yellow in sunlit areas, dark blue in shadows), with a red spot corresponding
to the end of the axis "b" (to visualize rotation around the axis of symmetry). To make scientific images (black and white, with the integrated pixel brightness
proportional to the integrated model brightness), set parameters IMAX_R,G,B to 255, and SMIN_R,G,B and SMAX_R,G,B to 0 (in asteroid.h). To get rid of the red spot,
set SPOT_RAD to 0.0. 

To convert the sequence of PNG images to animation, one can use the ffmpeg command line tool:
```
 ffmpeg -r 60 -f image2 -i image_%05d.png -vcodec libx264 -crf 10 -pix_fmt yuv420p out.mp4
```
