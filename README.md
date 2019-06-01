Instructions for the `Oumuamua paper, Mashchenko (2019)

1) The light curve file
 - Format (each line corresponds to one observation):
```
filter  MJD  V  sgm  Aux
``` 
Here 
 - filter: identifier for a homogeneous set of observations (for example, a specific filter, or uncalibrated observations done homogeneously).
           special value W: if the light curve was already converted to absolute magnitudes and to asteroidal time (light travel corrected)
           Data will be grouped by the filter value, and these separate groups will be fitted separately. Make sure parameter N_FILTERS in asteroid.h is as large
           as the number of different filters in the light curve data
 - MJD: observational time (light travel corrected with W filter)
 - V: visual magnitude (absolute magnitude with W filter)
 - sgm: std for the brightness measurement, mag
 - Aux: any character (not used, but needs to be present)

2) Three ephemeris files (should be present in the directory where the code is executed). At least three moments of time have to be present, bracketing the
light curve time span. If INTERP macro parameter is used (needed for light curves >490 points), exactly three moments of time have to be present. The positions
of the asteroid, Sun, and Earth will be interpolated (second order) for specific observed times, using the three (or more) ephemeris points.
 - asteroid.eph
 - sun.eph
 - earth.eph
 
 The ephemeris files are generated using the online NASA tool HORIZONS (https://ssd.jpl.nasa.gov/horizons.cgi).
  - Ephemeris Type : VECTORS
  - Target Body: Earth [Geocenter] [399], or Sun [Sol] [10], or Asteroid 'Oumuamua (A/2017 U1)  (search for "A/2017 U1")
  - Coordinate Origin :	Solar System Barycenter (SSB) [500@0]
  - Time Span: Start=JD 2452566.5, Stop=JD 2452621.5, Intervals=2
  - Table settings: Type 1 (x,y,z only), Labels: No
  - Display/Output: plain text
  
Press Generate Ephemeris, then Ctrl-S to save the text file as one of the three *.eph files

3) The fiducial model from the paper: self-consistent LS brightness ellipsoid with torque, small light curve dataset (<490 points). 
 * To do a relaxed brightness ellipsoid, add "-DBC" option in makefile. 
 * To do a relaxed brightness ellipsoid with zero torque, add "-DBC" option, and remove the "-DTORQUE" option in makefile. 
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
 - Stage One (random search) run (if used with SLURM scheduler; in other cases, replace $SLURM_JOB_ID with a suitable choice of a unique integer number):
```
 ./asteroid  -Nstages 2 -seed $SLURM_JOB_ID -keep  -i light_curve_data  -o output_file  -Ppsi 2 4800
```
 - Stage Two (reoptimization) run ($i is the job number in the GPU farm, or another suitable unique integer number; par1 ... par11 is a model from the Stage One):
```
 ./asteroid -t -N 20 -reopt -best -seed $i -i light_curve_data  -o output_file  -m par1 par2 par3 ... par11
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

 -- Execution (par1 ... par11 is a model from Stage One or Two):
```
 ./asteroid -t -reopt -best -seed $i -i light_curve_data  -o output_file  -m par1 par2 par3 ... par11
``` 
 
 
 
