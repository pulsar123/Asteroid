#include "asteroid.h"

/*  Reading input data files - ephemerides for asteroid, earth, sun, and the brightness curve data.  
*/

int read_data(char *data_file, int *N_data, int *N_filters)
{
 FILE *fp;
 FILE *fpA;
 FILE *fpE;
 FILE *fpS;
// char filename[MAX_FILE_NAME];
 char line[MAX_LINE_LENGTH];
 char lineA[MAX_LINE_LENGTH];
 char lineE[MAX_LINE_LENGTH];
 char lineS[MAX_LINE_LENGTH];
 char ch;
// int N;
  
 // Number of brightness data points:
 fp = fopen(data_file, "r");
 *N_data = 0;
 while(!feof(fp))
{
  ch = fgetc(fp);
  if(ch == '\n')
  {
    *N_data = *N_data + 1;
  }
}
fclose(fp);
// Minus one header line:
*N_data = *N_data - 1;

 // Allocating the data arrays:
ERR(cudaMallocHost(&hData, *N_data * sizeof(struct obs_data)));
ERR(cudaMallocHost(&MJD_obs, *N_data * sizeof(double)));

// Reading the input data file
fp = fopen(data_file, "r");
int i = -1;
char filter;
int j = 0;
int k;
int p_filter = -1;
double sgm, MJD1, V1;
printf("Filters:\n");
while (fgets(line, sizeof(line), fp)) 
{
    i++;
    if (i >= 0)
    {
        sscanf(line, "%c %lf %lf %lf", &filter, &MJD1, &V1, &sgm);
        MJD_obs[i] = MJD1;
        hData[i].V = V1;
        hData[i].w = 1.0/(sgm*sgm);
        // Finding all unique filters:
        int found = 1;
        for (k=0; k<j; k++)
        {
            if (filter == all_filters[k])
            {
                found = 0;
                break;
            }                
        }
        if (found)
        {
            all_filters[j] = filter;
            j++;
            printf("%d: %c\n", j, filter);
            // The special case of the "p" filter (from Drahus et al.; presumably already geometry corrected)
            if (filter == 'p')
                p_filter = j;
        }
        // Translating filter char to filter number:
        for (k=0; k<j; k++)
        {
            if (filter == all_filters[k])
                hData[i].Filter = k;
        }
    }
}
*N_filters = j;
if (*N_filters > N_FILTERS)
{
    printf("Too many filters - increase N_FILTERS parameter! %d\n", *N_filters);
    exit (1);
}

fclose(fp);

// Reading the three ephemerides files and computing the data values:
fpA = fopen("asteroid.eph", "r");
fpE = fopen("earth.eph", "r");
fpS = fopen("sun.eph", "r");
// Pointing to the data portion in each file
while (fgets(lineA, sizeof(lineA), fpA))
{
    if (strcmp(lineA, "$$SOE\n") == 0)
        break;
}
while (fgets(lineE, sizeof(lineE), fpE))
{
    if (strcmp(lineE, "$$SOE\n") == 0)
        break;
}
while (fgets(lineS, sizeof(lineS), fpS))
{
    if (strcmp(lineS, "$$SOE\n") == 0)
        break;
}

j = -1;
double JD;
double Xa,Ya,Za, Xe,Ye,Ze, Xs,Ys,Zs;
double delay;
int l = 0;
int m;
i = 0;
while (fgets(lineA, sizeof(lineA), fpA))
{
    // The marking for the end of data:
    if (strcmp(lineA, "$$EOE\n") == 0)
        break;
    j++;
    fgets(lineE, sizeof(lineE), fpE);
    fgets(lineS, sizeof(lineA), fpS);
    if (j % 2 == 0)
        // Even data lines contain JD
    {
        sscanf(lineA, "%lf", &JD);
    }
    else
        // Odd data lines contain X,Y,Z
    {
        if (l > 2)
        {
            // Shifting the arrays to the left by one
            for (m=0; m<2; m++)
            {
                E_x0[m] = E_x0[m+1];
                E_y0[m] = E_y0[m+1];
                E_z0[m] = E_z0[m+1];
                S_x0[m] = S_x0[m+1];
                S_y0[m] = S_y0[m+1];
                S_z0[m] = S_z0[m+1];
                MJD0[m] = MJD0[m+1];
            }
            l = 2;
        }
        sscanf(lineA, "%lE %lE %lE", &Xa, &Ya, &Za);
        sscanf(lineE, "%lE %lE %lE", &Xe, &Ye, &Ze);
        sscanf(lineS, "%lE %lE %lE", &Xs, &Ys, &Zs);
        // Asteroid -> Earth vector:
        E_x0[l] = Xe - Xa;
        E_y0[l] = Ye - Ya;
        E_z0[l] = Ze - Za;
        // Asteroid -> Sun vector:
        S_x0[l] = Xs - Xa;
        S_y0[l] = Ys - Ya;
        S_z0[l] = Zs - Za;
//printf("AA  %20.12lf %20.12lf\n",MJD0[0],E_x0[0]);
        // Computing the delay (light time), in days:
        delay = sqrt(E_x0[l]*E_x0[l] + E_y0[l]*E_y0[l]+ E_z0[l]*E_z0[l]) / light_speed;
        // Corresponding earth observer time:
        MJD0[l] = JD - 2400000.5 + delay;
        l++;
        if (l > 2)
        {
            // Using "while" here as there may be more than one data point corresponding to the given MJD0 bracket:
            //!!! [1]...[2]
            while (i<*N_data && MJD_obs[i]>=MJD0[0] && MJD_obs[i]<MJD0[2])
                // We just found the MJD0[1-2] interval bracketing the i-th data point
            {
                // Using the quadratic Lagrange polynomial to do second degree interpolation for E and S vector components
//                double E_x1, E_y1, E_z1, S_x1, S_y1, S_z1;
                quadratic_interpolation(MJD_obs[i], &(hData[i].E_x), &(hData[i].E_y), &(hData[i].E_z), &(hData[i].S_x), &(hData[i].S_y), &(hData[i].S_z));
                //                quadratic_interpolation(MJD_obs[i], &E_x1, &E_y1, &E_z1, &S_x1, &S_y1, &S_z1);
                /*
                hData[i].E_x = E_x1;
                hData[i].E_y = E_y1;
                hData[i].E_z = E_z1;
                hData[i].S_x = S_x1;
                hData[i].S_y = S_y1;
                hData[i].S_z = S_z1;
                */
//printf("%20.12lf %20.12lf %20.12lf %20.12lf %20.12lf %20.12lf %20.12lf\n", (*MJD)[i], (*E_x)[i],(*E_y)[i],(*E_z)[i],(*S_x)[i],(*S_y)[i],(*S_z)[i]);                
                // Switching to the next data point:
                i++;
            }
        }
    }
    if (i >= *N_data)
        // No more data points; exiting
        break;
}
fclose(fpA);
fclose(fpE);
fclose(fpS);

// Converting the observed data
double E, S, E0, S0;
for (i=0; i<*N_data; i++)
{
    E = sqrt(hData[i].E_x*hData[i].E_x + hData[i].E_y*hData[i].E_y+ hData[i].E_z*hData[i].E_z);
    S = sqrt(hData[i].S_x*hData[i].S_x + hData[i].S_y*hData[i].S_y+ hData[i].S_z*hData[i].S_z);
    if (i == 0)
    {
        E0 = E;
        S0 = S;
    }
    // Convertimg visual magnitudes to the asteroid/Earth/Sun distances at the first observed moment:
    if (hData[i].Filter != p_filter)
        hData[i].V = hData[i].V + 5.0*log10(E0/E * S0/S);
    // Computing the delay (light time), in days:
    delay = E / light_speed;
    // Converting to asteroidal time (minus light time):
    hData[i].MJD = MJD_obs[i] - delay;
    if (i == 0)
        hMJD0 = hData[i].MJD;
    hData[i].MJD = hData[i].MJD - hMJD0;
    // Making S,E a unit vector:
    hData[i].E_x = hData[i].E_x / E;
    hData[i].E_y = hData[i].E_y / E;
    hData[i].E_z = hData[i].E_z / E;
    hData[i].S_x = hData[i].S_x / S;
    hData[i].S_y = hData[i].S_y / S;
    hData[i].S_z = hData[i].S_z / S;
}

return 0;
}
