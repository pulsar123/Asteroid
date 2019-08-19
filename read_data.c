#include "asteroid.h"

/*  Reading input data files - ephemerides for asteroid, earth, sun, and the brightness curve data.  
*/

int read_data(char *data_file, int *N_data, int *N_filters, int Nplot)
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
 #ifdef INTERP
 struct obs_data_h *hhData = (obs_data_h *)malloc(MAX_DATA * sizeof(struct obs_data_h));
 struct obs_data_h *hhPlot = (obs_data_h *)malloc(Nplot * sizeof(struct obs_data_h));
 #else
 #define hhData hData
 #define hhPlot hPlot
 #endif    
// int N;
  
 // Number of brightness data points:
 fp = fopen(data_file, "r");
 if (!fp)
 {
     printf("Input file %s does not exist!\n", data_file);
     exit(1);
 }
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
// Minus one header line: (???)
//*N_data = *N_data - 1;

if (*N_data > MAX_DATA)
{
    fprintf(stderr,"Error: N_data (%d) > MAX_DATA!\n", *N_data);
    exit(1);
}

 // Allocating the data arrays:
ERR(cudaMallocHost(&hData, *N_data * sizeof(struct obs_data)));
ERR(cudaMallocHost(&MJD_obs, *N_data * sizeof(double)));

#ifdef DUMP_DV
FILE *fpdump = fopen("dV.dat", "w");
#endif
#ifdef DUMP_RED_BLUE
FILE *fpdump = fopen("dump.dat", "w");
#endif

// Reading the input data file
fp = fopen(data_file, "r");
int i = -1;
char filter;
int j = 0;
int k;
int W_filter = -1;
int D_filter = -1;
double sgm, MJD1, V1;
printf("Filters:\n");
while (fgets(line, sizeof(line), fp)) 
{
    i++;
    if (i >= 0)
    {
        sscanf(line, "%c %lf %lf %lf", &filter, &MJD1, &V1, &sgm);
        MJD_obs[i] = MJD1;
        if (i>0 && MJD_obs[i] <= MJD_obs[i-1])
        {
            printf("Error: the data have to be sorted chronologically!\n");
            exit(1);
        }
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
            // A special case - Wesley et al data (time is light travel corrected, and magnitudes are geometry and color corrected)
            if (filter == 'W')
                W_filter = j;
            if (filter == 'D')
                D_filter = j;
            j++;
            printf("%d: %c\n", j, filter);
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
    fprintf(stderr,"Too many filters - increase N_FILTERS parameter! %d\n", *N_filters);
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
        // Corresponding earth observer time - wrong, as my data is normally light travel corrected!
//        MJD0[l] = JD - 2400000.5 + delay;
        // Light travel corrected:
        MJD0[l] = JD - 2400000.5;
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
                quadratic_interpolation(MJD_obs[i], &(hhData[i].E_x), &(hhData[i].E_y), &(hhData[i].E_z), &(hhData[i].S_x), &(hhData[i].S_y), &(hhData[i].S_z));
                //                quadratic_interpolation(MJD_obs[i], &E_x1, &E_y1, &E_z1, &S_x1, &S_y1, &S_z1);
                /*
                hhData[i].E_x = E_x1;
                hhData[i].E_y = E_y1;
                hhData[i].E_z = E_z1;
                hhData[i].S_x = S_x1;
                hhData[i].S_y = S_y1;
                hhData[i].S_z = S_z1;
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
double E, S;
#ifdef SEGMENT
int iseg = 0;
#endif

for (i=0; i<*N_data; i++)    
{
#ifdef SEGMENT
    if (iseg < N_SEG && MJD_obs[i] >= T_START[iseg])
    // We found the start of the next data segment
    {        
        h_start_seg[iseg] = i;
        iseg ++;
    }
#endif

    E = sqrt(hhData[i].E_x*hhData[i].E_x + hhData[i].E_y*hhData[i].E_y+ hhData[i].E_z*hhData[i].E_z);
    S = sqrt(hhData[i].S_x*hhData[i].S_x + hhData[i].S_y*hhData[i].S_y+ hhData[i].S_z*hhData[i].S_z);

    // Convertimg visual magnitudes to absolute magnitudes (at 1 au from sun and earth):
    // W_filter data is skipped, but this does apply to all other filters, including D_filter
    if (hData[i].Filter != W_filter)
        hData[i].V = hData[i].V + 5.0*log10(1.0/E * 1.0/S);
#ifdef DUMP_DV
    fprintf(fpdump, "%f\n", 5.0*log10(1.0/E * 1.0/S));
#endif
    // Computing the delay (light time), in days:
    delay = E / light_speed;
    hData[i].MJD = MJD_obs[i];
    // Converting to asteroidal time (minus light time):
    // Both W_filter and D_filter data is skipped
    if (hData[i].Filter != W_filter && hData[i].Filter != D_filter)
        hData[i].MJD = hData[i].MJD - delay;
#ifdef DUMP_RED_BLUE
    fprintf(fpdump, "W %12.6lf %6.3f %5.3f r\n", hData[i].MJD, hData[i].V, 1.0/sqrt(hData[i].w));
#endif
    if (i == 0)
        hMJD0 = hData[i].MJD;
    hData[i].MJD = hData[i].MJD - hMJD0;
    // Making S,E a unit vector:
    hhData[i].E_x = hhData[i].E_x / E;
    hhData[i].E_y = hhData[i].E_y / E;
    hhData[i].E_z = hhData[i].E_z / E;
    hhData[i].S_x = hhData[i].S_x / S;
    hhData[i].S_y = hhData[i].S_y / S;
    hhData[i].S_z = hhData[i].S_z / S;
    
}



#ifdef DUMP_DV
    fclose(fpdump);
    exit (0);
#endif
#ifdef DUMP_RED_BLUE
    fclose(fpdump);
    exit (0);
#endif
    
#ifdef INTERP
    // Converting ephemeridal time to the same time as used on GPU:
    for (int i=0; i<3; i++)
        MJD0[i] = MJD0[i] - hMJD0;
#endif    

// Computing a fake data set, only for plotting
// Explicitely assuming that ephemeride files contain three data points each    
#ifdef SEGMENT
    iseg = 0;
#endif
if (Nplot > 0)        
{
    ERR(cudaMallocHost(&hPlot, Nplot * sizeof(struct obs_data)));
    // Time step for plotting:
    double h = hData[*N_data-1].MJD / (Nplot - 1);
    double tplot;
    int iplot;
    #ifndef INTERP
    // Changing the ephemeride times:
    for (l=0; l<3; l++)
        MJD0[l] = MJD0[l] - hMJD0;
    #endif    
    
    for (iplot=0; iplot<Nplot; iplot++)
    {
        tplot = iplot * h;
        
        // Handling two end points:
        if (iplot == 0 || iplot == Nplot-1)
        {
            int i;
            if (iplot == 0)
                i = 0;
            else
                i = *N_data - 1;
            hPlot[iplot].MJD = hData[i].MJD;
            hhPlot[iplot].E_x = hhData[i].E_x;
            hhPlot[iplot].E_y = hhData[i].E_y;
            hhPlot[iplot].E_z = hhData[i].E_z;
            hhPlot[iplot].S_x = hhData[i].S_x;
            hhPlot[iplot].S_y = hhData[i].S_y;
            hhPlot[iplot].S_z = hhData[i].S_z;
        }
        else
        {
            hPlot[iplot].MJD = tplot;
            quadratic_interpolation(tplot, &(hhPlot[iplot].E_x), &(hhPlot[iplot].E_y), &(hhPlot[iplot].E_z), &(hhPlot[iplot].S_x), &(hhPlot[iplot].S_y), &(hhPlot[iplot].S_z));
            hPlot[iplot].V = 0.0;            

            E = sqrt(hhPlot[iplot].E_x*hhPlot[iplot].E_x + hhPlot[iplot].E_y*hhPlot[iplot].E_y+ hhPlot[iplot].E_z*hhPlot[iplot].E_z);
            S = sqrt(hhPlot[iplot].S_x*hhPlot[iplot].S_x + hhPlot[iplot].S_y*hhPlot[iplot].S_y+ hhPlot[iplot].S_z*hhPlot[iplot].S_z);
            hhPlot[iplot].E_x = hhPlot[iplot].E_x / E;
            hhPlot[iplot].E_y = hhPlot[iplot].E_y / E;
            hhPlot[iplot].E_z = hhPlot[iplot].E_z / E;
            hhPlot[iplot].S_x = hhPlot[iplot].S_x / S;
            hhPlot[iplot].S_y = hhPlot[iplot].S_y / S;
            hhPlot[iplot].S_z = hhPlot[iplot].S_z / S;
        }
        
#ifdef SEGMENT
        if (iseg < N_SEG && hPlot[iplot].MJD+hMJD0 >= T_START[iseg])
        // We found the start of the next data segment
        {        
            h_plot_start_seg[iseg] = iplot;
            iseg ++;
        }
#endif
        
    }

}

#ifdef INTERP
free(hhData);
free(hhPlot);
#endif
    
return 0;
}
