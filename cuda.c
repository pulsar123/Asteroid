/* CUDA stuff.
 * Computing chi^2 on GPU for a given combination of free parameters
 * 
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "asteroid.h"



__device__ void ODE_func (double y[], double f[], double mu[])
/* Three ODEs for the tumbling evolution of the three Euler angles, phi, theta, and psi.
 *   Derived in a manner similar to Kaasalainen 2001, but for the setup of Samarasinha and A'Hearn 1991
 *   (a > b > c; Il > Ii > Is; either a or c can be the axis of rotation). This is so called "L-convention"
 *   (Samarasinha & Mueller 2015).
 *   
 *   The setup of Kaasalainen 2001 results
 *   in large derivatives, and low accuracy and instability of ODEs for small I1.
 *   
 *   Here Ip=(1/Ii+1/Is)/2; Im=(1/Ii-1/Is)/2.
 */

{    
    #ifdef TORQUE
    // Simplest constant (in the co-moving frame) torque model applied to a tumbling atseroid. Solving 6 ODEs - three Euler equations of motion
    // in the presence of constant torque, and three more to get the Euler angles phi, psi, theta.
    
    // mu[0]: (Is-Il)/Ii
    // mu[1]: (Il-Ii)/Is
    // mu[2]: (Ii-Is)/Il
    // mu[3]: Ki/Ii
    // mu[4]: Ks/Is
    // mu[5]: Kl/Il
    
    // y[0]: Omega_i
    // y[1]: Omega_s
    // y[2]: Omega_l
    // y[3]: phi
    // y[4]: theta
    // y[5]: psi
    
    // Three Euler equations of motion:
    // d Omega_i /dt:
    f[0] = mu[0]*y[1]*y[2] + mu[3];
    // d Omega_s /dt:
    f[1] = mu[1]*y[2]*y[0] + mu[4];
    // d Omega_l /dt:
    f[2] = mu[2]*y[0]*y[1] + mu[5];
    
    // Three ODEs for the three Euler angles
    // dphi/dt:
    f[3] = (y[0]*sin(y[5]) + y[1]*cos(y[5])) / sin(y[4]);
    // dtheta/dt:
    f[4] =  y[0]*cos(y[5]) - y[1]*sin(y[5]);
    // dpsi/dt:
    f[5] = y[2] - f[3]*cos(y[4]);
    
    #else    
    
    // No torque case (so we don't need to use the Euler's equations)
    // mu[0,1,2] -> L, Ip, Im
    //    double phi = y[0];
    //    double theta = y[1];
    //    double psi = y[2];
    
    // dphi/dt:
    f[0] = mu[0]*(mu[1]-mu[2]*cos(2.0*y[2]));
    
    // dtheta/dt:
    f[1] = mu[0]*mu[2]*sin(y[1])*sin(2.0*y[2]);
    
    // dpsi/dt (assuming Il=1):
    f[2] = cos(y[1])*(mu[0]-f[0]);
    #endif  // TORQUE
}



__device__ CHI_FLOAT chi2one(double *params, struct obs_data *sData, int N_data, int N_filters, CHI_FLOAT *delta_V, int Nplot, struct chi2_struct *sp,
#ifdef ANIMATE
                             unsigned char * d_rgb,
#endif                             
                             int sTypes[][N_SEG])
// Computung chi^2 for a single model parameters combination, on GPU, by a single thread
// NUDGE is not supported in SEGMENT mode!
{
    int i, m;
    double Ep_b, Ep_c, Ep_a, Sp_b, Sp_c, Sp_a;
    CHI_FLOAT chi2a;
    double sum_y2[N_FILTERS];
    double sum_y[N_FILTERS];
    double sum_w[N_FILTERS];
    
    
    for (m=0; m<N_filters; m++)
    {
        sum_y2[m] = 0.0;
        sum_y[m] = 0.0;
        sum_w[m] = 0.0;
    }
    
    /*  Tumbling model description:
     *     
     *     Triaxial ellipsoid with physical axes a, b, c. a and c are extremal ,
     *     b is always intermediate. (c < b < a=1) Photometric a,b,c can be totally different.
     * 
     *     The corresponding moments of inertia are Il (a), Ii (b), Is(c); Il < Ii < Is.
     *     
     *     The frame of reference is that of Samarasinha and A'Hearn 1991: b-c-a (i-s-l) stands for x-y-z
     *     
     *     Input parameters with fixed constraints (5-6): 
     *      - <M>: angular momentum vector described by params.theta_M and params.phi_M
     *      - phi_0: initial Euler angle for precession, 0...2*pi
     *      - L: angular momentum L value, radians/day; if P is perdiod in hours, L=48*pi/P
     *      - c_tumb: log10 of the physical (tumbling) value of the smallest axis c size; c < b < a=1
     *      - A (only in TREND mode): scaling parameter "A" for de-trending the brightness curve, in magnitude/radian units (to be multiplied by the phase angle alpha to get magnitude correction)
     *     TORQUE parameters:
     *      - Ti, Ts, Tl: derivatives for the corresponding Omega_i,s,l angles; units are rad/day
     *      
     *     Derived values:
     *      - Ii_inv = 1/Ii; inverse principle moment of inertia Ii
     *      - Is_inv = 1/Is; inverse principle moment of inertia Is
     *      
     *     Parameters which are constrained by other parameters (3):
     *      - b_tumb: log10 of the physical (tumbling) value of the intermediate axis b size; constrained by c: log10(b)=log10(c)...0
     *      - Es: dimensionless total energy, constrained by Ii: SAM: Es<1/Ii; LAM: Es>1/Ii
     *      - psi_0: initial Euler angle of rotation of the body, constrained (only for SAM) by Ii, Is, Einv=1/Es: psi_max=atan(sqrt(Ii*(Is-Einv)/Is/(Einv-Ii))); psi_0=[-psi_max,psi_max]
     *      
     *     Derived values:
     *      - theta_0: initial Euler nutation angle; 0 ... pi range, derived from Ii, Is, Es, psi_0
     *     
     *     Time integration to compute the three Euler angles (phi, psi, theta) values for all observed data points
     *      - Initial conditions: phi_0, psi_0, theta_0
     *      - Parameters needed: L, Ip=0.5*(Ii_inv+Is_inv); Im=0.5*(Ii_inv-Is_inv);  
     *      - time step - macro parameter TIME_STEP (days)
     *     
     */
    
    // We work in the inertial observed (Solar barycentric) frame of reference X, Y, Z.
    // By applying sequentially the three Euler angles of the asteroid (which we compute by solving the three ODEs numerically),
    // we derive the asteroid's internal axes (coinciding with b, c, a for x, y, z, or i, s, l)
    // orienation in the barycentric frame of reference. This allows us to compute the orientation of the asteroid->sun and asteroid->earth
    // vectors in the asteroid's frame of reference, which is then used to compute it's apparent brightness for the observer.

    #ifdef NUDGE    
    int M = 0;
    float t_mod[M_MAX], V_mod[M_MAX];
    float t_old[2];
    float V_old[2];
    #endif

    #ifdef MINIMA_TEST
    int N_minima = 0;    
    float Vmin[MAX_MINIMA];
    float t_old[2];
    float V_old[2];
    #endif    
    
    // Loop for multiple data segments
    // (Will use one segment, for all the data, when SEGMENT is not defined)
    for (int iseg=0; iseg<N_SEG; iseg++)
    {
        // Defining these for better readability:
        #define P_theta_M  params[sTypes[T_theta_M][iseg]]
        #define P_phi_M    params[sTypes[T_phi_M][iseg]]
        #define P_phi_0    params[sTypes[T_phi_0][iseg]]
        #define P_L        params[sTypes[T_L][iseg]]
        #define P_A        params[sTypes[T_A][iseg]]
        #define P_Ti       params[sTypes[T_Ti][iseg]]
        #define P_Ts       params[sTypes[T_Ts][iseg]]
        #define P_Tl       params[sTypes[T_Tl][iseg]]
        #define P_T2i      params[sTypes[T_T2i][iseg]]
        #define P_T2s      params[sTypes[T_T2s][iseg]]
        #define P_T2l      params[sTypes[T_T2l][iseg]]
        #define P_Tt       params[sTypes[T_Tt][iseg]]
        #define P_c_tumb   params[sTypes[T_c_tumb][iseg]]
        #define P_b_tumb   params[sTypes[T_b_tumb][iseg]]
        #define P_Es       params[sTypes[T_Es][iseg]]
        #define P_psi_0    params[sTypes[T_psi_0][iseg]]
        #define P_c        params[sTypes[T_c][iseg]]
        #define P_b        params[sTypes[T_b][iseg]]
        #define P_theta_R  params[sTypes[T_theta_R][iseg]]
        #define P_phi_R    params[sTypes[T_phi_R][iseg]]
        #define P_psi_R    params[sTypes[T_psi_R][iseg]]
        #define P_kappa    params[sTypes[T_kappa][iseg]]
        
        
        // Calculations which are time independent:            
        
        // In tumbling mode, the vector M is the angular momentum vector (fixed in the inertial - barycentric - frame of reference)
        // It is defined by the two angles (input parameters) - params.theta_M and params.phi_M
        // It's a unit vector
        double M_x = sin(P_theta_M) * cos(P_phi_M);
        double M_y = sin(P_theta_M) * sin(P_phi_M);
        double M_z = cos(P_theta_M);
        
        // In the new inertial frame of reference with the <M> vector being the z-axis, we arbitrarily choose the x-axis, <XM>, to be [y x M].
        // It is fixed in the inertial frame of reference, so can be computed here:
        // Made a unit vector
        double XM = sqrt(M_z*M_z+M_x*M_x);
        double XM_x = M_z / XM;
        //    double XM_y = 0.0;
        double XM_z = -M_x / XM;
        // The third axis, YM, is derived as [M x XM]; a unit vector by design:
        double YM_x = M_y*XM_z;
        double YM_y = M_z*XM_x - M_x*XM_z;
        double YM_z = -M_y*XM_x;
        
        // We set Il (moment of inertia corresponding to the largest axis, a) to 1.
        // Shortest axis (c), largest moment of inertia:
        double Is = (1.0 + P_b_tumb*P_b_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
        // Intermediate axis (b), intermediate moment of inertia:
        double Ii = (1.0 + P_c_tumb*P_c_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
        double Is_inv = 1.0 / Is;
        double Ii_inv = 1.0 / Ii;
        
        // Now we have a=1>b>c, and Il=1<Ii<Is
        // Axis of rotation can be either "a" (LAM) or "c" (SAM)
        
        // Initial Euler angles values:
        double phi = P_phi_0;
        // Initial value of the Euler angle theta is determined by other parameters:
        // It's ok to use only plus sign in from of sqrt, because theta changes in [0..pi] interval
        double theta = asin(sqrt((P_Es-1.0)/(sin(P_psi_0)*sin(P_psi_0)*(Ii_inv-Is_inv)+Is_inv-1.0)));    
        double psi = P_psi_0;
        
        #ifdef TORQUE
        double mu[6];
        // Parameters for the ODEs (don't change with time):
        // Using the fact that Il = 1:
        mu[0] = (Is-1.0)*Ii_inv;
        mu[1] = (1.0-Ii)*Is_inv;
        mu[2] = Ii - Is;
        mu[3] = P_Ti;
        mu[4] = P_Ts;
        mu[5] = P_Tl;
        // Initial values for the three components of the angular velocity vector in the asteroid's frame of reference
        // Initially this vector's orientation is determined by the initial Euler angles, theta, psi, and phi
        double Omega_i = P_L * Ii_inv * sin(theta) * sin(psi);
        double Omega_s = P_L * Is_inv * sin(theta) * cos(psi);
        double Omega_l = P_L * cos(theta);
        #else
                
        double mu[3];
        double Ip = 0.5*(Ii_inv + Is_inv);
        double Im = 0.5*(Ii_inv - Is_inv);
        mu[0] = P_L;
        mu[1] = Ip;
        mu[2] = Im;    
        #endif
        
        #ifdef MIN_DV
        double Vmin = 1e20;
        double Vmax = -1e20;
        #endif
        
        int i1, i2;
        #ifdef SEGMENT
        i1 = sp->start_seg[iseg];
        if (iseg < N_SEG-1)
            i2 = sp->start_seg[iseg+1];
        else
            i2 = N_data;
        #else
        i1 = 0;
        i2 = N_data;
        #endif
        
        #ifdef ANIMATE
        int i1_rgb = d_i1;
        int i2_rgb = d_i2;
        #endif
        
        // The loop over all data points in the current segment 
        for (i=i1; i<i2; i++)
        {                                
            
            // Derive the three Euler angles theta, phi, psi here, by solving three ODEs numerically
            if (i > i1)
            {
                int N_steps;
                double h;
                OBS_TYPE t1 = sData[i-1].MJD;
                OBS_TYPE t2 = sData[i].MJD;
                
                #ifdef TORQUE2
                // The split point (in time) between the two torque regimes (can vary between sData[i1].MJD and sData[i2-1].MJD):
                OBS_TYPE t_split = P_Tt*(sData[i2-1].MJD - sData[i1].MJD) + sData[i1].MJD;
                int Nsplit;
                if (t_split >= t1 && t_split < t2)
                    // We are in the split time interval (when torque changes inside the interval), so need to run the ODE loop twice - 
                    // before and after the torque change
                    Nsplit = 2;
                else
                    Nsplit = 1;
                for (int isplit=0; isplit<Nsplit; isplit++)
                {
                    if (Nsplit == 2)
                    {
                        if (isplit == 0)
                        {
                            t2 = t_split;
                        }
                        else
                        {
                            t1 = t_split;
                            t2 = sData[i].MJD;
                            // Right after the split point, changing the torque parameters to th second set:
                            mu[3] = P_T2i;
                            mu[4] = P_T2s;
                            mu[5] = P_T2l;
                        }
                    }
                        
                #endif
                
                // How many integration steps to the current (i-th) observed value, from the previous (i-1) one:
                // Forcing the maximum possible time step of TIME_STEP days (macro parameter), to ensure accuracy
                N_steps = (t2 - t1) / TIME_STEP + 1;
                // Current equidistant time steps (h<=TIME_STEP):
                h = (t2 - t1) / N_steps;
                
                // Initial values for ODEs variables = the old values, from the previous i cycle:
                #ifdef TORQUE
                const int N_ODE = 6;
                double y[6];
                y[0] = Omega_i;
                y[1] = Omega_s;
                y[2] = Omega_l;
                y[3] = phi;
                y[4] = theta;
                y[5] = psi;
                #else
                const int N_ODE = 3;
                double y[3];
                y[0] = phi;
                y[1] = theta;
                y[2] = psi;
                #endif            
                
                #ifdef PLOT_OMEGA
                double K1[N_ODE];
                #endif
                
                // RK4 method for solving ODEs with a fixed time step h
                for (int l=0; l<N_steps; l++)
                {
                    double f[N_ODE], K2[N_ODE], K3[N_ODE], K4[N_ODE];
                    #ifndef PLOT_OMEGA
                    double K1[N_ODE];
                    #endif
                    
                    ODE_func (y, K1, mu);
                    
                    int j;
                    for (j=0; j<N_ODE; j++)
                        f[j] = y[j] + 0.5*h*K1[j];
                    ODE_func (f, K2, mu);
                    
                    for (j=0; j<N_ODE; j++)
                        f[j] = y[j] + 0.5*h*K2[j];
                    ODE_func (f, K3, mu);
                    
                    for (j=0; j<N_ODE; j++)
                        f[j] = y[j] + h*K3[j];
                    ODE_func (f, K4, mu);
                    
                    for (j=0; j<N_ODE; j++)
                        y[j] = y[j] + 1/6.0 * h *(K1[j] + 2*K2[j] + 2*K3[j] + K4[j]);
                }
                
                
                // New (current) values of the ODEs variables derived from solving the ODEs:
                #ifdef TORQUE
                Omega_i = y[0];
                Omega_s = y[1];
                Omega_l = y[2];
                phi     = y[3];
                theta   = y[4];
                psi     = y[5];
                #ifdef PLOT_OMEGA
                    if (Nplot > 0)
                    {
                        double phi_dot = K1[3];
                        double theta_dot = K1[4];
                        double psi_dot = K1[5];
                        // Computing angular velocity vector components in inertial coordinate system XYZ
                        double Omega_X = psi_dot*sin(theta)*sin(phi) + theta_dot*cos(phi);
                        double Omega_Y =-psi_dot*sin(theta)*cos(phi) + theta_dot*sin(phi);
                        double Omega_Z = psi_dot*cos(theta) + phi_dot;
                        // Converting to spherical inertial coordinate system:
                        // Absolute magnitude Omega:
                        d_Omega[0][i] = sqrt(Omega_X*Omega_X + Omega_Y*Omega_Y + Omega_Z*Omega_Z);
                        // Polar angle theta_Omega:
                        d_Omega[1][i] = acos(Omega_Z/d_Omega[0][i]);
                        // Azimuthal angle phi_Omega:
                        d_Omega[2][i] = atan2(Omega_Y, Omega_X);
                        // In comoving spherical coordinate system:
                        // Absolute magnitude Omega:
                        d_Omega[3][i] = sqrt(Omega_i*Omega_i + Omega_s*Omega_s + Omega_l*Omega_l);
                        // Polar angle theta_Omega:
                        d_Omega[4][i] = acos(Omega_l/d_Omega[3][i]);
                        // Azimuthal angle phi_Omega:
                        d_Omega[5][i] = atan2(Omega_s, Omega_i);
                    }
                #endif
                #ifdef LAST
                if (Nplot>0 && i==Nplot-1)
                // Preserving the final values of L nd E:
                {
                    double L_last = sqrt(Omega_i*Omega_i*Ii*Ii + Omega_s*Omega_s*Is*Is + Omega_l*Omega_l);
                    double E_last = 1 + 1/(L_last*L_last) * (sin(psi)*sin(psi)*(Ii_inv-Is_inv)+Is_inv-1) * (Omega_i*Omega_i*Ii*Ii + Omega_s*Omega_s*Is*Is);
                    d_L_last = L_last;
                    d_E_last = E_last;
                }
                #endif
                #else
                phi = y[0];
                theta = y[1];
                psi = y[2];                    
                #endif
                
                #ifdef TORQUE2
                }  // isplit loop
                #endif
            }                
            
            // At this point we know the three Euler angles for the current moment of time (data point) - phi, theta, psi.
            
            double cos_phi = cos(phi);
            double sin_phi = sin(phi);
            
            // Components of the node vector N=[M x a], derived by rotating vector XM towards vector YM by Euler angle phi
            // It is unit by design
            // Using XM_y = 0
            double N_x = XM_x*cos_phi + YM_x*sin_phi;
            double N_y =                YM_y*sin_phi;
            double N_z = XM_z*cos_phi + YM_z*sin_phi;
            
            // Vector p=[N x M]; a unit one
            double p_x = N_y*M_z - N_z*M_y;
            double p_y = N_z*M_x - N_x*M_z;
            double p_z = N_x*M_y - N_y*M_x;
            
            double cos_theta = cos(theta);
            double sin_theta = sin(theta);
            
            // Vector of rotation <a> (the longest axes of the ellipsoid; x3; z; l) is derived by rotating <M> by Euler angle theta towards <p>,
            // with the node vector <N> being the rotation vector (Rodrigues formula); a unit vector
            double a_x = M_x*cos_theta + p_x*sin_theta;
            double a_y = M_y*cos_theta + p_y*sin_theta;
            double a_z = M_z*cos_theta + p_z*sin_theta;
            
            // Vector w=[a x N]; a unit one
            double w_x = a_y*N_z - a_z*N_y;
            double w_y = a_z*N_x - a_x*N_z;
            double w_z = a_x*N_y - a_y*N_x;
            
            double sin_psi = sin(psi);
            double cos_psi = cos(psi);
            
            // Second axis of the ellipsoid, b (x1; x; i); a unit vector; derived by rotating <N> by Euler angle psi towards <w>,
            // with vector <a> being the rotation axis
            double b_x = N_x*cos_psi + w_x*sin_psi;
            double b_y = N_y*cos_psi + w_y*sin_psi;
            double b_z = N_z*cos_psi + w_z*sin_psi;
            
            // Third ellipsoid axis c (x2; y; s) - the shortest one; c=[a x b]; unit vector by design
            double c_x = a_y*b_z - a_z*b_y;
            double c_y = a_z*b_x - a_x*b_z;
            double c_z = a_x*b_y - a_y*b_x;
            
            #if defined(ROTATE) || defined(BW_BALL)
            // Optional rotation of the brightness ellipsoid relative to the kinematic ellipsoid, using angles theta_R, phi_R, psi_R
            // Using the same setup, as for the main Euler rotation, above (substituting XM->b, YM->c, M->a)
            // The meaning of the vectors b,c,a is changing here. At the end these are the new (rotated) basis.
            cos_phi = cos(P_phi_R);
            sin_phi = sin(P_phi_R);

            N_x = b_x*cos_phi + c_x*sin_phi;
            N_y = b_y*cos_phi + c_y*sin_phi;
            N_z = b_z*cos_phi + c_z*sin_phi;
            
            p_x = N_y*a_z - N_z*a_y;
            p_y = N_z*a_x - N_x*a_z;
            p_z = N_x*a_y - N_y*a_x;
            
            cos_theta = cos(P_theta_R);
            sin_theta = sin(P_theta_R);
            
            // Vector a is changing meaning - now it is the rotated one:
            a_x = a_x*cos_theta + p_x*sin_theta;
            a_y = a_y*cos_theta + p_y*sin_theta;
            a_z = a_z*cos_theta + p_z*sin_theta;
            #endif
            
            #ifdef ROTATE
            w_x = a_y*N_z - a_z*N_y;
            w_y = a_z*N_x - a_x*N_z;
            w_z = a_x*N_y - a_y*N_x;
            
            sin_psi = sin(P_psi_R);
            cos_psi = cos(P_psi_R);
            
            // Vector b is changing meaning - now it is the rotated one:
            b_x = N_x*cos_psi + w_x*sin_psi;
            b_y = N_y*cos_psi + w_y*sin_psi;
            b_z = N_z*cos_psi + w_z*sin_psi;
            
            // Vector c is changing meaning - now it is the rotated one:
            c_x = a_y*b_z - a_z*b_y;
            c_y = a_z*b_x - a_x*b_z;
            c_z = a_x*b_y - a_y*b_x;
            #endif  // ROTATE
            
            // Now following Muinonen & Lumme, 2015 to compute the visual brightness of the asteroid.
            // Attention! My (Samarasinha and A'Hearn 1991) frame of reference is b-c-a, but the Muinonen's frame is a-b-c
            // On 17.10.2018 the bug was fixed, and now I properly convert the Muinonen's equations to the b-c-a frame
                        
            #ifdef INTERP
            // Using Sun and Earth coordinates interpolated in situ
            double rr[3];
            double E_x1,E_y1,E_z1, S_x1,S_y1,S_z1;
    
            // Quadratic interpolation:
            rr[0] = (sData[i].MJD-sp->MJD0[1]) * (sData[i].MJD-sp->MJD0[2]) / (sp->MJD0[0]-sp->MJD0[1]) / (sp->MJD0[0]-sp->MJD0[2]);
            rr[1] = (sData[i].MJD-sp->MJD0[0]) * (sData[i].MJD-sp->MJD0[2]) / (sp->MJD0[1]-sp->MJD0[0]) / (sp->MJD0[1]-sp->MJD0[2]);
            rr[2] = (sData[i].MJD-sp->MJD0[0]) * (sData[i].MJD-sp->MJD0[1]) / (sp->MJD0[2]-sp->MJD0[0]) / (sp->MJD0[2]-sp->MJD0[1]);
            E_x1 = sp->E_x0[0]*rr[0] + sp->E_x0[1]*rr[1] + sp->E_x0[2]*rr[2];
            E_y1 = sp->E_y0[0]*rr[0] + sp->E_y0[1]*rr[1] + sp->E_y0[2]*rr[2];
            E_z1 = sp->E_z0[0]*rr[0] + sp->E_z0[1]*rr[1] + sp->E_z0[2]*rr[2];
            S_x1 = sp->S_x0[0]*rr[0] + sp->S_x0[1]*rr[1] + sp->S_x0[2]*rr[2];
            S_y1 = sp->S_y0[0]*rr[0] + sp->S_y0[1]*rr[1] + sp->S_y0[2]*rr[2];
            S_z1 = sp->S_z0[0]*rr[0] + sp->S_z0[1]*rr[1] + sp->S_z0[2]*rr[2];
            // Normalizing the vectors E and S:
            double E = sqrt(E_x1*E_x1 + E_y1*E_y1 + E_z1*E_z1);
            E_x1= E_x1 / E;
            E_y1= E_y1 / E;
            E_z1= E_z1 / E;
            double S = sqrt(S_x1*S_x1 + S_y1*S_y1 + S_z1*S_z1);
            S_x1= S_x1 / S;
            S_y1= S_y1 / S;
            S_z1= S_z1 / S;
            #else
            // Using Sun and Earth coordinates interpolated previously on CPU
            #define E_x1 sData[i].E_x
            #define E_y1 sData[i].E_y
            #define E_z1 sData[i].E_z
            #define S_x1 sData[i].S_x
            #define S_y1 sData[i].S_y
            #define S_z1 sData[i].S_z
            #endif

            // Earth vector in the new (b,c,a) basis
            // Switching from Muinonen coords (abc) to Samarasinha coords (bca)
            Ep_b = b_x*E_x1 + b_y*E_y1 + b_z*E_z1;
            Ep_c = c_x*E_x1 + c_y*E_y1 + c_z*E_z1;
            Ep_a = a_x*E_x1 + a_y*E_y1 + a_z*E_z1;
            
            // Sun vector in the new (b,c,a) basis
            // Switching from Muinonen coords (abc) to Samarasinha coords (bca)
            Sp_b = b_x*S_x1 + b_y*S_y1 + b_z*S_z1;
            Sp_c = c_x*S_x1 + c_y*S_y1 + c_z*S_z1;
            Sp_a = a_x*S_x1 + a_y*S_y1 + a_z*S_z1;
            
            #ifdef BC
            double b = P_b;
            double c = P_c;
            #else
            double b = P_b_tumb;
            double c = P_c_tumb;
            #endif        
            
            #ifdef ANIMATE
            if (i>=i1_rgb && i<i2_rgb)
                // Module to compute the image of the asteroid using all threads in this block
                compute_rgb(d_rgb, b,  c, 
                            E_x1,  E_y1,  E_z1,
                            Ep_b,  Ep_c,  Ep_a,
                            Sp_b,  Sp_c,  Sp_a,
                            b_x,  b_y,  b_z,
                            c_x,  c_y,  c_z,
                            a_x,  a_y,  a_z,
                            i, i1_rgb);
            continue;
            #endif
            
            
            // Now that we converted the Earth and Sun vectors to the internal asteroidal basis (a,b,c),
            // we can apply the formalism of Muinonen & Lumme, 2015 to calculate the brightness of the asteroid.
            
            double Vmod;
            
            #if defined(BW_BALL)
            /* The simplest non-geometric brightness model - "black and white ball".
             * The "a" axis end hemisphere is dark (albedo kappa<1), the oppostire hemisphere is bright (albedo=1).
             * Assuming the phase angle = 0 (sun is behind the observer) for simplicity.
             */
            
            // Zeta is the angle between the rotated axis "a1" and the direction to the observer (Ep)
            double cos_zeta = Ep_a;
            // Relative bw ball brightness (1 when only the bright hemisphere is visible; kappa when only the dark one):
//            Vmod = -2.5*log10(0.5*(P_kappa*(1+cos_alpha) + (1-cos_alpha)));
            // Bug fix on May 29, 2019:
            Vmod = -2.5*log10(0.5*(P_kappa*(1+cos_zeta) + 0.5*(1-cos_zeta)));
            
            #elif defined(RECT)
            /* Simplified (phase is fixed at 0) rectangular prism brightness model.
             * Here a, b, c correspond to half-lengths of the longest, intermediate, and shortest sides.
             * Phase is fixed at 0, so all we are computing is the surface area of the prism projected on the plane of sky.
             * Except for special cases, projected rectangular prism will consist of the three projected sides
             * (which are parallelograms in projection): (a,b), (b,c), and (a,c).
             * */
            // Scaling the unit vectors b,c by the lengths provided by b and c (a=1):
            b_x = b * b_x;
            b_y = b * b_y;
            b_z = b * b_z;
            
            c_x = c * c_x;
            c_y = c * c_y;
            c_z = c * c_z;
            
            // Projected axes (onto the plane of sky):
            double ap_x = a_y*Ep_a - a_z*Ep_c;
            double ap_y = a_z*Ep_b - a_x*Ep_a;
            double ap_z = a_x*Ep_c - a_y*Ep_b;
            
            double bp_x = b_y*Ep_a - b_z*Ep_c;
            double bp_y = b_z*Ep_b - b_x*Ep_a;
            double bp_z = b_x*Ep_c - b_y*Ep_b;

            double cp_x = c_y*Ep_a - c_z*Ep_c;
            double cp_y = c_z*Ep_b - c_x*Ep_a;
            double cp_z = c_x*Ep_c - c_y*Ep_b;

            // Vector products to be used in surface area of the projected prism computation:
            double ab_x = ap_y*bp_z - ap_z*bp_y;
            double ab_y = ap_z*bp_x - ap_x*bp_z;
            double ab_z = ap_x*bp_y - ap_y*bp_x;
            
            double ac_x = ap_y*cp_z - ap_z*cp_y;
            double ac_y = ap_z*cp_x - ap_x*cp_z;
            double ac_z = ap_x*cp_y - ap_y*cp_x;
            
            double bc_x = bp_y*cp_z - bp_z*cp_y;
            double bc_y = bp_z*cp_x - bp_x*cp_z;
            double bc_z = bp_x*cp_y - bp_y*cp_x;    
            
            // Brightness is assumed to be proportional to the surface area of the projected rectangular prism (so no phase effects):
            double ab = ab_x*ab_x+ab_y*ab_y+ab_z*ab_z;
            double ac = ac_x*ac_x+ac_y*ac_y+ac_z*ac_z;
            double bc = bc_x*bc_x+bc_y*bc_y+bc_z*bc_z;
            if (ab < 0.0)
                ab = 0.0;
            if (bc < 0.0)
                bc = 0.0;
            if (ac < 0.0)
                ac = 0.0;
            Vmod = -2.5*log10(4*(sqrt(ab) + sqrt(bc) + sqrt(ac)));
            
            #else
            /* The defaul brightness model (triaxial ellipsoid, constant albedo), from Muinonen & Lumme, 2015
             */
            double cos_alpha_p, sin_alpha_p, scalar_Sun, scalar_Earth, scalar;
            double cos_lambda_p, sin_lambda_p, alpha_p, lambda_p;
            
            // The two scalars from eq.(12) of Muinonen & Lumme, 2015; assuming a=1
            // Switching from Muinonen coords (abc) to Samarasinha coords (bca)
            scalar_Sun   = sqrt(Sp_b*Sp_b/(b*b) + Sp_c*Sp_c/(c*c) + Sp_a*Sp_a);
            scalar_Earth = sqrt(Ep_b*Ep_b/(b*b) + Ep_c*Ep_c/(c*c) + Ep_a*Ep_a);
            
            // From eq.(13):
            // Switching from Muinonen coords (abc) to Samarasinha coords (bca)
            cos_alpha_p = (Sp_b*Ep_b/(b*b) + Sp_c*Ep_c/(c*c) + Sp_a*Ep_a) / (scalar_Sun * scalar_Earth);
            sin_alpha_p = sqrt(1.0 - cos_alpha_p*cos_alpha_p);
            alpha_p = atan2(sin_alpha_p, cos_alpha_p);
            
            // From eq.(14):
            scalar = sqrt(scalar_Sun*scalar_Sun + scalar_Earth*scalar_Earth + 2*scalar_Sun*scalar_Earth*cos_alpha_p);
            cos_lambda_p = (scalar_Sun + scalar_Earth*cos_alpha_p) / scalar;
            sin_lambda_p = scalar_Earth*sin_alpha_p / scalar;
            lambda_p = atan2(sin_lambda_p, cos_lambda_p);
            
            // Asteroid's model visual brightness, from eq.(10):
            // Simplest case of isotropic single-particle scattering, P(alpha)=1:
            Vmod = -2.5*log10(b*c * scalar_Sun*scalar_Earth/scalar * (cos(lambda_p-alpha_p) + cos_lambda_p +
            sin_lambda_p*sin(lambda_p-alpha_p) * log(1.0 / tan(0.5*lambda_p) / tan(0.5*(alpha_p-lambda_p)))));
            #endif  // if BW_BALL
            
            #ifdef TREND
            // Solar phase angle:
            double alpha = acos(Sp_b*Ep_b + Sp_c*Ep_c + Sp_a*Ep_a);
            // De-trending the brightness curve:
            Vmod = Vmod - P_A*alpha;
            #endif        
            
            if (Nplot > 0)
            {
                #ifndef MINIMA_TEST                
                d_Vmod[i] = Vmod + delta_V[0]; //???
                #endif                
            }
            else
            {
                // Filter:
                int m = sData[i].Filter;
                // Difference between the observational and model magnitudes:
                double y = sData[i].V - Vmod;  
//        printf("%f %f\n",sData[i].V ,Vmod);
                sum_y2[m] = sum_y2[m] + y*y*sData[i].w;
                sum_y[m] = sum_y[m] + y*sData[i].w;
                sum_w[m] = sum_w[m] + sData[i].w;
            }
            #ifdef NUDGE
            // Determining if the previous time point was a local minimum
            if (i < i1 + 2)
            {
                t_old[i-i1] = sData[i].MJD;
                V_old[i-i1] = Vmod;
            }
            else
            {
                if (V_old[1]>V_old[0] && V_old[1]>=Vmod) 
                    // We just found a brightness minimum (V maximum), between i-2 ... i
                {
                    bool local=0;
                    for (int ii=0; ii<sp->N_obs; ii++)
                        // If the model minimum at t_old[1] is within DT_MAX2 days from any observed minimum in sp structure, we mark it as local.
                        // It can now contribute to the merit function calculations later in the kernel.
                        if (fabs(t_old[1]-sp->t_obs[ii]) < DT_MAX2)
                            local = 1;
                        if (local)
                            // Only memorising model minima in the vicinity of observed minima (within DT_MAX2 days) - along the time axis:
                        {
                            M++;  // Counter of model minima in the vicinity of observed minima in t dimension
                            if (M > M_MAX)
                            {
                                // Too many local minima - a fail:
                                return 1e30;
                            }
                            // Using parabolic approximatioin to find the precise location of the local model minimum in the [i-2 ... i] interval
                            //Fitting a parabola to the three last points:
                            double a = ((Vmod-V_old[1])/(sData[i].MJD-t_old[1]) - (V_old[1]-V_old[0])/(t_old[1]-t_old[0])) / (sData[i].MJD-t_old[0]);
                            double b = (V_old[1]-V_old[0])/(t_old[1]-t_old[0]) - a*(t_old[1]+t_old[0]);
                            double c = V_old[1] - a*t_old[1]*t_old[1] - b*t_old[1];
                            // Maximum point for the parabola:
                            t_mod[M-1] = -b/2.0/a;
                            V_mod[M-1] = a*t_mod[M-1]*t_mod[M-1] + b*t_mod[M-1] + c;
                        }
                }
                
                // Shifting the values:
                t_old[0] = t_old[1];
                V_old[0] = V_old[1];
                t_old[1] = sData[i].MJD;
                V_old[1] = Vmod;           
            }
            #endif
            
            #ifdef MINIMA_TEST
            if (Nplot > 0)  
            {
                // Determining if the previous time point was a local minimum
                if (i < i1 + 2)
                {
                    t_old[i-i1] = sData[i].MJD;
                    V_old[i-i1] = Vmod;
                }
                else
                {
                    if (V_old[1]>V_old[0] && V_old[1]>=Vmod) 
                        // We just found a brightness minimum (V maximum), between i-2 ... i
                    {                                                
                        //!!! Assumes that the input data always starts from the same point (all_new.dat file):
                        double t = 58051.044624 + t_old[1];
                        
                        // Only accepting minima within the time intervals (well) covered by observations:
                        if (t>=58051.044624 && t<=58051.117754 ||
                            t>=58051.977665 && t<=58052.185066 ||
                            t>=58053.078873 && t<=58053.528586 ||
                            t>=58054.093274 && t<=58054.514202 ||
                            t>=58055.234145 && t<=58055.354832 ||
                            t>=58056.181290 && t<=58056.278901)
                        {
                            N_minima++;
                            if (N_minima > MAX_MINIMA)
                                return -1;
                            Vmin[N_minima-1] = V_old[1] + delta_V[0];
                        }
                    }
                // Shifting the values:
                t_old[0] = t_old[1];
                V_old[0] = V_old[1];
                t_old[1] = sData[i].MJD;
                V_old[1] = Vmod;           
                }
            }
            #endif
            
            #ifdef MIN_DV
            if (sData[i].MJD > DV_MARGIN && sData[i].MJD < sData[N_data-1].MJD-DV_MARGIN)
                if (Vmod > Vmax)
                    Vmax = Vmod;
                if (Vmod < Vmin)
                    Vmin = Vmod;
                #endif
                
        } // data points loop
        
        
    } // for (iseg) loop
    
    
    
    #ifdef MINIMA_TEST
    if (Nplot > 0)
    {
        if (N_minima == 0)
            return 0.0;
        float Vbest[7];
        // Finding the 7 deepest minima (largest Vmod maxima)
        int N_min = 7;
        if (N_minima < N_min)
            N_min = N_minima;
        for (int j=0; j<N_min; j++)
        {
            float Vmax = -1e30;
            int kmax = -1;
            for (int k=0; k<N_minima; k++)
            {
                if (Vmin[k] > Vmax)
                {
                    Vmax = Vmin[k];
                    kmax = k;
                }
            }  // k loop
            if (kmax == -1)
                return -1;
            Vbest[j] = Vmax;  // Memorizing the minimum
            Vmin[kmax] = -2e30;  // Erasing the minimum we found, so we can search for the next deepest minimum
        }  // j loop
        
        // Computing the score: number of model minima which are deeper than the same ranking deepest observed minima:
        // Full range: from 0 (worst) to 7 (best).
        int score = 0;
        if (N_minima == 0)
            return (CHI_FLOAT)score;
        if (Vbest[0]>=25.715)  // Feature D
            score++;
        if (N_minima == 1)
            return (CHI_FLOAT)score;
        if (Vbest[1]>=25.254) // Feature E
            score++;
        if (N_minima == 2)
            return (CHI_FLOAT)score;
        if (Vbest[2]>=25.234) // Feature C
            score++;
        if (N_minima == 3)
            return (CHI_FLOAT)score;
        if (Vbest[3]>=25.212) // Feature A
            score++;
        if (N_minima == 4)
            return (CHI_FLOAT)score;
        if (Vbest[4]>=24.940) // Feature B
            score++;
        if (N_minima == 5)
            return (CHI_FLOAT)score;
        if (Vbest[5]>=24.846) // Feature F
            score++;
        if (N_minima == 6)
            return (CHI_FLOAT)score;
        if (Vbest[6]>=24.834) // Feature L
            score++;

        // Returning score (instead of the usual chi2):
        return (CHI_FLOAT)score;
    }  // if Nplot>0
    #endif // MINIMA_TEST
    
    
    
    if (Nplot > 0)
        return 0.0;
    
    // Computing chi^2
    CHI_FLOAT chi2m;
    chi2a=0.0;    
    #ifdef RMSD
    CHI_FLOAT SUM_w = 0.0;
    #endif
    for (m=0; m<N_filters; m++)
    {
        // Chi^2 for the m-th filter:
        chi2m = sum_y2[m] - sum_y[m]*sum_y[m]/sum_w[m];
        chi2a = chi2a + chi2m;
        // Average difference Vdata-Vmod for each filter (used for plotting):
        // In SEGMENT mode, computation is done here, over all the segments, as the model scaling (with its size) is fixed across all the segments
        delta_V[m] = sum_y[m] / sum_w[m];
        #ifdef RMSD
        SUM_w = SUM_w + sum_w[m];
        #endif
    }   
    
    #ifdef RMSD // Computing RMSD:
    chi2a = sqrt(chi2a / SUM_w);
    #else  // Normal case: computing chi^2:
    chi2a = chi2a / (N_data - N_PARAMS - N_filters);
    #endif
    
    #ifdef NUDGE
    // Here we will modify the chi2a value based on how close model minima are to the corresponding observed minima (in 2D - both t and V axes),
    // and will punish if the number of model local minima gets too high.    
    float S_M = 0.0;
    float P_tot = 1.0;
    for (int imod=0; imod < M; imod++)
        // Loop over all detected local model minima
    {
        for (int iobs=0; iobs < sp->N_obs; iobs++)
            // Loop over all the observed minima in sp structure
        {
            float dt = fabs(t_mod[imod]-sp->t_obs[iobs]);
            if (dt < DT_MAX2)
                // Only local model minima are processed
            {
                if (dt > DT_MAX)
                    // dt is between DT_MAX and DT_MAX2; we use this transition area to punish for too many model minima; it doesn't contribute to nudging
                {
                    // x=0 when data minimum enters the remote (DT_MAX2) vicinity of the iobs observed minimum, and becomes 1 when it enters the close (DT_MAX) vicinity:
                    float x = (DT_MAX2 - dt) / (DT_MAX2 - DT_MAX);
                    S_M = S_M + x*x*(-2.0*x+3.0); // Computing the effective number of model minima, using a cubic spline
                }
                else
                    // Inside the DT_MAX area
                {
                    S_M = S_M + 1.0;
                    // !!! Only works properly if N_filters=1 !!!
                    float dV = V_mod[imod] + delta_V[0] - sp->V_obs[iobs];
                    #ifdef V1S
                    // One-sided treatment of dV: if model minimum is below the observed one, keep dV=0 (don't punish). Only punish when dV>0.
                    // This should promote minima which are at least as deep as the observed ones
                    if (dV > 0.0)
                        dV = 0.0;
                    #endif                    
                    // 2D distance of the model minimum from the observed one, with different scales for t and V axes, normalized to DT_MAX and DV_MAX, respectively:
                    float x = sqrt(dt*dt/DT_MAX/DT_MAX + dV*dV/DV_MAX/DV_MAX);
                    if (x < 1.0)
                        // The model minimum is inside the 2D vicinity area near the observed minimum
                    {
                        //                        float P_i = x*x*(-2.0*x+3.0); // Using a cubic spline for a smooth reward function
                        // Using inverted Lorentzian function instead, with the core radius L_RC=0..1 (L_RC2=L_RC^2)
                        // It is not perfect (at x=1 the derivative is not perfectly smooth, but good enogh for small L_RC)
                        float P_i = L_A * x*x/(x*x + L_RC2);
                        // Computing the cumulative reward function based on how close model minima are to observed ones.
                        // 0<P_MIN<1 sets how strong the reward is (the closer to 0, the stronger)
                        P_tot = P_tot * (P_MIN*(1.0 + P_MIN2*P_i));                    
                    }
                }
            }
        }
    }
    P_tot = powf(P_tot, 1.0/sp->N_obs); // Normalizing the reward to the number of observed minima
    // P_tot is the reward factor for how close all observed minima are to model minima. It varies between P_MIN (likely a perfect match) to 1 (no match)
    if (P_tot < P_MIN)
        // This might happen if there is more than one model minimum per observed one; we don't want to encourage that:
        P_tot = P_MIN;
    if (chi2a > CHI2_1)
        P_tot = 1.0;
    else if (chi2a > CHI2_0)
    {
        float x = (CHI2_1 - chi2a) / (CHI2_1 - CHI2_0);
        float beta = x*x*(-2*x+3);  // Using cubic spline for a smooth transition from the chi2a>CHI2_1 mode (P_tot=1) to chi2a<CHI2_0 mode (full P_tot)
        P_tot = powf(P_tot, beta);
    }
    
    float P_M;
    if (S_M < M_MAX2)
        P_M = 1.0;
    else if (S_M < M_MAX)
    {
        float x = (S_M-M_MAX2) / (M_MAX-M_MAX2);
        // ??? I could introduce a const parameter to regulate the strength of the punishment
        P_M = 1.0 + 3*x*x*(-2*x+3); // Using cubic spline to smoothen the punishment function for too many model minima; varies from 1 (no punishment) to 4 (maximum punishment)
    }
    else
        P_M = 4.0; // Will need to change if the strength of punishment is an ajustable parameter
        
        // !!! Need to fix edge effects!!!
        // Also, should only reward when chi2 is good enough
        // Applying the reward and the punishment to chi2:
        chi2a = chi2a * P_tot * P_M;
    #endif
    
    #ifdef MIN_DV
    double x = (Vmax-Vmin-DV_MIN1)/(DV_MIN2-DV_MIN1);
    double P;
    if (x < 0.0)
        P = 1.0;
    else if (x < 1.0)
    {
        // Merit function multiplier: P=1 when x->0 (dV->0), P=PV_MIN<1 when x>=1 (dV>=DV_MIN):
        P = (1.0 - x*x*(-2*x+3))*(1.0-PV_MIN) + PV_MIN;
    }
    else
        P = PV_MIN;
    chi2a = chi2a * P;
    #endif
        
    return chi2a;
}           


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__device__ void params2x(CHI_FLOAT *x, double *params, CHI_FLOAT sLimits[][N_TYPES], int sProperty[][N_COLUMNS], int sTypes[][N_SEG], volatile struct x2_struct *s_x2_params)
// Converting from dimensional params structure to dimensionless array x. Used for plotting. 
// P_PHI, P_PSI, P_BOTH, and RANDOM_BC are not supported ???
// It is assumed that in the params vector there is the following order: ..., c_tumb, ..., b_tumb, ..., Es, ..., psi_0, ...
// Also, c,b should follow c_tumb.
{    
    int LAM, i_Es;
    
    // Explicitly assuming here that c_tumb and b_tumb are multi-segment (do not change between segments)
    double b_tumb = params[sTypes[T_b_tumb][0]];
    double c_tumb = params[sTypes[T_c_tumb][0]];
    double Is = (1.0+b_tumb*b_tumb) / (b_tumb*b_tumb + c_tumb*c_tumb);
    double Ii = (1.0+c_tumb*c_tumb) / (b_tumb*b_tumb + c_tumb*c_tumb);
    
    for (int i=0; i<N_PARAMS; i++)
    {
        int param_type = sProperty[i][P_type];
        if (sProperty[i][P_frozen] != 0)
        {
            // For frozen (P_frozen=1) or fully relaxed (P_frozen=-1) parameters, arbitrarily setting x to zero:
            x[i] = 0;
            continue;
        }
        
        if (sProperty[i][P_independent] == 1)
        {
            double par = params[i];
            if (sProperty[i][P_periodic] == 1)
            {
                x[i] = par / (2*PI);
                x[i] = x[i] - floor(x[i]);  // Converting to the canonical interval (0..1)
            }
            else
            {
                #ifdef BC
                if (param_type == T_c)
                {
                    // Parameter "c" has the same limits as "c_tumb", and log distribution:
                    x[i] = (log(params[i]) - sLimits[0][T_c_tumb]) / (sLimits[1][T_c_tumb] - sLimits[0][T_c_tumb]);        
                    continue;
                }
                #endif
                if (param_type == T_c_tumb)
                    par = log(par);
                #ifdef BW_BALL
                if (param_type == T_kappa)
                    par = log(par);
                #endif
                x[i] = (par - sLimits[0][param_type]) / (sLimits[1][param_type] - sLimits[0][param_type]);        
            }
        }
        else
        {
            
            if (param_type == T_b_tumb)
            {
                double par = log(b_tumb)/log(c_tumb);
                x[i] = (par - sLimits[0][param_type]) / (sLimits[1][param_type] - sLimits[0][param_type]);        
            } 
            
            else if (param_type == T_Es)
            {
                i_Es = i;
                LAM = params[i] > 1.0/Ii;
                if (LAM)
                    // LAM: Es>1.0/Ii; x=[0.5,1]
                    x[i] = 0.5*((params[i]-1.0/Ii) / (1.0-1.0/Ii) + 1.0);
                else
                    // SAM: Es<1.0/Ii; x=[0,0.5]
                    x[i] = 0.5*(params[i]-1.0/Is) / (1.0/Ii - 1.0/Is);
            }
            
            else if (param_type == T_psi_0)
            {
                double psi_min, psi_max;
                if (LAM)
                {
                    psi_min = 0.0;
                    psi_max = 2.0*PI;
                }
                else
                {
                    psi_max = atan(sqrt(Ii*(Is-1.0/params[i_Es])/Is/(1.0/params[i_Es]-Ii)));
                    psi_min = -psi_max;
                }
                x[i] = (params[i] - psi_min) / (psi_max - psi_min);                
            }
            
            #ifdef BC            
            else if (param_type == T_b)
            {
                double par = log(params[i]) / log(params[sTypes[T_c][sProperty[i][P_iseg]]]);
                x[i] = (par - sLimits[0][param_type]) / (sLimits[1][param_type] - sLimits[0][param_type]);        
            }
            #endif            
        }
    }
    
    return;
}    

 
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__device__ int x2params(CHI_FLOAT *x, double *params, CHI_FLOAT sLimits[][N_TYPES], volatile struct x2_struct *s_x2_params, int sProperty[][N_COLUMNS], int sTypes[][N_SEG])
// Conversion from dimensionless x[] parameters to the physical ones params[]
// RANDOM_BC is not supported yet
{    
    // LAM (=1) or SAM (=0):
    int LAM = 0;
    
    // Checking if we went beyond the hard limits:
    for (int i=0; i<N_PARAMS; i++)
    {
        if (sProperty[i][P_type] == T_Es)
            LAM = x[i]>=0.5;

//???
/* s       
        #if defined(P_PHI) || defined(P_PSI) || defined(P_BOTH)
        // In random search mode (reopt=0) we do not enforce L here - it will be done later
        if (s_x2_params->reopt==0  && sProperty[i][P_type] == T_L)
              continue;
        #endif
*/        
        if (x[i]<0.0 && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_LEFT  || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM) ||
            x[i]>1.0 && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_RIGHT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
        {
            // We stepped outside hard limits - fail:
            return 1;
        }        
    }
    
    double log_c_tumb, log_b_tumb, Is, Ii, psi_min, psi_max;
    #ifdef BC
    double log_c;
    #endif
    
    // The x -> params conversion
    for (int i=0; i<N_PARAMS; i++)
    {
        int param_type = sProperty[i][P_type];
        int iseg = sProperty[i][P_iseg];
        
        // First we start with special cases parameters:
        
        if (param_type == T_b_tumb)
        {
            log_b_tumb = log_c_tumb * (x[i]*(sLimits[1][param_type]-sLimits[0][param_type]) + sLimits[0][param_type]);
            params[i] = exp(log_b_tumb);
            double b_tumb = params[i];
            double c_tumb = params[sTypes[T_c_tumb][iseg]];
            Is = (1.0+b_tumb*b_tumb) / (b_tumb*b_tumb + c_tumb*c_tumb);
            Ii = (1.0+c_tumb*c_tumb) / (b_tumb*b_tumb + c_tumb*c_tumb);
        }
        
        else if (param_type == T_Es)
        {
            if (LAM)
                // LAM: Es>1.0/Ii
            {
                params[i] = 2.0*(x[i]-0.5)*(1.0-1.0/Ii)+1.0/Ii;
                psi_min = 0.0;
                psi_max = 2.0*PI;
            }
            else                
                // SAM: Es<1.0/Ii
            {
                params[i] = 2.0*x[i]*(1.0/Ii-1.0/Is)+1.0/Is;
                psi_max = atan(sqrt(Ii*(Is-1.0/params[i])/Is/(1.0/params[i]-Ii)));
                psi_min = -psi_max;
            }
        }
        
        else if (param_type == T_psi_0)
        {
            params[i] = x[i]*(psi_max-psi_min) + psi_min;
        }
        
        #ifdef BC
        else if (param_type == T_b)
        {
            double log_b = log_c * (x[i]*(sLimits[1][param_type]-sLimits[0][param_type]) + sLimits[0][param_type]);
            if (fabs(log_b-log_b_tumb) > BC_DEV_MAX)
                return 1;
            params[i] = exp(log_b);
        }
        #endif
        
        // Then we continue with general classes of parameters
        
        // All periodic parameters (excluding the special case of T_psi_0 && LAM - this is handled separately, above, via psi_min and psi_max)
        else if (sProperty[i][P_periodic] == PERIODIC)
        {
            params[i] = x[i] * 2.0*PI;
        }
        
        // Independent non-periodic parameters; all dependent non-periodic parameters have to be handled separately, as special cases
        else if (sProperty[i][P_periodic] != PERIODIC && sProperty[i][P_independent] == 1)
        {
            #if defined(P_PSI) || defined(P_PHI) || defined(P_BOTH)                       
            // In P_* modes, L parameter is not computed here, but a few lines below
            // (Unless it's reoptimization)
            if (param_type != T_L || s_x2_params->reopt)
            #endif
                // The default way to compute independent params[i] from x[i]:
                params[i] = x[i] * (sLimits[1][param_type]-sLimits[0][param_type]) + sLimits[0][param_type];

            if (param_type == T_c_tumb)
            {
                log_c_tumb = params[i];
                params[i] = exp(log_c_tumb);
            }
            #ifdef BC
            else if (param_type == T_c)
            {
                log_c = params[i];
                if (fabs(log_c-log_c_tumb) > BC_DEV_MAX)
                    return 1;
                params[i] = exp(log_c);
            }
            #endif            
            #ifdef BW_BALL
            else if (param_type == T_kappa)
            {
                params[i] = exp(params[i]);
            }
            #endif            
            else if (param_type == T_L && !s_x2_params->reopt)
            {
                #if defined(P_PSI) || defined(P_PHI) || defined(P_BOTH)            
                // In P_* modes, T_L parameter has a different meaning
                #ifdef P_PHI
                /* Using the empirical fact that for a wide range of c_tumb, b_tumb, Es, L parameters, Pphi = S0*2*pi/Es/L (SAM)
                 * and S1*2*pi*Ii/L (LAM) with ~20% accuracy; S0=[1,S_LAM0], S1=[1,S_LAM1]. 
                 * This allows an easy constraint on L if the range of Pphi is given. 
                 * When generating L, we use both the S0/1 ranges, and the given Phi1...Pphi2 range.
                 */
                if (LAM)
                    params[i] = (x[i] * (S_LAM0*s_x2_params->Pphi2-s_x2_params->Pphi) + s_x2_params->Pphi) / params[sTypes[T_Es][0]];
                else
                    params[i] = (x[i] * (S_LAM1*s_x2_params->Pphi2-s_x2_params->Pphi) + s_x2_params->Pphi) * Ii;                
                #endif
                
                #if defined(P_PSI) || defined(P_BOTH)
                // 1/P_psi is computed and stored in params(T_L):
                params[i] = x[i] * (s_x2_params->Ppsi2-s_x2_params->Ppsi1) + s_x2_params->Ppsi1;
                // In P_PSI/combined modes the actual optimization parameter is Ppsi which is stored in params.L, and L is derived from Ppsi and Is, Ii, Es
                double Einv = 1.0/params[sTypes[T_Es][iseg]];
                double k2;
                if (LAM)
                    k2=(Is-Ii)*(Einv-1.0)/((Ii-1.0)*(Is-Einv));
                else
                    k2=(Ii-1.0)*(Is-Einv)/((Is-Ii)*(Einv-1.0));
                // Computing the complete eliptic integral K(k2) using the efficient AGM (arithemtic-geometric mean) method
                // With double precision, converges to better than 1e-10 after 5 loops, for k2=0...9.999998e-01
                double a = 1.0;   double g = sqrt(1.0-k2);
                double a1, g1;
                for (int ii=0; ii<5; ii++)
                {
                    a1 = 0.5 * (a+g);
                    g1 = sqrt(a*g);
                    a = a1;  g = g1;
                }
                // Now that we know K(k2)=PI/(a+g), we can derive L from Ppsi:
                // Here the meaning of params.L changes: from 1/Ppsi to L
                if (LAM)
                    params[i] = 4.0*params[i]* PI/(a+g) *sqrt(Ii*Is/(params[sTypes[T_Es][iseg]]*(Ii-1.0)*(Is-Einv)));
                else
                    params[i] = 4.0*params[i]* PI/(a+g) *sqrt(Ii*Is/(params[sTypes[T_Es][iseg]]*(Is-Ii)*(Einv-1.0)));
                #ifdef P_BOTH    
                // In the P_BOTH mode we have to use a rejection method to prune out models with the wrong combination of Ppsi and Pphi
                double S, S2;
                // Here dPphi = P_phi / (2*PI)
                if (LAM == 0)
                {
                    S  = params[i] * s_x2_params->Pphi  * params[sTypes[T_Es][iseg]];
                    S2 = params[i] * s_x2_params->Pphi2 * params[sTypes[T_Es][iseg]];
                    if (S2 < 1.0 || S > S_LAM0)
                        // Out of the emprirical boundaries for P_phi constraining:
                        return 2;
                }
                else
                {
                    S  = params[i] * s_x2_params->Pphi   / Ii;
                    S2 = params[i] * s_x2_params->Pphi2  / Ii;
                    if (S2 < 1.0 || S > S_LAM1)
                        // Out of the emprirical boundaries for P_phi constraining:
                        return 2;
                }
                #endif  // P_PSI || P_BOTH
                #endif  // P_BOTH      
                
                #endif  // if any P_* mode                
            } // if (param_type == T_L)
            
        }  // if param_type
        
    }  // for (i)
    
    
    
    return 0;
}



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef ANIMATE
__global__ void chi2_gpu (struct obs_data *dData, int N_data, int N_filters, int reopt, int Nstages,
                          curandState* globalState, CHI_FLOAT *d_f, double* d_params, double* d_dV)
// CUDA kernel computing chi^2 on GPU
{        
    #ifndef NO_SDATA
    __shared__ struct obs_data sData[MAX_DATA];
    #endif
    __shared__ CHI_FLOAT sLimits[2][N_TYPES];
    __shared__ volatile CHI_FLOAT s_f[BSIZE];
//    __shared__ volatile int s_thread_id[BSIZE];
    __shared__ int sProperty[N_PARAMS][N_COLUMNS];
    __shared__ int sTypes[N_TYPES][N_SEG];
    __shared__ struct chi2_struct sp;
    __shared__ volatile struct x2_struct s_x2_params;
    __shared__ volatile CHI_FLOAT s_x0[N_PARAMS];
    __shared__ volatile int thread_min;
    __shared__ volatile CHI_FLOAT smin;
    int i, j;
    double params[N_PARAMS];
    CHI_FLOAT delta_V[N_FILTERS];
    int ind[N_PARAMS+1]; // Indexes to the sorted array (point index)
    
    
    // Not efficient, for now:
    if (threadIdx.x == 0)
    {
        #ifndef NO_SDATA
          for (i=0; i<N_data; i++)
              sData[i] = dData[i];
          #ifdef INTERP
          for (i=0; i<3; i++)
          {
              sp.E_x0[i] = dE_x0[i];
              sp.E_y0[i] = dE_y0[i];
              sp.E_z0[i] = dE_z0[i];
              sp.S_x0[i] = dS_x0[i];
              sp.S_y0[i] = dS_y0[i];
              sp.S_z0[i] = dS_z0[i];
              sp.MJD0[i] = dMJD0[i];
          }
          #endif
        #endif        
        #ifdef NUDGE
        // Copying the data on the observed minima from device to shared memory:
        sp.N_obs = d_chi2_params.N_obs;
        for (i=0; i<sp.N_obs; i++)
        {
            sp.t_obs[i] = d_chi2_params.t_obs[i];
            sp.V_obs[i] = d_chi2_params.V_obs[i];
        }
        #endif
        for (i=0; i<N_TYPES; i++)
        {
            sLimits[0][i] = dLimits[0][i];
            sLimits[1][i] = dLimits[1][i];
            for (int iseg=0; iseg<N_SEG; iseg++)
                sTypes[i][iseg] = dTypes[i][iseg];
        }
        for (i=0; i<N_PARAMS; i++)
            for (j=0; j<N_COLUMNS; j++)
                sProperty[i][j] = dProperty[i][j];
        #ifdef P_PSI            
        s_x2_params.Ppsi1 = d_x2_params.Ppsi1;
        s_x2_params.Ppsi2 = d_x2_params.Ppsi2;
        #endif       
        #ifdef P_BOTH            
        s_x2_params.Pphi =  d_x2_params.Pphi;
        s_x2_params.Pphi2 = d_x2_params.Pphi2;
        #endif       
        #ifdef SEGMENT
        // Copying the starting indexes for data segments to shared memory:
        for (i=0; i<N_SEG; i++)
            sp.start_seg[i] = d_start_seg[i];
        #endif   
        
        s_x2_params.reopt = reopt;
    }
    
    CHI_FLOAT x[N_PARAMS+1][N_PARAMS];  // simplex points (point index, coordinate)
    CHI_FLOAT f[N_PARAMS+1]; // chi2 values for the simplex edges (point index)

    __syncthreads();
    
    if (s_x2_params.reopt)
    {
        // Reading the initial point from device memory
        for (i=0; i<N_PARAMS; i++)
            params[i] = d_params0[i];
        // Converting from physical to dimensionless (0...1 scale) parameters:
        params2x(x[0], params, sLimits, sProperty, sTypes, &s_x2_params);
    }
        
    // Global thread index:
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    // Reading the global states from device memory:
    curandState localState = globalState[id];    

    for (int istage=0; istage<Nstages; istage++)
    {
     
    __syncthreads();
    if (Nstages>1 && istage>0)
    {
        // Reading the best point obtained in the previous stage:
        for (int i=0; i<N_PARAMS; i++)
            x[0][i] = s_x0[i];
    }
    
    //Simplex steps counter:
    int l = 0;
    
    bool failed;
    #ifdef P_BOTH
    //    for (int itry=0; itry<100; itry++)
    while (1)
    {
    #endif
        
        /* Startegy for placing the initial point:
         *   - When reopt=0 (random search), we make sure that the whole simplex will fit in the dimensionless range
         * [0,1] for all parameters, except for PERIODIC ones. That means that the initial point should be within
         * [SMALL+DX_INI ... 1-SMALL-DX_INI], where SMALL is a small number, and DX_INI is the largest allowed initial
         * simplex size for each parameter.
         * 
         *   - For reopt=1 (reoptimization of the existing model point), it's a bit trickier. We allow all soft limits
         * to be crossed, and only for hard limits the same interval is enforced. If the hard limit is on one side
         * only (HARD_LEFT, HARD_RIGHT), only that limit is enforced.
         */
        #define SMALL 1e-8  // Small offset from the hard parameter limits
        int LAM = 0;

        // Setting the initial values of x[0][i] vector
        for (i=0; i<N_PARAMS; i++)
        {
            // Generating random number in [0..1[ interval:
            float r = curand_uniform(&localState);
            
            #ifdef BC
            #ifndef RANDOM_BC
            if (!s_x2_params.reopt)
            {
                // Initial vales of c/b are equal to initial values of c_tumb/b_tumb:
                if (sProperty[i][P_type] == T_c)
                {
                    x[0][i] = x[0][sTypes[T_c_tumb][sProperty[i][P_iseg]]];
                    continue;
                }            
                else if (sProperty[i][P_type] == T_b)
                {
                    x[0][i] = x[0][sTypes[T_b_tumb][sProperty[i][P_iseg]]];
                    continue;
                }            
            }
            #endif  // RANDOM_BC 
            #endif  // BC      
            
            
            if (!s_x2_params.reopt || s_x2_params.reopt && sProperty[i][P_frozen]==-1)
                // Points placed randomly within the full allowed interval
            {
                // The allowed interval is DX_INI+SMALL  ... 1-(DX_INI-SMALL):
                x[0][i] = DX_INI+SMALL + r*(1.0 - 2*(SMALL+DX_INI));
            }
            else
             // During reoptimization, P_frozen=0 parameters start close to the original values, within +-DX_RAND
            {
                CHI_FLOAT xmin = x[0][i] - DX_RAND;
                CHI_FLOAT xmax = x[0][i] + DX_RAND;
                // Enforcing hard limits:
                if (xmin<SMALL && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_LEFT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
                    xmin = SMALL;
                if (xmax>1.0-SMALL && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_RIGHT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
                    xmax = 1.0 - SMALL;
                
                x[0][i] = xmin + r*(xmax-xmin);
            }            
        
        if (sProperty[i][P_type] == T_Es)
            // We need to know LAM to figure out whether psi_0 is periodic (LAM=1) or not (LAM=0):
            LAM = x[0][i]>=0.5;
            
        }
            
        // Simplex initialization (initial values x[j][i] for all j>0)
        // Vertex loop:
        for (j=1; j<N_PARAMS+1; j++)
        {
            // Coordinates (parameters) loop:
            for (i=0; i<N_PARAMS; i++)
            {
                if (i == j-1)
                {
                    float d2x = curand_uniform(&localState);
                    // Initial step size is log-random, in the interval exp(D2X_INI)*DX_INI .. DX_INI:
                    CHI_FLOAT dx = DX_INI * exp(D2X_INI*d2x);
                    // Step can be both negative and positiven (random choice):
                    if (curand_uniform(&localState) < 0.5)
                    {
                        // Changing to negative direction:
                        dx = -dx;
                        if (x[0][i]+dx<SMALL && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_LEFT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
                            // Changing back to positive direction if there is no room on the left:
                            dx = -dx;                        
                    }
                    else
                    {
                        if (x[0][i]+dx>1.0-SMALL && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_RIGHT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
                            // Changing to negative direction if there is no room on the right:
                            dx = -dx;                        
                    }
                    // At this point, dx has a random  but safe sign, so the point will stay within the safe (hard) limits
                    x[j][i] = x[0][i] + dx;
                }
                else
                {
                    x[j][i] = x[0][i];
                }
            }
        }
        #undef SMALL
       
        // Computing the initial function values (chi2):        
        failed = 0;
        for (j=0; j<N_PARAMS+1; j++)
        {
            if (x2params(x[j], params, sLimits, &s_x2_params, sProperty, sTypes))
            {
                failed = 1;
                break;
            }

            f[j] = chi2one(params, sData, N_data, N_filters, delta_V, 0, &sp, sTypes);    
        }
                
        #ifdef P_BOTH
        if (failed == 0)
            break;
    }  // end of for loop
    #endif
    
    // The main simplex loop
    while (1)
    {
        if (failed == 1)
            break;
        l++;  // Incrementing the global (for the whole lifetime of the thread) simplex steps counter by one
        
        // Sorting the simplex:
        bool ind2[N_PARAMS+1];
        for (j=0; j<N_PARAMS+1; j++)
        {
            ind2[j] = 0;  // Uninitialized flag
        }
        for (j=0; j<N_PARAMS+1; j++)
        {
            CHI_FLOAT fmin = 1e30;
            int jmin = -1;
            for (int j2=0; j2<N_PARAMS+1; j2++)
            {
                if (ind2[j2]==0 && f[j2] <= fmin)
                {
                    fmin = f[j2];
                    jmin = j2;
                }            
            }
            if (jmin < 0)
                // All f[] values are NaN, so exiting the thread
            {
                ind[0] = 0;
                f[ind[0]] = 1e30;
                break;
            }
            ind[j] = jmin;
            ind2[jmin] = 1;
        }    
        
        // Simplex centroid:
        CHI_FLOAT x0[N_PARAMS];
        for (i=0; i<N_PARAMS; i++)
        {
            CHI_FLOAT sum = 0.0;
            for (j=0; j<N_PARAMS+1; j++)
                sum = sum + x[j][i];
            x0[i] = sum / (N_PARAMS+1);
        }           
        
        // Simplex size squared:
        CHI_FLOAT size2 = 0.0;
        for (j=0; j<N_PARAMS+1; j++)
        {
            CHI_FLOAT sum = 0.0;
            for (i=0; i<N_PARAMS; i++)
            {
                CHI_FLOAT dx = x[j][i] - x0[i];
                sum = sum + dx*dx;
            }
            size2 = size2 + sum;
        }
        size2 = size2 / N_PARAMS;  // Computing the std square of the simplex points relative to the centroid point
        
        if (size2 < SIZE2_MIN)
            // We converged
            break;
        if (l > N_STEPS)
            // We ran out of time
            break;
        
        // Reflection
        CHI_FLOAT x_r[N_PARAMS];
        for (i=0; i<N_PARAMS; i++)
        {
            if (sProperty[i][P_frozen] != 1)
                x_r[i] = x0[i] + ALPHA_SIM*(x0[i] - x[ind[N_PARAMS]][i]);
        }
        CHI_FLOAT f_r;
        if (x2params(x_r,params,sLimits, &s_x2_params, sProperty, sTypes))
            f_r = 1e30;
        else
            f_r = chi2one(params, sData, N_data, N_filters, delta_V, 0, &sp, sTypes);
        if (f_r >= f[ind[0]] && f_r < f[ind[N_PARAMS-1]])
        {
            // Replacing the worst point with the reflected point:
            for (i=0; i<N_PARAMS; i++)
            {
                x[ind[N_PARAMS]][i] = x_r[i];
            }
            f[ind[N_PARAMS]] = f_r;
            continue;  // Going to the next simplex step
        }
        
        // Expansion
        if (f_r < f[ind[0]])
        {
            CHI_FLOAT x_e[N_PARAMS];
            for (i=0; i<N_PARAMS; i++)
            {
                if (sProperty[i][P_frozen] != 1)
                    x_e[i] = x0[i] + GAMMA_SIM*(x_r[i] - x0[i]);
            }
            CHI_FLOAT f_e;
            if (x2params(x_e,params,sLimits, &s_x2_params, sProperty, sTypes))
                f_e = 1e30;
            else
                f_e = chi2one(params, sData, N_data, N_filters, delta_V, 0, &sp, sTypes);
            if (f_e < f_r)
            {
                // Replacing the worst point with the expanded point:
                for (i=0; i<N_PARAMS; i++)
                {
                    x[ind[N_PARAMS]][i] = x_e[i];
                }
                f[ind[N_PARAMS]] = f_e;
            }
            else
            {
                // Replacing the worst point with the reflected point:
                for (i=0; i<N_PARAMS; i++)
                {
                    x[ind[N_PARAMS]][i] = x_r[i];
                }
                f[ind[N_PARAMS]] = f_r;
            }
            continue;  // Going to the next simplex step
        }
        
        // Contraction
        // (Here we repurpose x_r and f_r for the contraction stuff)
        for (i=0; i<N_PARAMS; i++)
        {
            if (sProperty[i][P_frozen] != 1)
                x_r[i] = x0[i] + RHO_SIM*(x[ind[N_PARAMS]][i] - x0[i]);
        }
        if (x2params(x_r,params,sLimits, &s_x2_params, sProperty, sTypes))
            f_r = 1e30;
        else
            f_r = chi2one(params, sData, N_data, N_filters, delta_V, 0, &sp, sTypes);
        if (f_r < f[ind[N_PARAMS]])
        {
            // Replacing the worst point with the contracted point:
            for (i=0; i<N_PARAMS; i++)
            {
                x[ind[N_PARAMS]][i] = x_r[i];
            }
            f[ind[N_PARAMS]] = f_r;
            continue;  // Going to the next simplex step
        }
        bool bad = 0;
        
        // If all else fails - shrink
        for (j=1; j<N_PARAMS+1; j++)
        {
            for (i=0; i<N_PARAMS; i++)
            {
                if (sProperty[i][P_frozen] != 1)
                    x[ind[j]][i] = x[ind[0]][i] + SIGMA_SIM*(x[ind[j]][i] - x[ind[0]][i]);
            }           
            if (x2params(x[ind[j]],params,sLimits, &s_x2_params, sProperty, sTypes))
                bad = 1;
            else
                f[ind[j]] = chi2one(params, sData, N_data, N_filters, delta_V, 0, &sp, sTypes);
        }
        // We failed the optimization
        if (bad)
        {
            failed = 1;
            break;
        }
        
    }  // inner while loop
    
    
//    if (failed == 1 || f[ind[0]] < 1e-5)
    if (failed == 1)
        s_f[threadIdx.x] = 1e30;
    else
        s_f[threadIdx.x] = f[ind[0]];
//    s_thread_id[threadIdx.x] = threadIdx.x;
    
    __syncthreads();
    //!!! AT this point not all warps initialized s_f and s_thread_id! It looks like __syncthreads doesn't work!
    
    /*
    // Binary reduction:
    int nTotalThreads = blockDim.x;
    while(nTotalThreads > 1)
    {
        int halfPoint = nTotalThreads / 2; // Number of active threads
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint; // the second element index
            CHI_FLOAT temp = s_f[thread2];
            if (temp < s_f[threadIdx.x])
            {
                s_f[threadIdx.x] = temp;
                s_thread_id[threadIdx.x] = s_thread_id[thread2];
            }
        }
        __syncthreads();
        nTotalThreads = halfPoint; // Reducing the binary tree size by two
    }
    // At this point, the smallest chi2 in the block is in s_f[0]
    */
    
    // Serial reduction:
    if (threadIdx.x == 0)
    {
        thread_min = 0;
        smin = HUGE;
        for (int j=0; j<blockDim.x; j++)
        {
            if (s_f[j] < smin)
            {
                smin = s_f[j];
                thread_min = j;
            }
        }
    }
    __syncthreads();
    
    if (threadIdx.x == thread_min && Nstages>1 && istage<Nstages-1)
        // When Nstages>1, for each block of threads the best point is used to run reoptimization
    {
        #if defined(P_PSI) || defined(P_PHI) || defined(P_BOTH)            
        // Only in one of P_* modes, and after the first stage (when reopt changes from 0 to 1),
        // we need to do the x->params->x transformation, to switch the meaning of x for L parameter
        if (s_x2_params.reopt == 0)
        {
            // Converting to physical parameters (L now is physical angular momentum):
            x2params(x[ind[0]], params, sLimits, &s_x2_params, sProperty, sTypes);
//            s_x2_params.reopt = 1;
            // Converting back to dimensionless x[] values. Now x(L) has the proper value:
            params2x(x[ind[0]], params, sLimits, sProperty, sTypes, &s_x2_params);
        }
        #endif
        // Memorizing the best point in this block in a shared memory vector:
        for (int i=0; i<N_PARAMS; i++)
            s_x0[i] = x[ind[0]][i];
        // Enabling reoptimization:
        s_x2_params.reopt = 1;
    }
    
    } // istage loop

    //    if (threadIdx.x == s_thread_id[0] && s_f[0] < d_f[blockIdx.x])
    if (threadIdx.x == thread_min && smin < d_f[blockIdx.x])
        // Keeping the current best result if it's better than the previous kernel result for the same blockID
    {
        // Copying the found minimum to device memory:
//        d_f[blockIdx.x] = s_f[0];
        d_f[blockIdx.x] = smin;
        x2params(x[ind[0]],params,sLimits, &s_x2_params, sProperty, sTypes);
        for (int i=0; i<N_PARAMS; i++)
        {
            d_params[blockIdx.x*N_PARAMS + i] = params[i];
            for (int m=0; m<N_filters; m++)
                d_dV[blockIdx.x*N_FILTERS + m] = delta_V[m];
        }
    }
    
    

    
    // Writing the global states to device memory:
    globalState[id] = localState;
    
    return;        
    
}
#endif // not ANIMATE
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// For debugging:
#ifdef DEBUG2
__global__ void debug_kernel(struct parameters_struct params, struct obs_data *dData, int N_data, int N_filters)
{
    __shared__ struct obs_data sData[MAX_DATA];
    int i;
    CHI_FLOAT f;
    
    // Not efficient, for now:
    if (threadIdx.x == 0)
    {
        for (i=0; i<N_data; i++)
            sData[i] = dData[i];
    }
    
    // !!! Will not work in NUDGE mode - NULL
    f = chi2one(params, sData, N_data, N_filters, delta_V, 0, NULL);
    
    return;
    
}

#endif


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__global__ void chi2_plot (struct obs_data *dData, int N_data, int N_filters,
                           struct obs_data *dPlot, int Nplot, double * d_dlsq2, 
#ifdef ANIMATE
                           unsigned char * d_rgb,
#endif
                           float dx_rand)
// CUDA kernel to compute plot data from input params structure
{     
    __shared__ double sd2_min[BSIZE];
    CHI_FLOAT delta_V[N_FILTERS];
    __shared__ struct chi2_struct sp;
    __shared__ CHI_FLOAT sLimits[2][N_TYPES];
    __shared__ int sProperty[N_PARAMS][N_COLUMNS];
    __shared__ int sTypes[N_TYPES][N_SEG];
    __shared__ volatile struct x2_struct s_x2_params;
    #if defined(SPHERICAL_K) && defined(TORQUE) && defined(PROFILES)
    __shared__ volatile double phi0, K0, theta0;
    int iseg=0;
    #endif
    #ifdef ANIMATE
    __shared__ double params[N_PARAMS];
    #else
    double params[N_PARAMS];
    #endif
    
    // Global thread index for points:
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    //    if (id == 0 && blockIdx.y == 0)    
    if (threadIdx.x == 0)
    {

        for (int i=0; i<N_TYPES; i++)
        {
            sLimits[0][i] = dLimits[0][i];
            sLimits[1][i] = dLimits[1][i];
            for (int iseg=0; iseg<N_SEG; iseg++)
                sTypes[i][iseg] = dTypes[i][iseg];
        }
        for (int i=0; i<N_PARAMS; i++)
            for (int j=0; j<N_COLUMNS; j++)
                sProperty[i][j] = dProperty[i][j];
        
        s_x2_params.reopt = 1;
        
        // Reading the initial point from device memory
        for (int i=0; i<N_PARAMS; i++)
            params[i] = d_params0[i];
        
        #ifdef INTERP
        for (int i=0; i<3; i++)
          {
              sp.E_x0[i] = dE_x0[i];
              sp.E_y0[i] = dE_y0[i];
              sp.E_z0[i] = dE_z0[i];
              sp.S_x0[i] = dS_x0[i];
              sp.S_y0[i] = dS_y0[i];
              sp.S_z0[i] = dS_z0[i];
              sp.MJD0[i] = dMJD0[i];
          }
        #endif
        #ifdef NUDGE
        // Copying the data on the observed minima from device to shared memory:
        sp = d_chi2_params;
        #endif
        #ifdef SEGMENT
        // Copying the starting indexes for data segments to shared memory:
        for (int i=0; i<N_SEG; i++)
            sp.start_seg[i] = d_start_seg[i];
        #endif    
        
        #ifndef ANIMATE
        // !!! Will not work in NUDGE mode - NULL
        // Step one: computing constants for each filter (delta_V[]) using chi^2 method, and the chi2 value
        d_chi2_plot = chi2one(params, dData, N_data, N_filters, delta_V, 0,  &sp, sTypes);
        for (int m=0; m<N_filters; m++)
            d_delta_V[m] = delta_V[m];
        
        #ifdef SEGMENT
        // Copying the starting indexes for data segments to shared memory:
        for (int i=0; i<N_SEG; i++)
            sp.start_seg[i] = d_plot_start_seg[i];
        #endif    

        // Step two: computing the Nplots data points using the delta_V values from above:
        chi2one(params, dPlot, Nplot, N_filters, delta_V, Nplot,  &sp, sTypes);

        #if defined(SPHERICAL_K) && defined(TORQUE) && defined(PROFILES)
        // Converting torque vector from Cartesian to spherical coordinates, for confidence interval estimation
        // Shortest axis (c), largest moment of inertia:
        double Is = (1.0 + P_b_tumb*P_b_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
        // Intermediate axis (b), intermediate moment of inertia:
        double Ii = (1.0 + P_c_tumb*P_c_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
        // Torque vector cartesean components:
        double Ki = Ii * P_Ti;
        double Ks = Is * P_Ts;
        double Kl = P_Tl;
        K0 = sqrt(Ki*Ki + Ks*Ks + Kl*Kl);  // r
        theta0 = acos(Kl/K0);  // theta (polar angle; 0..pi)            
        phi0 = atan2(Ks, Ki);  // phi (azimuthal angle; 0..2*pi) for the input model
        #endif
        #endif // not ANIMATE
    }
    
    __syncthreads();
    
    #ifdef ANIMATE
    chi2one(params, dPlot, Nplot, N_filters, delta_V, 0,  &sp,
            d_rgb,
            sTypes);
    return;
    #endif
    
    int blockid = blockIdx.x + gridDim.x*blockIdx.y;
        
    #ifdef PROFILES
    if (id < C_POINTS*BSIZE)
    {
        CHI_FLOAT x[N_PARAMS];
        
        // Reading the initial point from device memory
        for (int i=0; i<N_PARAMS; i++)
            params[i] = d_params0[i];
        // Converting from physical to dimensionless (0...1 scale) parameters:
        params2x(x, params, sLimits, sProperty, sTypes, &s_x2_params);
        #if defined(SPHERICAL_K) && defined(TORQUE)
        // Converting manually to x spherical torque components for the initial model:
        x[sTypes[T_Ti][0]] = K0 / sLimits[1][T_Ti];
        x[sTypes[T_Ts][0]] = theta0 / PI;
        x[sTypes[T_Tl][0]] = phi0 / (2*PI);
        #endif
        
    // Parameter index:
        int iparam = blockIdx.y;            
        // Changes from -DELTA_MAX to +DELTA_MAX:
        // With id+1.0 we ensure that delta=0 corresponds to one of the threads
        double delta = 2.0 * dx_rand * ((id+1.0)/(blockDim.x*gridDim.x) - 0.5);
        
        // Modyfing slightly the corresponding parameter:
        x[iparam] = x[iparam] + delta;

        // Converting back to params:
        x2params(x, params, sLimits, &s_x2_params, sProperty, sTypes);
        #if defined(SPHERICAL_K) && defined(TORQUE)
        // Manual x->params conversion for spherical torque components:
        double K = x[sTypes[T_Ti][0]] * sLimits[1][T_Ti];
        double theta = x[sTypes[T_Ts][0]] * PI;
        double phi = x[sTypes[T_Tl][0]] * 2*PI;
        double Is = (1.0 + P_b_tumb*P_b_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
        double Ii = (1.0 + P_c_tumb*P_c_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
        P_Ti = K * cos(phi) * sin(theta) / Ii;
        P_Ts = K * sin(phi) * sin(theta) / Is;
        P_Tl = K * cos(theta);
        #endif
        // Computing the chi2 for the shifted parameter:
        // !!! Will not work in NUDGE mode - NULL
        d_chi2_lines[iparam][id] = chi2one(params, dData, N_data, N_filters, delta_V, 0, &sp, sTypes);
        
        #if defined(SPHERICAL_K) && defined(TORQUE)
        P_Ti = K;
        P_Ts = theta;
        P_Tl = phi;
        #endif            
        
        d_param_lines[iparam][id] = params[iparam];
    }
    #endif    
    
    return;   
}



//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void setup_kernel ( curandState * state, unsigned long seed, CHI_FLOAT *d_f, int generate_seeds)
{
    // Global thread index:
    unsigned long long id = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (generate_seeds)
        // Generating initial states for all threads in a kernel:
        curand_init ( (unsigned long long)seed, id, 0, &state[id] );
    
    if (threadIdx.x==0)
    {
        d_f[blockIdx.x] = 1e30;    
    }
    
    return;
} 


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ifdef MINIMA_TEST
__global__ void chi2_minima (struct obs_data *dData, int N_data, int N_filters,
                           struct obs_data *dPlot, int Nplot, CHI_FLOAT delta_V1)
// CUDA kernel to run minima test in parallel
// threadId.x: phi_0
// blockIdx.x: theta_M
// blockIdx.y: phi_M
// Doesn't work in NUDGE and SEGMENT
{     
//    CHI_FLOAT delta_V[N_FILTERS];
    __shared__ CHI_FLOAT delta_V[N_FILTERS];
    __shared__ struct chi2_struct sp;
    __shared__ int sTypes[N_TYPES][N_SEG];
    __shared__ volatile int score[N_PHI_0];
    double params[N_PARAMS];
    
    // Global thread index for points:
    int id = threadIdx.x + blockDim.x*blockIdx.x;

    // Reading the initial point from device memory
    for (int i=0; i<N_PARAMS; i++)
        params[i] = d_params0[i];
    
    if (threadIdx.x == 0)
    {
        for (int i=0; i<N_TYPES; i++)
        {
            for (int iseg=0; iseg<N_SEG; iseg++)
                sTypes[i][iseg] = dTypes[i][iseg];
        }
        #ifdef INTERP
        for (int i=0; i<3; i++)
          {
              sp.E_x0[i] = dE_x0[i];
              sp.E_y0[i] = dE_y0[i];
              sp.E_z0[i] = dE_z0[i];
              sp.S_x0[i] = dS_x0[i];
              sp.S_y0[i] = dS_y0[i];
              sp.S_z0[i] = dS_z0[i];
              sp.MJD0[i] = dMJD0[i];
          }
        #endif
        
        // Step one: computing constant delta_V using chi^2 method. 
        // This will be fixed for all models during the second step below. (Ensuring the physical dimensions of asteroid are fixed)
        chi2one(params, dData, N_data, N_filters, delta_V, 0,  &sp, sTypes);
    }
    
    __syncthreads();
    
    // Each thread gets a different value of the phi_0 parameter
    // phi_0 range is 0...2*pi  (evenly distributed)
    params[sTypes[T_phi_0][0]] = 2.0*PI * (double)threadIdx.x / (double)blockDim.x;
    
    // theta_M and phi_M are chosen to correspond to centers of sphere segments of equal area (=> equal probability)
    // gridDim.x should be an even number
    params[sTypes[T_theta_M][0]] = acos((blockIdx.x-(gridDim.x-1)/2.0) / (gridDim.x/2.0));
    params[sTypes[T_phi_M][0]] = blockIdx.y/(double)gridDim.y * 2*PI;
    
    // Computing the Nplots data points using the delta_V value provided at command line, then computing the minima score (from 0 - the worst - to 7 - the best):
    score[threadIdx.x] = (int)chi2one(params, dPlot, Nplot, N_filters, delta_V, Nplot,  &sp, sTypes);
    
    __syncthreads();

    // Averaging scores over all phi_0 values
    float sum = 0.0;
    int N7 = 0;
    if (threadIdx.x == 0)
    {
        for (int i=0; i<blockDim.x; i++)
        {
            // Skipping bad score value (-1) and zeros:
            if (score[i] > 0)
                sum = sum + (float)score[i];
            // Counting cases with the perfect score (7)
            if (score[i] == 7)
                N7++;
        }
        d_Scores[blockIdx.x][blockIdx.y] = sum / blockDim.x;  // Average number of good minima
        d_Prob[blockIdx.x][blockIdx.y] = (float)N7/(float)blockDim.x; // Probability
        atomicAdd(&d_N7all, N7);
    }
    
    return;
}
#endif


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifdef RMSD
__global__ void chi2_gpu_rms (struct obs_data *dData, int N_data, int N_filters, int reopt, int Nstages,
                          curandState* globalState, CHI_FLOAT *d_f, double* d_params, double* d_dV, float dx_rand, float* dpar_min, float* dpar_max)
// CUDA kernel computing the confidence intervals for the input model, using RMSD method (Bartczak & Dudziński 2019)
{        
    #ifndef NO_SDATA
    __shared__ struct obs_data sData[MAX_DATA];
    #endif
    __shared__ CHI_FLOAT sLimits[2][N_TYPES];
    __shared__ volatile CHI_FLOAT s_f[BSIZE];
    __shared__ int sProperty[N_PARAMS][N_COLUMNS];
    __shared__ int sTypes[N_TYPES][N_SEG];
    __shared__ struct chi2_struct sp;
    __shared__ volatile struct x2_struct s_x2_params;
    __shared__ volatile CHI_FLOAT s_x0[N_PARAMS];
    __shared__ volatile int thread_min;
    __shared__ volatile CHI_FLOAT smin;
    __shared__ volatile float s_min[BSIZE], s_max[BSIZE];    
    
    int i, j;
    double params[N_PARAMS];
    CHI_FLOAT delta_V[N_FILTERS];
    int ind[N_PARAMS+1]; // Indexes to the sorted array (point index)
    
    
    if (threadIdx.x == 0)
    {
        #ifndef NO_SDATA
          for (i=0; i<N_data; i++)
              sData[i] = dData[i];
          #ifdef INTERP
          for (i=0; i<3; i++)
          {
              sp.E_x0[i] = dE_x0[i];
              sp.E_y0[i] = dE_y0[i];
              sp.E_z0[i] = dE_z0[i];
              sp.S_x0[i] = dS_x0[i];
              sp.S_y0[i] = dS_y0[i];
              sp.S_z0[i] = dS_z0[i];
              sp.MJD0[i] = dMJD0[i];
          }
          #endif
        #endif        
        for (i=0; i<N_TYPES; i++)
        {
            sLimits[0][i] = dLimits[0][i];
            sLimits[1][i] = dLimits[1][i];
            for (int iseg=0; iseg<N_SEG; iseg++)
                sTypes[i][iseg] = dTypes[i][iseg];
        }
        for (i=0; i<N_PARAMS; i++)
            for (j=0; j<N_COLUMNS; j++)
                sProperty[i][j] = dProperty[i][j];
        
        s_x2_params.reopt = reopt;
    }
    
    __syncthreads();
    
    CHI_FLOAT x[2][N_PARAMS];  // simplex points (point index, coordinate)
    CHI_FLOAT par_min[N_PARAMS];
    CHI_FLOAT par_max[N_PARAMS];
    for (i=0; i<N_PARAMS; i++)
    {
        par_min[i] = 1e30;
        par_max[i] = -1e30;
    }
    int Ntot = 0;
    int Nbad = 0;

    // Reading the initial point from device memory
    for (i=0; i<N_PARAMS; i++)
        params[i] = d_params0[i];
    // Converting from physical to dimensionless (0...1 scale) parameters:
    params2x(x[0], params, sLimits, sProperty, sTypes, &s_x2_params);

    // RMSD value for the input model:
    CHI_FLOAT f0 = chi2one(params, sData, N_data, N_filters, delta_V, 0, &sp, sTypes);
    // + one sigma value for RMSD (all good models will have RMSD smaller than this value):
    CHI_FLOAT f1 = f0 + f0/sqrt((CHI_FLOAT)(N_data - N_PARAMS - N_filters));
    int iseg = 0;
    double phi_M0 = P_phi_M;
    double phi_00 = P_phi_0;
    #if defined(SPHERICAL_K) && defined(TORQUE)
    // Converting torque vector from Cartesian to spherical coordinates, for confidence interval estimation
    // Shortest axis (c), largest moment of inertia:
    double Is = (1.0 + P_b_tumb*P_b_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
    // Intermediate axis (b), intermediate moment of inertia:
    double Ii = (1.0 + P_c_tumb*P_c_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
    // Torque vector cartesean components:
    double Ki = Ii * P_Ti;
    double Ks = Is * P_Ts;
    double phi0 = atan2(Ks, Ki);  // phi (azimuthal angle; 0..2*pi) for the input model
    #endif
    
    if (blockIdx.x==0 && threadIdx.x==0)
    {
        d_f0 = f0;
        d_f1 = f1;
    }
    
    // Global thread index:
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    // Reading the global states from device memory:
    curandState localState = globalState[id];    

//  In RMSD mode, Nstages mean number of random points generated per thread
    for (int istage=0; istage<Nstages; istage++)
    {
        #define SMALL 1e-8  // Small offset from the hard parameter limits
        int LAM = 0;

        for (i=0; i<N_PARAMS; i++)
        {
/*
            // Generating random number in [0..1[ interval:
            float r = curand_uniform(&localState);
            
            CHI_FLOAT xmin = x[0][i] - dx_rand;
            CHI_FLOAT xmax = x[0][i] + dx_rand;
            // Enforcing hard limits:
            if (xmin<SMALL && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_LEFT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
                xmin = SMALL;
            if (xmax>1.0-SMALL && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_RIGHT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
                xmax = 1.0 - SMALL;
            
            x[1][i] = xmin + r*(xmax-xmin);
            */
        CHI_FLOAT xx = x[0][i] + dx_rand * curand_normal(&localState);
            if (xx<SMALL && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_LEFT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
                xx = SMALL;
            else if (xx>1.0-SMALL && (sProperty[i][P_periodic]==HARD_BOTH || sProperty[i][P_periodic]==HARD_RIGHT || LAM==0 && sProperty[i][P_periodic]==PERIODIC_LAM))
                xx = 1.0 - SMALL;
        x[1][i] = xx;
            
        if (sProperty[i][P_type] == T_Es)
            // We need to know LAM to figure out whether psi_0 is periodic (LAM=1) or not (LAM=0):
            LAM = x[1][i]>=0.5;
            
        }

        
        if (x2params(x[1], params, sLimits, &s_x2_params, sProperty, sTypes))
        {
            continue;
        }

        // RMSD value for a randomly shifted model:
        CHI_FLOAT f = chi2one(params, sData, N_data, N_filters, delta_V, 0, &sp, sTypes);  
        
        Ntot++;
        if (f < f1)
        // We found a good model (within one sigma from the input model, in terms of RMSD)
        {
            // Converting periodic angles to proper intervals, for confidence interval calculations
            if (P_phi_M > phi_M0 + PI)
                P_phi_M = phi_M0 - 2*PI;
            if (P_phi_M < phi_M0 - PI)
                P_phi_M = phi_M0 + 2*PI;
            if (P_phi_0 > phi_00 + PI)
                P_phi_0 = phi_00 - 2*PI;
            if (P_phi_0 < phi_00 - PI)
                P_phi_0 = phi_00 + 2*PI;
            #if defined(SPHERICAL_K) && defined(TORQUE)
            // Converting torque vector from Cartesian to spherical coordinates, for confidence interval estimation
            // The results are stored back inside the params vector
            // Shortest axis (c), largest moment of inertia:
            double Is = (1.0 + P_b_tumb*P_b_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
            // Intermediate axis (b), intermediate moment of inertia:
            double Ii = (1.0 + P_c_tumb*P_c_tumb) / (P_b_tumb*P_b_tumb + P_c_tumb*P_c_tumb);
            // Torque vector cartesean components:
            double Ki = Ii * P_Ti;
            double Ks = Is * P_Ts;
            double Kl = P_Tl;
            // Torque vector spherical components (storing them back inside the params vector):
            P_Ti = sqrt(Ki*Ki + Ks*Ks + Kl*Kl);  // r
            P_Ts = acos(Kl/P_Ti);  // theta (polar angle; 0..pi)            
            double phi = atan2(Ks, Ki);
            // Converting phi to the interval phi0+-pi, for proper confidence interval calculations:
            if (phi > phi0 + PI)
                phi = phi - 2*PI;
            if (phi < phi0 - PI)
                phi = phi + 2*PI;
            P_Tl = phi;  // phi (azimuthal angle; phi0-pi..phi0+pi)
            #endif            
            
            // Searching the min/max of dimensional model parameters, for the current thread
            for (i=0; i<N_PARAMS; i++)
            {
                if (params[i] < par_min[i])
                    par_min[i] = params[i];
                if (params[i] > par_max[i])
                    par_max[i] = params[i];
            }
            
        }
        else
        {
            Nbad++;
        }
        
    } // istage loop     
    
    atomicAdd(&d_Ntot, Ntot);
    atomicAdd(&d_Nbad, Nbad);
    
    for (i=0; i<N_PARAMS; i++)
    {
        s_min[threadIdx.x] = par_min[i];
        s_max[threadIdx.x] = par_max[i];
        
        __syncthreads();        
        
        // Doing serial reduction, one parameter at a time - should be fine, as we do this step very infrequently:
        if (threadIdx.x == 0)
        {
            float fmin =  1e30;
            float fmax = -1e30;
            for (int j=0; j<blockDim.x; j++)
            {
                if (s_min[j] < fmin)
                    fmin = s_min[j];
                if (s_max[j] > fmax)
                    fmax = s_max[j];
            }
            // Storing the min/max values of the i-th parameter, found in the current block, into device memory:
            dpar_min[blockIdx.x*N_PARAMS + i] = fmin;
            dpar_max[blockIdx.x*N_PARAMS + i] = fmax;
        }

        __syncthreads();        
        
    }
        

    // Writing the global states to device memory:
    globalState[id] = localState;
    
    return;
    }
    #endif //RMSD


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifdef ANIMATE
    __device__ void compute_rgb(unsigned char * d_rgb, double b, double c, 
                                double E_x, double E_y, double E_z,
                                double Ep_b, double Ep_c, double Ep_a,
                                double Sp_b, double Sp_c, double Sp_a,
                                double b_x, double b_y, double b_z,
                                double c_x, double c_y, double c_z,
                                double a_x, double a_y, double a_z,
                                int i_snapshot, int i1
                               )
    /* Performs projection of asteroid onto the observer's sky plane; computes RGB values for all pixels in the image (size SIZE_PIX x SIZE_PIX)
     * Each thread processes one pixel (described by 4 pixel lines of sight - for the 4 pixel corners)
     */
    {
        __shared__ double Z1x, Z1y, Z1z, Y1x, Y1y, Y1z;
        
        // Global index:
        int id = threadIdx.x + blockDim.x*blockIdx.x;
        // Pixel coordinates:
        int iz = id / SIZE_PIX;
        int iy = id % SIZE_PIX;
        
        if (iy >= SIZE_PIX || iz >= SIZE_PIX)
            return;

        if (threadIdx.x == 0)
        {
            // SSB coordinates of the image coordinate system axes, where X1 axis is directed towards observer (line of sight; =E vector),
            // and Z1=[E x y] (here y is the SSB axis y)
            double scale = 1.0/sqrt(E_x*E_x + E_z*E_z);
            Z1x = -E_z * scale;
            Z1y =  0.0;
            Z1z =  E_x * scale;
            Y1x = -E_x*E_y * scale;
            Y1y = (E_x*E_x + E_z*E_z) * scale;
            Y1z = -E_y*E_z * scale;                        
        }
        
        __syncthreads();        

        float br[3];
        for (int i=0; i<3; i++)
            br[i] = 0.0;
            
        // Loop over four corners of a pixel:
        for (int jz=0; jz<2; jz++)
        {
            // Scale-free offset for x and y in the image plane (total range -1.2...1.2, to include some margins):
            double z1 = 2.4 * (double)(iz - SIZE_PIX/2 + jz) / (double)SIZE_PIX;
            for (int jy=0; jy<2; jy++)
            {
                double y1 = 2.4 * (double)(iy - SIZE_PIX/2 + jy) / (double)SIZE_PIX;
            
                // SSB coordinates of the pixel corner:
                double p_x = y1*Y1x + z1*Z1x;
                double p_y = y1*Y1y + z1*Z1y;
                double p_z = y1*Y1z + z1*Z1z;
                // Same converted to bca coordinates:
                double p_b = b_x*p_x + b_y*p_y + b_z*p_z;
                double p_c = c_x*p_x + c_y*p_y + c_z*p_z;
                double p_a = a_x*p_x + a_y*p_y + a_z*p_z;
                
                // Computing intersection of the pixel line with the triaxial ellipsoid
                // Coefficients used in quadratic equation solution:
                double A = Ep_b*Ep_b/(b*b) + Ep_c*Ep_c/(c*c) + Ep_a*Ep_a;
                double B = 2.0*(p_b*Ep_b/(b*b) + p_c*Ep_c/(c*c) + p_a*Ep_a);
                double C = p_b*p_b/(b*b) + p_c*p_c/(c*c) + p_a*p_a - 1.0;
                // Discriminant:
                double D = B*B - 4.0*A*C;
                int red_spot;
                if (D >= 0.0)
                    // We have an intersection of the pixel line with the ellipsoid
                {
                    // Two solutions for lambda:
                    double lambda = (-B-sqrt(D))/(2.0*A);
                    double lambda1 = (-B+sqrt(D))/(2.0*A);
                    // Choosing the larger one:
                    if (lambda1 > lambda)
                        lambda = lambda1;
                    
                    // bca coordinates of the intersection point:
                    double ri_b = p_b + lambda*Ep_b;
                    double ri_c = p_c + lambda*Ep_c;
                    double ri_a = p_a + lambda*Ep_a;
                    
                    // Creating a red spot at the end of the axis b, to track axial rotation:
                    if (ri_c*ri_c + ri_a*ri_a < SPOT_RAD*SPOT_RAD && ri_b>0.0)
                        red_spot = 1;
                    else
                        red_spot = 0;
                    
                    // Normal to the surface vector at the intersection point:
                    double ni_b = ri_b/(b*b);
                    double ni_c = ri_c/(c*c);
                    double ni_a = ri_a;
                    double scale = 1.0/sqrt(ni_b*ni_b + ni_c*ni_c + ni_a*ni_a);
                    // Normalizing the vector:
                    ni_b = ni_b * scale;
                    ni_c = ni_c * scale;
                    ni_a = ni_a * scale;
                    
                    // Surface brightness at the intersection point (isotropic reflectance assumed); goes from -1 (deepest shadow) to 1 (solar point)
                    double I = ni_b*Sp_b + ni_c*Sp_c + ni_a*Sp_a;
                    if (I >= 0.0)
                    // I>0 corresponds to sun-lit surface element
                    {
                        // Illuminated surface brightness (varies between SMAX and IMAX):
                        br[0] = br[0] + SMAX_R + I*(IMAX_R-SMAX_R);
                        if (red_spot == 0)
                        {
                            br[1] = br[1] + SMAX_G + I*(IMAX_G-SMAX_G);
                            br[2] = br[2] + SMAX_B + I*(IMAX_B-SMAX_B);
                        }
                    }
                    else
                    // If I<0, we have shadow:
                    {
                        // Shadowed surface brightness varies between SMIN (deepest shadow) and SMAX (terminator area):
                        br[0] = br[0] + SMAX_R + I*(SMAX_R-SMIN_R);
                        if (red_spot == 0)
                        {
                            br[1] = br[1] + SMAX_G + I*(SMAX_G-SMIN_G);
                            br[2] = br[1] + SMAX_B + I*(SMAX_B-SMIN_B);                        
                        }
                    }
                    
                    // Brightness is the sum of the constant shadow
                }  // if D>0                        
            } // jy loop
        } // jx loop
        
        // Computing integer RGB values for the pixel (averaging over the 4 pixel corners):
        for (int ic=0; ic<3; ic++)
        {
           int br_int = (int)(br[ic]/4.0);
           if (br_int > 255)
               br_int = 255;
           if (br_int < 0)
               br_int = 0;
           // Flattened array index:
           long int iflat = ic + 3*(iy + (long int)SIZE_PIX*(iz + (long int)SIZE_PIX*(i_snapshot-i1)));
           d_rgb[iflat] = br_int;
        }
               
        return;
    }
#endif
