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


__device__ CHI_FLOAT chi2one(double *params, struct obs_data *sData, int N_data, int N_filters, CHI_FLOAT *delta_V, int Nplot, struct chi2_struct *s_chi2_params, int sTypes[][N_SEG])
// Computung chi^2 for a single model parameters combination, on GPU, by a single thread
// NUDGE is not supported in SEGMENT mode!
{
    int i, m;
    double cos_alpha_p, sin_alpha_p, scalar_Sun, scalar_Earth, scalar;
    double cos_lambda_p, sin_lambda_p, Vmod, alpha_p, lambda_p;
    double Ep_x, Ep_y, Ep_z, Sp_x, Sp_y, Sp_z;
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
     *      - theta_K: Orientation angle theta for the vector <r> for the surface point where the torque is applied (asteroid's frame of reference); range 0 ... pi
     *      - phi_K: Orientation angle phi for the vector <r> for the surface point where the torque is applied (asteroid's frame of reference); range 0... 2*pi (periodic)
     *      - phi_F: direction of the torque force in the plane, perpendiculat to <r> (the reference direction is that of T=[r x a], towards the vector D=[T x r]); range 0... 2*pi (periodic)
     *      - K: amplitude of the torque force, |r|*|F_t|, where F_t is the tangential componenbt of the force; units are rad/day^2
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
        #define P_theta_K  params[sTypes[T_theta_K][iseg]]
        #define P_phi_K    params[sTypes[T_phi_K][iseg]]
        #define P_phi_F    params[sTypes[T_phi_F][iseg]]
        #define P_K        params[sTypes[T_K][iseg]]
        #define P_c_tumb   params[sTypes[T_c_tumb][iseg]]
        #define P_b_tumb   params[sTypes[T_b_tumb][iseg]]
        #define P_Es       params[sTypes[T_Es][iseg]]
        #define P_psi_0    params[sTypes[T_psi_0][iseg]]
        #define P_c        params[sTypes[T_c][iseg]]
        #define P_b        params[sTypes[T_b][iseg]]
        
        
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
        double theta = asin(sqrt((P_Es-1.0)/(sin(P_psi_0)*sin(P_psi_0)*(Ii_inv-Is_inv)+Is_inv-1.0)));    
        double psi = P_psi_0;
        
        #ifdef NUDGE    
        float t_mod[M_MAX], V_mod[M_MAX];
        float t_old[2];
        float V_old[2];
        int M = 0;
        #endif    
        
        #ifdef TORQUE
        // Coordinates of the unit vector <r> (the point where the torque force is applied) in the asteroid's frame of reference, bca (isl):
        double r_x = sin(P_theta_K) * sin(P_phi_K);
        double r_y = sin(P_theta_K) * cos(P_phi_K);
        double r_z = cos(P_theta_K);
        
        double r_xy = sqrt(r_x*r_x + r_y*r_y);
        // Unit vector T=[r x a] (the z-th component = 0):
        double T_x = r_y / r_xy;
        double T_y = -r_x / r_xy;
        // Unit vector D=[T x r]:
        double D_x = -r_x*r_z/r_xy;
        double D_y = -r_y*r_z/r_xy;
        double D_z = r_xy;
        
        double cos_phi_F = cos(P_phi_F);
        double sin_phi_F = sin(P_phi_F);
        // The torque force is in the plane defined by two perpendicular vectors T and D, at the angle phi_F counting from T towards D
        // Unit vector in the direction of the torque force:
        double F_x = T_x*cos_phi_F + D_x*sin_phi_F;
        double F_y = T_y*cos_phi_F + D_y*sin_phi_F;
        double F_z =                 D_z*sin_phi_F;
        
        // The torque K=[r x F] , here isl is xyz:
        double Ki = P_K * (r_y*F_z - r_z*F_y);
        double Ks = P_K * (r_z*F_x - r_x*F_z);
        double Kl = P_K * (r_x*F_y - r_y*F_x);
        
        double mu[6];
        // Parameters for the ODEs (don't change with time):
        // Using the fact that Il = 1:
        mu[0] = (Is-1.0)*Ii_inv;
        mu[1] = (1.0-Ii)*Is_inv;
        mu[2] = Ii - Is;
        mu[3] = Ki * Ii_inv;
        mu[4] = Ks * Is_inv;
        mu[5] = Kl;
        
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
        i1 = s_chi2_params->start_seg[iseg];
        if (iseg < N_SEG-1)
            i2 = s_chi2_params->start_seg[iseg+1];
        else
            i2 = N_data;
        #else
        i1 = 0;
        i2 = N_data;
        #endif    
        
        // The loop over all data points in the current segment 
        for (i=i1; i<i2; i++)
        {                                
            
            // Derive the three Euler angles theta, phi, psi here, by solving three ODEs numerically
            if (i > i1)
            {
                int N_steps;
                double h;
                
                // How many integration steps to the current (i-th) observed value, from the previous (i-1) one:
                // Forcing the maximum possible time step of TIME_STEP days (macro parameter), to ensure accuracy
                N_steps = (sData[i].MJD - sData[i-1].MJD) / TIME_STEP + 1;
                // Current equidistant time steps (h<=TIME_STEP):
                h = (sData[i].MJD - sData[i-1].MJD) / N_steps;
                
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
                
                // RK4 method for solving ODEs with a fixed time step h
                for (int l=0; l<N_steps; l++)
                {
                    double K1[N_ODE], K2[N_ODE], K3[N_ODE], K4[N_ODE], f[N_ODE];
                    
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
                #else
                phi = y[0];
                theta = y[1];
                psi = y[2];                    
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
            
            // Now following Muinonen & Lumme, 2015 to compute the visual brightness of the asteroid.
            // Attention! My (Samarasinha and A'Hearn 1991) frame of reference is b-c-a, but the Muinonen's frame is a-b-c
            // On 17.10.2018 the bug was fixed, and now I properly convert the Muinonen's equations to the b-c-a frame
            
            // Earth vector in the new (b,c,a) basis
            // Switching from Muinonen coords (abc) to Samarasinha coords (bca)
            Ep_x = b_x*sData[i].E_x + b_y*sData[i].E_y + b_z*sData[i].E_z;
            Ep_y = c_x*sData[i].E_x + c_y*sData[i].E_y + c_z*sData[i].E_z;
            Ep_z = a_x*sData[i].E_x + a_y*sData[i].E_y + a_z*sData[i].E_z;
            
            // Sun vector in the new (b,c,a) basis
            // Switching from Muinonen coords (abc) to Samarasinha coords (bca)
            Sp_x = b_x*sData[i].S_x + b_y*sData[i].S_y + b_z*sData[i].S_z;
            Sp_y = c_x*sData[i].S_x + c_y*sData[i].S_y + c_z*sData[i].S_z;
            Sp_z = a_x*sData[i].S_x + a_y*sData[i].S_y + a_z*sData[i].S_z;
            
            // Now that we converted the Earth and Sun vectors to the internal asteroidal basis (a,b,c),
            // we can apply the formalism of Muinonen & Lumme, 2015 to calculate the brightness of the asteroid.
            
            #ifdef BC
            double b = P_b;
            double c = P_c;
            #else
            double b = P_b_tumb;
            double c = P_c_tumb;
            #endif        
            
            // The two scalars from eq.(12) of Muinonen & Lumme, 2015; assuming a=1
            // Switching from Muinonen coords (abc) to Samarasinha coords (bca)
            scalar_Sun   = sqrt(Sp_x*Sp_x/(b*b) + Sp_y*Sp_y/(c*c) + Sp_z*Sp_z);
            scalar_Earth = sqrt(Ep_x*Ep_x/(b*b) + Ep_y*Ep_y/(c*c) + Ep_z*Ep_z);
            
            // From eq.(13):
            // Switching from Muinonen coords (abc) to Samarasinha coords (bca)
            cos_alpha_p = (Sp_x*Ep_x/(b*b) + Sp_y*Ep_y/(c*c) + Sp_z*Ep_z) / (scalar_Sun * scalar_Earth);
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
            
            #ifdef TREND
            // Solar phase angle:
            double alpha = acos(Sp_x*Ep_x + Sp_y*Ep_y + Sp_z*Ep_z);
            // De-trending the brightness curve:
            Vmod = Vmod - P_A*alpha;
            #endif        
            
            if (Nplot > 0)
            {
                d_Vmod[i] = Vmod + delta_V[0]; //???
            }
            else
            {
                // Filter:
                int m = sData[i].Filter;
                // Difference between the observational and model magnitudes:
                double y = sData[i].V - Vmod;                    
                sum_y2[m] = sum_y2[m] + y*y*sData[i].w;
                sum_y[m] = sum_y[m] + y*sData[i].w;
                sum_w[m] = sum_w[m] + sData[i].w;
            }
            #ifdef NUDGE
            // Determining if the previous time point was a local minimum
            if (i < 2)
            {
                t_old[i] = sData[i].MJD;
                V_old[i] = Vmod;
            }
            else
            {
                if (V_old[1]>V_old[0] && V_old[1]>=Vmod) 
                    // We just found a brightness minimum (V maximum), between i-2 ... i
                {
                    bool local=0;
                    for (int ii=0; ii<s_chi2_params->N_obs; ii++)
                        // If the model minimum at t_old[1] is within DT_MAX2 days from any observed minimum in s_chi2_params structure, we mark it as local.
                        // It can now contribute to the merit function calculations later in the kernel.
                        if (fabs(t_old[1]-s_chi2_params->t_obs[ii]) < DT_MAX2)
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
            
            #ifdef MIN_DV
            if (sData[i].MJD > DV_MARGIN && sData[i].MJD < sData[N_data-1].MJD-DV_MARGIN)
                if (Vmod > Vmax)
                    Vmax = Vmod;
                if (Vmod < Vmin)
                    Vmin = Vmod;
                #endif
                
        } // data points loop
        
        
    } // for (iseg) loop
    
    if (Nplot > 0)
        return 0.0;
    
    CHI_FLOAT chi2m;
    chi2a=0.0;    
    for (m=0; m<N_filters; m++)
    {
        // Chi^2 for the m-th filter:
        chi2m = sum_y2[m] - sum_y[m]*sum_y[m]/sum_w[m];
        chi2a = chi2a + chi2m;
        // Average difference Vdata-Vmod for each filter (used for plotting):
        // In SEGMENT mode, computation is done here, over all the segments, as the model scaling (with its size) is fixed across all the segments
        delta_V[m] = sum_y[m] / sum_w[m];
    }   
    
    chi2a = chi2a / (N_data - N_PARAMS - N_filters);
    
    #ifdef NUDGE
    // Here we will modify the chi2a value based on how close model minima are to the corresponding observed minima (in 2D - both t and V axes),
    // and will punish if the number of model local minima gets too high.    
    float S_M = 0.0;
    float P_tot = 1.0;
    for (int imod=0; imod < M; imod++)
        // Loop over all detected local model minima
    {
        for (int iobs=0; iobs < s_chi2_params->N_obs; iobs++)
            // Loop over all the observed minima in s_chi2_params structure
        {
            float dt = fabs(t_mod[imod]-s_chi2_params->t_obs[iobs]);
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
                    float dV = V_mod[imod] + delta_V[0] - s_chi2_params->V_obs[iobs];
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
    P_tot = powf(P_tot, 1.0/s_chi2_params->N_obs); // Normalizing the reward to the number of observed minima
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

__device__ void params2x(CHI_FLOAT *x, double *params, CHI_FLOAT sLimits[][N_TYPES], int sProperty[][N_COLUMNS], int sTypes[][N_SEG])
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
        if (sProperty[i][P_frozen] == 1)
        {
            // For frozen parameters, arbitrarily setting x to zero:
            x[i] = 0;
            continue;
        }
        
        if (sProperty[i][P_independent] == 1)
        {
            double par = params[i];
            if (sProperty[i][P_periodic] == 1)
            {
                x[i] = par / (2*PI);
            }
            else
            {
                if (param_type == T_c_tumb)
                    par = log(par);
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
            else if (param_type == T_c)
            {
                // Parameter "c" has the same limits as "c_tumb", and log distribution:
                x[i] = (log(params[i]) - sLimits[0][T_c_tumb]) / (sLimits[1][T_c_tumb] - sLimits[0][T_c_tumb]);        
            }
            
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

__device__ int x2params(CHI_FLOAT *x, double *params, CHI_FLOAT sLimits[][N_TYPES], struct x2_struct *s_x2_params, int sProperty[][N_COLUMNS], int sTypes[][N_SEG])
// Conversion from dimensionless x[] parameters to the physical ones params[]
// RANDOM_BC is not supported yet
{    
    // LAM (=1) or SAM (=0):
    int LAM;
    
    // Checking if we went beyond the limits:
    int failed = 0;
    for (int i=0; i<N_PARAMS; i++)
    {
        int param_type = sProperty[i][P_type];
        
        if (param_type == T_Es)
            LAM = x[i]>=0.5;
        
        // Periodic parameters (all phi parameters and psi_0 - for LAM=1 only) can have any value during optimization:
        if (sProperty[i][P_periodic]==1 || param_type==T_psi_0 && LAM)
            continue;
        
        #ifdef RELAXED
        #if defined(P_PHI) || defined(P_PSI) || defined(P_BOTH)
        // Relaxing only c_tumb in P_PHI / P_PSI / combined modes
        if (param_type == T_c_tumb)
            continue;
        #else        
        // Relaxing L and c_tumb: (physical values are enforced below)
        if (param_type == T_L || param_type == T_c_tumb)
            continue;
        #endif
        #ifdef BC        
        // Relaxing c:
        if (param_type == T_c)
            continue;
        #endif        
        #endif
        
        if (x[i]<0.0 || x[i]>=1.0)
            failed = 1;
    }
    
    if (failed)
        return failed;
    
    double log_c_tumb, log_b_tumb, Is, Ii, psi_min, psi_max;
    int iseg;
    #ifdef BC
    double log_c;
    #endif
    
    // The x -> params conversion
    for (int i=0; i<N_PARAMS; i++)
    {
        int param_type = sProperty[i][P_type];
        
        // First we start with special cases parameters:
        
        if (param_type == T_b_tumb)
        {
            log_b_tumb = log_c_tumb * (x[i]*(sLimits[1][param_type]-sLimits[0][param_type]) + sLimits[0][param_type]);
            params[i] = exp(log_b_tumb);
            double b_tumb = params[i];
            double c_tumb = params[sTypes[T_c_tumb][0]];
            Is = (1.0+b_tumb*b_tumb) / (b_tumb*b_tumb + c_tumb*c_tumb);
            Ii = (1.0+c_tumb*c_tumb) / (b_tumb*b_tumb + c_tumb*c_tumb);
        }
        
        else if (param_type == T_Es)
        {
            LAM = x[i]>=0.5;
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
        else if (sProperty[i][P_periodic] == 1)
        {
            params[i] = x[i] * 2.0*PI;
        }
        
        // Independent non-periodic parameters; all dependent non-periodic parameters have to be handled separately, as special cases
        else if (sProperty[i][P_periodic] == 0 && sProperty[i][P_independent] == 1)
        {
            #ifdef P_PHI
            // Only in P_PHI mode, L parameter is not computed here, but a few lines below
            if (param_type != T_L)
                #endif
                // The default way to compute params[i] from x[i]:
                params[i] = x[i] * (sLimits[1][param_type]-sLimits[0][param_type]) + sLimits[0][param_type];
            
            if (param_type == T_c_tumb)
            {
                log_c_tumb = params[i];
                #ifdef RELAXED
                // Enforcing minimum limits on physical values of L and c:
                if (log_c_tumb>0.0)
                    return 1;
                #endif
                params[i] = exp(log_c_tumb);
            }
            #ifdef BC
            else if (param_type == T_c)
            {
                log_c = params[i];
                #ifdef RELAXED
                // Minimum enforcement on c2 in relaxed mode:
                if (log_c > 0.0)
                    return 1;
                #endif  
                if (fabs(log_c-log_c_tumb) > BC_DEV_MAX)
                    return 1;
                params[i] = exp(log_c);
            }
            #endif            
            else if (param_type == T_L)
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
                    params[i] = (x[i] * (S_LAM0*sLimits[1][param_type]-sLimits[0][param_type]) + sLimits[0][param_type]) / params[sTypes[T_Es][0]];
                else
                    params[i] = (x[i] * (S_LAM1*sLimits[1][param_type]-sLimits[0][param_type]) + sLimits[0][param_type]) * Ii;                
                #endif
                
                #if defined(P_PSI) || defined(P_BOTH)
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
                // In the P_BOTH mode we have to use a rejection method to prune out modesl with the wrong combination of Ppsi and Pphi
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
                
                #else  // if any P_* mode                
                #ifdef RELAXED
                // Enforcing minimum limits on physical values of L:
                if (params[sTypes[T_L][iseg]] < 0.0)
                    return 1;
                #endif                
                #endif  // if any P_* mode                
            } // if (param_type == T_L)
            
        }  // if param_type
        
    }  // for (i)
    
    
    
    return 0;
}



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__global__ void chi2_gpu (struct obs_data *dData, int N_data, int N_filters,
                          curandState* globalState, CHI_FLOAT *d_f, struct x2_struct x2_params)
// CUDA kernel computing chi^2 on GPU
{        
    #ifndef NO_SDATA
    __shared__ struct obs_data sData[MAX_DATA];
    #endif
    __shared__ CHI_FLOAT sLimits[2][N_TYPES];
    __shared__ volatile CHI_FLOAT s_f[BSIZE];
    __shared__ volatile int s_thread_id[BSIZE];
    __shared__ int sProperty[N_PARAMS][N_COLUMNS];
    __shared__ int sTypes[N_TYPES][N_SEG];
    //    __shared__ CHI_FLOAT s_f[BSIZE];
    //    __shared__ int s_thread_id[BSIZE];
    __shared__ struct chi2_struct s_chi2_params;
    __shared__ struct x2_struct s_x2_params;
    int i, j;
    double params[N_PARAMS];
    CHI_FLOAT delta_V[N_FILTERS];
    
    // Not efficient, for now:
    if (threadIdx.x == 0)
    {
        #ifndef NO_SDATA
        for (i=0; i<N_data; i++)
            sData[i] = dData[i];
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
            #ifdef NUDGE
            // Copying the data on the observed minima from device to shared memory:
            s_chi2_params = d_chi2_params;
        #endif
        #ifdef P_BOTH
        s_x2_params = x2_params;
        #endif       
    }
    #ifdef REOPT
    // Reading the initial point from device memory
    for (i=0; i<N_PARAMS; i++)
        params[i] = d_params0[i];
    #endif        
    #ifdef SEGMENT
    // Copying the starting indexes for data segments to shared memory:
    for (i=0; i<N_SEG; i++)
        s_chi2_params.start_seg[i] = d_start_seg[i];
    #endif    
    
    // Downhill simplex optimization approach
    
    __syncthreads();
    
    // Global thread index:
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    // Reading the global states from device memory:
    curandState localState = globalState[id];
    
    //Simplex steps counter:
    int l = 0;
    
    CHI_FLOAT x[N_PARAMS+1][N_PARAMS];  // simplex points (point index, coordinate)
    CHI_FLOAT f[N_PARAMS+1]; // chi2 values for the simplex edges (point index)
    int ind[N_PARAMS+1]; // Indexes to the sorted array (point index)
    
    #ifdef P_BOTH
    bool failed;
    //    for (int itry=0; itry<100; itry++)
    while (1)
    {
        #endif
        
        
        #ifdef REOPT
        // Converting from physical to dimensionless (0...1 scale) parameters:
        params2x(x[0], params, sLimits, sProperty, sTypes);
        
        // Random displacement of the initial point, uniformly distributed within +-0.5*DX_RAND:
        for (i=0; i<N_PARAMS; i++)
        {
            // Sticking to the 0...1 interval for x:
            // (Allowed to switch LAM/SAM here)
            double x_min = 0;
            if (x[0][i] - 0.5*DX_RAND > 0.0)
                x_min = x[0][i] - 0.5*DX_RAND;
            double x_max = 1;
            if (x[0][i] + 0.5*DX_RAND < 1.0)
                x_max = x[0][i] + 0.5*DX_RAND;
            x[0][i] = x[0][i] + DX_RAND*curand_uniform(&localState) + x_min;            
        }
        #else  // REOPT  
        // Initial random point
        for (i=0; i<N_PARAMS; i++)
        {
            #ifdef BC
            #ifndef RANDOM_BC
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
            #endif  // RANDOM_BC 
            #endif  // BC      
            // Random number [0,1[
            float r = curand_uniform(&localState);
            if (sProperty[i][P_type] == T_Es)
            {
                // Using the x value for Es to determine the mode (1:LAM. 0:SAM)
                int LAM = r>=0.5;
                if (LAM == 0)
                    // Interval 1e-6 ... 0.5-DX_INI-1e-6:
                    x[0][i] = 1e-6 + (1 - 2*DX_INI - 4e-6) * r;
                else
                    // Interval 0.5+1e-6 ... 1-DX_INI-1e-6:
                    x[0][i] = 0.5 + 1e-6 + (1 - 2*DX_INI - 4e-6) * (r-0.5);
            }
            else
                // The DX_INI business is to prevent the initial simplex going beyong the limits
                // The allowed interval is 1e-6 ... 1-DX_INI-1e-6
                x[0][i] = 1e-6 + (1.0 - DX_INI - 2e-6) * r;
        }
        #endif // REOPT    
        
        // Simplex initialization
        // Vertex loop:
        for (j=1; j<N_PARAMS+1; j++)
        {
            // Coordinates (parameters) loop:
            for (i=0; i<N_PARAMS; i++)
            {
                if (i == j-1)
                {
                    #ifdef REOPT
                    // In REOPT mode, initial displacements are random, with log distrubution between DX_MIN and DX_MAX:
                    CHI_FLOAT dx_ini = exp(curand_uniform(&localState) * (DX_MAX-DX_MIN) + DX_MIN);
                    // !!! Will fail for some displacements:
                    x[j][i] = x[0][i] + dx_ini;
                    #else                
                    x[j][i] = x[0][i] + DX_INI;
                    #endif                
                }
                else
                {
                    x[j][i] = x[0][i];
                }
            }
        }
        
        // Computing the initial function values (chi2):        
        #ifdef P_BOTH
        failed = 0;
        #endif    
        for (j=0; j<N_PARAMS+1; j++)
        {
            #ifdef P_BOTH
            if (x2params(x[j], params, sLimits, &s_x2_params, sProperty, sTypes))
            {
                failed = 1;
                break;
            }
            #else        
            x2params(x[j], params, sLimits, &s_x2_params, sProperty, sTypes);
            #endif        
            f[j] = chi2one(params, sData, N_data, N_filters, delta_V, 0, &s_chi2_params, sTypes);    
        }
        
        #ifdef P_BOTH
        if (failed == 0)
            break;
    }  // end of for loop
    #endif
    
    
    // The main simplex loop
    while (1)
    {
        #ifdef P_BOTH
        if (failed == 1)
            break;
        #endif
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
            if (sProperty[i][P_frozen] == 0)
                x_r[i] = x0[i] + ALPHA_SIM*(x0[i] - x[ind[N_PARAMS]][i]);
        }
        CHI_FLOAT f_r;
        if (x2params(x_r,params,sLimits, &s_x2_params, sProperty, sTypes))
            f_r = 1e30;
        else
            f_r = chi2one(params, sData, N_data, N_filters, delta_V, 0, &s_chi2_params, sTypes);
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
                if (sProperty[i][P_frozen] == 0)
                    x_e[i] = x0[i] + GAMMA_SIM*(x_r[i] - x0[i]);
            }
            CHI_FLOAT f_e;
            if (x2params(x_e,params,sLimits, &s_x2_params, sProperty, sTypes))
                f_e = 1e30;
            else
                f_e = chi2one(params, sData, N_data, N_filters, delta_V, 0, &s_chi2_params, sTypes);
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
            if (sProperty[i][P_frozen] == 0)
                x_r[i] = x0[i] + RHO_SIM*(x[ind[N_PARAMS]][i] - x0[i]);
        }
        if (x2params(x_r,params,sLimits, &s_x2_params, sProperty, sTypes))
            f_r = 1e30;
        else
            f_r = chi2one(params, sData, N_data, N_filters, delta_V, 0, &s_chi2_params, sTypes);
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
        
        // If all else fails - shrink
        bool bad = 0;
        for (j=1; j<N_PARAMS+1; j++)
        {
            for (i=0; i<N_PARAMS; i++)
            {
                if (sProperty[i][P_frozen] == 0)
                    x[ind[j]][i] = x[ind[0]][i] + SIGMA_SIM*(x[ind[j]][i] - x[ind[0]][i]);
            }           
            if (x2params(x[ind[j]],params,sLimits, &s_x2_params, sProperty, sTypes))
                bad = 1;
            else
                f[ind[j]] = chi2one(params, sData, N_data, N_filters, delta_V, 0, &s_chi2_params, sTypes);
        }
        // We failed the optimization
        if (bad)
        {
            f[ind[0]] = 1e30;
            break;
        }
        
    }  // inner while loop
    
    
    #ifdef P_BOTH
    if (failed == 1 || f[ind[0]] < 1e-5)
        s_f[threadIdx.x] = 1e30;
    else
        s_f[threadIdx.x] = f[ind[0]];
    #else        
    s_f[threadIdx.x] = f[ind[0]];
    #endif
    s_thread_id[threadIdx.x] = threadIdx.x;
    
    __syncthreads();
    //!!! AT this point not all warps initialized s_f and s_thread_id! It looks like __syncthreads doesn't work!
    
    // Binary reduction:
    int nTotalThreads = blockDim.x;
    while(nTotalThreads > 1)
    {
        int halfPoint = nTotalThreads / 2; // Number of active threads
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint; // the second element index
            float temp = s_f[thread2];
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
    
    if (threadIdx.x == s_thread_id[0] && s_f[0] < d_f[blockIdx.x])
        // Keeping the current best result if it's better than the previous kernel result for the same blockID
    {
        // Copying the found minimum to device memory:
        d_f[blockIdx.x] = s_f[0];
        x2params(x[ind[0]],params,sLimits, &s_x2_params, sProperty, sTypes);
        for (int i=0; i<N_PARAMS; i++)
            d_params[blockIdx.x][i] = params[i];
    }
    
    
    // Writing the global states from device memory:
    globalState[id] = localState;
    
    return;        
    
}

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
                           struct obs_data *dPlot, int Nplot, double * d_dlsq2)
// CUDA kernel to compute plot data from input params structure
{     
    __shared__ double sd2_min[BSIZE];
    CHI_FLOAT delta_V[N_FILTERS];
    __shared__ struct chi2_struct s_chi2_params;
    __shared__ int sTypes[N_TYPES][N_SEG];
    double params[N_PARAMS];
    
    // Global thread index for points:
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    //    if (id == 0 && blockIdx.y == 0)    
    // Doing once per kernel
    if (threadIdx.x == 0)
    {
        // Reading the initial point from device memory
        for (int i=0; i<N_PARAMS; i++)
            params[i] = d_params0[i];
        for (int i=0; i<N_TYPES; i++)
        {
//            sLimits[0][i] = dLimits[0][i];
//            sLimits[1][i] = dLimits[1][i];
            for (int iseg=0; iseg<N_SEG; iseg++)
                sTypes[i][iseg] = dTypes[i][iseg];
        }
        #ifdef NUDGE
        // Copying the data on the observed minima from device to shared memory:
        s_chi2_params = d_chi2_params;
        #endif
        // !!! Will not work in NUDGE mode - NULL
        // Step one: computing constants for each filter using chi^2 method, and the chi2 value
        d_chi2_plot = chi2one(params, dData, N_data, N_filters, delta_V, 0,  &s_chi2_params, sTypes);
        d_delta_V0 = delta_V[0];
        
        // Step two: computing the Nplots data points using the delta_V values from above:
        chi2one(params, dPlot, Nplot, N_filters, delta_V, Nplot,  &s_chi2_params, sTypes);
        
    }
    
    __syncthreads();
    
    int blockid = blockIdx.x + gridDim.x*blockIdx.y;
    
    #ifdef LSQ    
    // Computing 2D least squares distances between the data points and the model
    // Each block processes one data point
    // Asssuming that the number of blocks is larger or equal to the number of data points!
    int idata = blockid;
    if (idata < N_data)
    {
        double d2_min = HUGE;
        // Spreading N_plot model points over blockDim.x threads as evenly as possible:
        for (int imodel=threadIdx.x; imodel<Nplot; imodel=imodel+blockDim.x)
        {
            double dist_t = (dPlot[imodel].MJD - dData[idata].MJD) / T_SCALE;
            // To save some time:
            if (fabs(dist_t) < 2.0)
            {
                double dist_V = (d_Vmod[imodel] - dData[idata].V) / V_SCALE;
                // 2D (in t-V axes) distance between the imodel model point and idata data point, using scales V_SCALE and T_SCALE for the two axes:
                double d2 = dist_V*dist_V + dist_t*dist_t;
                // Per-thread minimum of d2:
                if (d2 < d2_min)
                    d2_min = d2;
            }
        }
        sd2_min[threadIdx.x] = d2_min;
        __syncthreads();
        // Binary reduction:
        int nTotalThreads = blockDim.x;
        while(nTotalThreads > 1)
        {
            int halfPoint = nTotalThreads / 2; // Number of active threads
            if (threadIdx.x < halfPoint) {
                int thread2 = threadIdx.x + halfPoint; // the second element index
                double temp = sd2_min[thread2];
                if (temp < sd2_min[threadIdx.x])
                    sd2_min[threadIdx.x] = temp;
            }
            __syncthreads();
            nTotalThreads = halfPoint; // Reducing the binary tree size by two
        }
        // At this point, the smallest d2 in the block is in sd2_min[0]
        if (threadIdx.x == 0)
            d_dlsq2[idata] = sd2_min[0];
    }
    #endif    
    
    #ifdef PROFILES
    if (id < Nplot)
    {
        // Parameter index:
        int iparam = blockIdx.y;            
        // Changes from -DELTA_MAX to +DELTA_MAX:
        // With id+1.0 we ensure that delta=0 corresponds to one of the threads
        double delta = 2.0 * DELTA_MAX * ((id+1.0)/blockDim.x/gridDim.x - 0.5);
        
        // Modyfing slightly the corresponding parameter:
        switch (iparam)
        {
            case 0:
                params.theta_M = params.theta_M + delta * (dLimits[1][iparam] - dLimits[0][iparam]);
                break;
                
            case 1:
                params.phi_M = params.phi_M + delta * (dLimits[1][iparam] - dLimits[0][iparam]);
                break;
                
            case 2:
                params.phi_0 = params.phi_0 + delta * (dLimits[1][iparam] - dLimits[0][iparam]);
                break;
                
            case 3:
                params.L = params.L + delta * (dLimits[1][iparam] - dLimits[0][iparam]);
                break;
                
            case 4:
                params.c_tumb = params.c_tumb * exp(delta * (dLimits[1][iparam] - dLimits[0][iparam]));
                break;
                
            case 5:
                //        params.b_tumb = params.b_tumb * exp(delta * (dLimits[1][4] - dLimits[0][4]));
                params.b_tumb = params.b_tumb * exp(delta * (-log(params.c_tumb)));
                break;
                
            case 6:
                params.Es = params.Es + delta*0.5;  //??
                break;
                
            case 7:
                params.psi_0 = params.psi_0 + delta*2.0*PI;
                break;                    
                #ifdef BC
            case 8:
                params.c = params.c * exp(delta * (dLimits[1][4+DN_IND] - dLimits[0][4+DN_IND]));
                break;
                
            case 9:
                params.b = params.b * exp(delta * (dLimits[1][4+DN_IND] - dLimits[0][4+DN_IND]));
                break;
                #endif        
        }
        
        // Computing the chi2 for the shifted parameter:
        // !!! Will not work in NUDGE mode - NULL
        d_chi2_lines[iparam][id] = chi2one(params, dData, N_data, N_filters, delta_V, 0, &s_chi2_params, sTypes);
    }
    #endif    
    
    return;   
}



//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void setup_kernel ( curandState * state, unsigned long seed, CHI_FLOAT *d_f)
{
    // Global thread index:
    unsigned long long id = blockIdx.x*blockDim.x + threadIdx.x;
    // Generating initial states for all threads in a kernel:
    curand_init ( (unsigned long long)seed, id, 0, &state[id] );
    
    if (threadIdx.x==0)
    {
        d_f[blockIdx.x] = 1e30;    
    }
    
    return;
} 
