import numpy as np
import scipy

def calc_kSpline(Foreground_cut, Shot_Noise_cut, Nspline_points):
    kSpline = np.zeros(Nspline_points)
    kSpline_min = Foreground_cut 
    kSpline_max = Shot_Noise_cut 

    for i in range(Nspline_points):
        kSpline[i] = kSpline_min + (kSpline_max - kSpline_min)*float(i)/(Nspline_points - 1)
    return kSpline

def interp_plan(k, pk):
    # Finds the B-spline representation, given k and pk.
    return scipy.interpolate.splrep(k, np.log10(pk), s=0)

def interp_pk(pk_spl, kSpline, Nspline_points):
    # Evaluates the B-spline at specific k.
    output = np.zeros(Nspline_points)
    for j in range(Nspline_points):
        output[j] = 10**(scipy.interpolate.splev(kSpline[j], pk_spl, der=0))
    return output
