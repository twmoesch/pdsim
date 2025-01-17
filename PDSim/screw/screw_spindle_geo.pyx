from __future__ import division
cimport cython
import cython
import numpy as np

cdef class VdVstruct:
    def __init__(self, V, dV):
        self.V = V
        self.dV = dV
        
    def __repr__(self):
        return str(list([self.V,self.dV]))

#This is a list of all the members in geoVals
# TODO Anpassung an tatsächliche Werte in geoVals
geoValsvarlist=['V_suc_plenum', 'V_dis_plenum', 'V_suc', 'V_dis', 'V_nan',
                'theta_min', 'theta_max', 'theta_suc', 'theta_dis', 'dtheta_chamb',
                'num_chambers', 'num_lobes', 
                #'theta_raw', 'V_raw', 'dV_raw',
                #'A_suc_ax_raw', 'A_suc_rad_raw', 'A_dis_ax_raw', 
                #'A_leak_housing_raw', 'A_leak_radial_raw', 'A_leak_intermesh_internal_raw',
                #'A_leak_intermesh_external_raw', 'A_leak_blowhole1_raw', 'A_leak_blowhole2_raw',
                #'theta_inj_raw', 'A_inj_raw', 'num_inj_tubes'
                #'A_hx_housing1_raw', 'A_hx_housing2_raw',
                #'A_hx_rotor_root_raw', 'A_hx_rotor_crown_raw', 'A_hx_rotor_flank1_raw', 'A_hx_rotor_flank2_raw'
                ]
 
cdef class geoVals:
    def __init__(self, num_lobes):
        self.num_lobes = num_lobes

    def __iter__(self):
        for atr in geoValsvarlist:
            yield atr, getattr(self,atr)
        return None

cdef double get_global_theta(double theta, geoVals geo, int ichamb):
    return theta + geo.theta_min + (ichamb - 1) * pi / geo.num_lobes

cdef double simple_interpolation(double X, double[:] X_all, double[:] Y_all, double Y_ext):
    cdef double X_i
    cdef int i = 0

    if X < X_all[0] or X >= X_all[-1]:
        return Y_ext
    
    for X_i in X_all:
        if X < X_i:
            return (Y_all[i] - Y_all[i-1]) / (X_all[i] - X_all[i-1]) * (X - X_all[i-1]) + Y_all[i-1]
        i += 1
    return Y_ext

cpdef tuple VdV(double theta, geoVals geo, int ichamb):
    """
    Evaluate V and dV/dtheta in a generalized manner for a chamber
    
    Parameters
    ----------
    theta : float
        crank angle in rad
    geo : geoVals
        The structure with the geometry obtained from get_geo()
    ichamb : int
        The chamber number between 1 and geo.num_chambers

    """
    cdef double theta_global = get_global_theta(theta, geo, ichamb)
    #cdef VdVstruct VdV = VdVstruct.__new__(VdVstruct)

    #VdV.V = simple_interpolation(theta_global, geo.theta_raw, geo.V_raw, geo.V_nan)
    #VdV.dV = simple_interpolation(theta_global, geo.theta_raw, geo.dV_raw, 0.0)
    V = simple_interpolation(theta_global, geo.theta_raw, geo.V_raw, geo.V_nan)
    dV = simple_interpolation(theta_global, geo.theta_raw, geo.dV_raw, 0.0)
    if V<geo.V_nan:
        V = geo.V_nan
        dV = 0.0
    return V, dV

cpdef double area_leak(double theta, geoVals geo, int ichamb, leak_id id):
    """
    Evaluate leakage area for a given leakage ID and a chamber 

    Parameters
    ----------
    theta : float
        crank angle in rad
    geo : geoVals
        The structure with the geometry obtained from get_geo()
    ichamb : int
        The chamber number between 1 and geo.num_chambers
    id : int
        The identifier of the leakage flow
    """
    cdef double area, theta_global = get_global_theta(theta, geo, ichamb)

    if id==HOUSING:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_housing_raw, 0.0)
    elif id==RADIAL:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_radial_raw, 0.0)
    elif id==BLOWHOLE:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_blowhole1_raw, 0.0) + simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_blowhole2_raw, 0.0)
    elif id==INTERMESH_INT:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_intermesh_internal_raw, 0.0)
    elif id==INTERMESH_EXT:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_intermesh_external_raw, 0.0)
    else:
        area = 0.0
    return area

cpdef double area_suction(double theta, geoVals geo, int ichamb):
    """
    Evaluate suction area for a chamber 

    Parameters
    ----------
    theta : float
        crank angle in rad
    geo : geoVals
        The structure with the geometry obtained from get_geo()
    ichamb : int
        The chamber number between 1 and geo.num_chambers
    """
    
    cdef double theta_global = get_global_theta(theta, geo, ichamb)
    return simple_interpolation(theta_global, geo.theta_raw, geo.A_suc_ax_raw, 0.0) + simple_interpolation(theta_global, geo.theta_raw, geo.A_suc_rad_raw, 0.0)

cpdef double area_discharge(double theta, geoVals geo, int ichamb):
    """
    Evaluate discharge area for a chamber 

    Parameters
    ----------
    theta : float
        crank angle in rad
    geo : geoVals
        The structure with the geometry obtained from get_geo()
    ichamb : int
        The chamber number between 1 and geo.num_chambers
    """
    cdef double theta_global = get_global_theta(theta, geo, ichamb)
    return simple_interpolation(theta_global, geo.theta_raw, geo.A_dis_ax_raw, 0.0)

cpdef double area_injection(double theta, geoVals geo, int ichamb):
    """
    Evaluate injection area for a chamber 

    Parameters
    ----------
    theta : float
        crank angle in rad
    geo : geoVals
        The structure with the geometry obtained from get_geo()
    ichamb : int
        The chamber number between 1 and geo.num_chambers
    """
    cdef double theta_global = get_global_theta(theta, geo, ichamb)
    return simple_interpolation(theta_global, geo.theta_inj_raw, geo.A_inj_raw, 0.0)

