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

cdef double get_global_theta(double theta, geoVals geo, int ichamb):
    return theta + geo.theta_min + (ichamb - 1) * pi / geo.num_lobes

cdef double simple_interpolation(double X, double[:] X_all, double[:] Y_all, double Y_ext = 0.0):
    cdef double X_i
    cdef int i = 0

    if X < X_all[0] or X >= X_all[-1]:
        return Y_ext
    
    for X_i in X_all:
        if X < X_i:
            return (Y_all[i] - Y_all[i-1]) / (X_all[i] - X_all[i-1]) * (X - X_all[i-1]) + Y_all[i-1]
        i += 1
    return Y_ext

cpdef VdVstruct VdV(double theta, geoVals geo, int ichamb):
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
    cdef VdVstruct VdV = VdVstruct.__new__(VdVstruct)

    VdV.V = simple_interpolation(theta_global, geo.theta_raw, geo.V_raw)
    VdV.dV = simple_interpolation(theta_global, geo.theta_raw, geo.dV_raw)
    return VdV

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
    id : leak_id
        The identifier of the leakage flow
    """
    cdef double area, theta_global = get_global_theta(theta, geo, ichamb)

    if id==HOUSING:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_housing_raw)
    elif id==RADIAL:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_radial_raw)
    elif id==BLOWHOLE:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_blowhole1_raw) + simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_blowhole2_raw)
    elif id==INTERMESH_INT:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_intermesh_internal_raw)
    elif id==INTERMESH_EXT:
        area = simple_interpolation(theta_global, geo.theta_raw, geo.A_leak_intermesh_external_raw)
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
    return simple_interpolation(theta_global, geo.theta_raw, geo.A_suc_ax_raw) + simple_interpolation(theta_global, geo.theta_raw, geo.A_suc_rad_raw)

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
    return simple_interpolation(theta_global, geo.theta_raw, geo.A_dis_ax_raw)

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
    return simple_interpolation(theta_global, geo.theta_raw, geo.A_inj_raw)

