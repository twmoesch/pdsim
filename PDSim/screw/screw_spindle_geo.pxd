import numpy as np
cimport numpy as np

import cython
cimport cython

from libc.math cimport sqrt,sin,cos,tan,atan2,acos,floor,M_PI as pi,pow

cdef class VdVstruct:
    """
    A struct with values for volume and derivative of volume w.r.t. crank angle
    """

    cdef public double V
    """ Volume [m^3] """

    cdef public double dV
    """ Derivative of volume with respect to crank angle [m^3/radian] """

cdef enum leak_id:
    HOUSING
    RADIAL
    INTERMESH_INT
    INTERMESH_EXT
    BLOWHOLE

cdef class geoVals:
    cdef public double num_lobes, V_suc_plenum, V_dis_plenum, V_suc, V_dis
    cdef public double theta_min, theta_max, theta_suc, theta_dis, dtheta_chamb
    cdef public int num_chambers, num_inj_tubes
    cdef public double[:] theta_raw, V_raw, dV_raw
    cdef public double[:] A_suc_ax_raw, A_suc_rad_raw, A_dis_ax_raw 
    cdef public double[:] A_leak_housing_raw, A_leak_radial_raw, A_leak_intermesh_internal_raw
    cdef public double[:] A_leak_intermesh_external_raw, A_leak_blowhole1_raw, A_leak_blowhole2_raw
    cdef public double[:][:] A_inj_raw
    cdef public double[:] A_hx_housing1_raw, A_hx_housing2_raw
    cdef public double[:] A_hx_rotor_root_raw, A_hx_rotor_crown_raw, A_hx_rotor_flank1_raw, A_hx_rotor_flank2_raw

cdef double get_global_theta(double theta, geoVals geo, int ichamb)
cdef double simple_interpolation(double X, double[:] X_all, double[:] Y_all, double Y_ext)
cpdef VdVstruct VdV(double theta, geoVals geo, int ichamb)
cpdef double area_leak(double theta, geoVals geo, int ichamb, leak_id id)
cpdef double area_suction(double theta, geoVals geo, int ichamb)
cpdef double area_discharge(double theta, geoVals geo, int ichamb)
cpdef double area_injection(double theta, geoVals geo, int ichamb, int itube)