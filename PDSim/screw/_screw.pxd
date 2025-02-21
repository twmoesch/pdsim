from PDSim.flow import flow_models
cimport PDSim.flow.flow_models as flow_models

from PDSim.flow.flow_models import FlowFunction
from PDSim.flow.flow_models cimport FlowFunction

from PDSim.flow.flow import FlowPath
from PDSim.flow.flow cimport FlowPath

from PDSim.misc.datatypes import arraym
from PDSim.misc.datatypes cimport arraym

#from PDSim.screw.screw_spindle_geo import geoVals, leak_id
from PDSim.screw.screw_spindle_geo cimport geoVals, leak_id

import numpy as np
cimport numpy as np

import cython
cimport cython



cdef class _ScrewSpindle(object):
    cdef public geoVals geo
    cdef public double theta
    cdef public double HTC
    cdef public bint incl_leakage, incl_injection

    cpdef dict __cdict__(self)
    cpdef double Suction(self, FlowPath FP, int ichamb)
    cpdef double Discharge(self, FlowPath FP, int ichamb)
    cpdef double SimpleFlow(self, FlowPath FP, float A)
    cpdef double Leakage(self, FlowPath FP, int ichamb, leak_id id, double flow_coeff)
    
    #cpdef double RadialLeakage(self, FlowPath FP, int ichamb)
    #cpdef double InternalIntermeshLeakage(self, FlowPath FP, int ichamb)
    #cpdef double ExternalIntermeshLeakage(self, FlowPath FP, int ichamb)
    #cpdef double BlowholeLeakage(self, FlowPath FP, int ichamb)
    cpdef double Injection(self, FlowPath FP, int ichamb, str upstream_key)






