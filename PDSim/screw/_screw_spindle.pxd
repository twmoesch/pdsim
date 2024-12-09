from PDSim.flow import flow_models
cimport PDSim.flow.flow_models as flow_models

from PDSim.flow.flow_models import FlowFunction
from PDSim.flow.flow_models cimport FlowFunction

from PDSim.flow.flow import FlowPath
from PDSim.flow.flow cimport FlowPath

from PDSim.misc.datatypes import arraym
from PDSim.misc.datatypes cimport arraym

from PDSim.screw.screw_spindle_geo import geoVals
from PDSim.screw.screw_spindle_geo cimport geoVals

import numpy as np
cimport numpy as np

import cython
cimport cython



cdef class _screw_spindle(object):
    cdef public geoVals geo
    cdef public double theta
    cdef public double HTC
    cdef public bint incl_leakage, incl_injection

    cpdef dict __cdict__(self) #TODO nochmal prüfen in _scroll.pyx was hier genau ausgegeben wird
    cpdef double Suction(self, FlowPath FP)
    cpdef double Discharge(self, FlowPath FP)
    cpdef double RadialLeakage(self, FlowPath FP)
    cpdef double InternalIntermeshLeakage(self, FlowPath FP)
    cpdef double ExternalIntermeshLeakage(self, FlowPath FP)
    cpdef double BlowholeLeakage(self, FlowPath FP)
    cpdef double Injection(self, FlowPath FP)






