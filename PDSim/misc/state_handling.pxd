import cython
cimport cython

from CoolProp.State cimport State

cpdef State copystate(State Original)