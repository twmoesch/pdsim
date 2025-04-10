
cpdef State copystate(State original):
    cdef State S
    cdef double T
    cdef double rho
    cdef bytes Fluid
    T = original.T
    rho = original.rho
    Fluid = original.Fluid

    S = State(Fluid.decode(encoding="utf-8"),{'T':T,'D':rho})
    return S