import pandas as pd
from pathlib import Path
from CoolProp.CoolProp import PropsSI
from CoolProp import State

from PDSim.screw.core import ScrewSpindle


import PDSim

data_folder = Path(PDSim.__file__).parent.parent.joinpath('data')

###################################
# Definition operating conditions #
###################################

fluid = "R718"
T0 = 5 + 273.15; p0 = PropsSI('P','T',T0,'Q',1,fluid) 
TC = 50 + 273.15; pC = PropsSI('P','T',TC,'Q',1,fluid) 
Tinj = TC; pinj = pC + 1e4

T1 = T0 + 6; p1 = p0
T2 = TC + 50; p2 = pC

n = 12000 / 60




inletState = State.State(fluid,{'T':T1,'P':p1/1000})
outletState = State.State(fluid,{'T':T2,'P':p2/1000})
injState = State.State(fluid,{'T':Tinj,'P':pinj/1000})

#########################################
# Definition of fluid and governing EoS #
#########################################

backend='BICUBIC'



#################################
# Generate Screw Spindle object #
#################################

GeomDataFilePath = data_folder.joinpath('GeomData_screw_spindle.csv')
filename='screw'
screw1 = ScrewSpindle(num_lobes=2)

screw1.set_base_geomdata(BaseGeomDataFilePath = GeomDataFilePath)
screw1.set_operation_data(inletState, outletState, n)
screw1.auto_add_CVs()
screw1.auto_add_suction_discharge_tubes()

filename+='_leak'
screw1.set_leakage_geomdata(LeakGeomDataFilePath = GeomDataFilePath)
screw1.auto_add_leakage()

filename+='_inj'
screw1.set_inj_geomdata(InjGeomDataFilePath = GeomDataFilePath)
screw1.auto_add_injection(injState=injState)

screw1.compressor_solve(
    solver_method = 'Euler', EulerN = 20000,
    # solver_method = 'RK45', RK45_eps = 1e-6,
    backend = backend,
    OneCycle = False,
    n = n,
    eps_cycle=0.005,
    HDF5filename=filename+'.h5'
)

