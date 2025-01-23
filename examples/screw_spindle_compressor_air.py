import pandas as pd
from pathlib import Path
from CoolProp.CoolProp import PropsSI
from CoolProp import State

from PDSim.screw.core import ScrewSpindle

# Zeitstempel
from datetime import datetime
# now = datetime.now()
# str_now = now.strftime('%Y%m%d%H%M%S')

import PDSim

data_folder = Path(PDSim.__file__).parent.parent.joinpath('data')

###################################
# Definition operating conditions #
###################################

OP1 = dict(T1=293.15, p1=400e2, T2=293.15, p2=1000e2, n=4580/60, fluid="air")
OP2 = dict(T1=293.15, p1=850e2, T2=293.15, p2=1000e2, n=5833/60, fluid="air")

OP = OP1
inletState = State.State(OP['fluid'],{'T':OP['T1'],'P':OP['p1']/1000})
outletState = State.State(OP['fluid'],{'T':OP['T2'],'P':OP['p2']/1000})

#########################################
# Definition of fluid and governing EoS #
#########################################
backend='BICUBIC'


#################################
# Generate Screw Spindle object #
#################################

GeomDataFilePath = data_folder.joinpath('GeomData_a-195_redStk-70.0_dphi_deg=1.0_delta_S=1.00e-04.csv')

delta_S = 1e-4

filename='Screw_air'
screw1 = ScrewSpindle(num_lobes=2)

screw1.set_base_geomdata(BaseGeomDataFilePath = GeomDataFilePath)
screw1.set_operation_data(inletState, outletState, OP['n'])
screw1.auto_add_CVs()
screw1.auto_add_suction_discharge_tubes()

filename+='_leak' + '_deltaS_{0:.3e}'.format(delta_S)
screw1.set_leakage_geomdata(LeakGeomDataFilePath = GeomDataFilePath, delta_S=delta_S)
screw1.auto_add_leakage()

# filename+='_inj'
# screw1.set_inj_geomdata(InjGeomDataFilePath = GeomDataFilePath)
# screw1.auto_add_injection(injState=injState)

now = datetime.now()
str_now = now.strftime('%Y%m%d%H%M%S')


screw1.compressor_solve(
    solver_method = 'Euler', EulerN = 20000,
    # solver_method = 'RK45', RK45_eps = 1e-6,
    backend = backend,
    OneCycle = False,
    n = OP["n"],
    eps_cycle=0.005,
    HDF5filename=filename+'_'+str_now+'.h5'
)