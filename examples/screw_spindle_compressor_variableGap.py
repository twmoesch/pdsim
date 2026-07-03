import pandas as pd
from pathlib import Path
from CoolProp.CoolProp import PropsSI, HAPropsSI
from CoolProp import State


from PDSim.screw.core import ScrewSpindle

# Zeitstempel
from datetime import datetime
# now = datetime.now()
# str_now = now.strftime('%Y%m%d%H%M%S')

import PDSim

data_folder = Path(PDSim.__file__).parent.parent.joinpath('data')
result_folder = Path(PDSim.__file__).parent.parent.joinpath('results')

###################################
# Definition operating conditions #
###################################

OPs = pd.read_csv(data_folder.joinpath('ExperimentalData_FuDe1.csv'),sep=';',index_col='OP').to_dict('index')

selected_OPs = ['OP6']    

#########################################
# Definition of fluid and governing EoS #
#########################################
backend='BICUBIC'


#################################
# Generate Screw Spindle object #
#################################

# GeomDataFilePath = data_folder.joinpath('GeomData_a-195_redStk-70.0_dphi_deg=1.0_delta_S=1.00e-04.csv')
GeomDataFilePath = data_folder.joinpath('GeomData_screw_spindle.csv')
# LeakDataFilePath = data_folder.joinpath('GeomData_a-195_LEAK_dphi_deg=1.0_delta_phi_fl_deg=1.50e+00_offN=3.50e-04_offR=1.00e-03.csv')
LeakDataFilePath = data_folder.joinpath('LeakData_a-195_redStk-70.0_dphi_deg=1.0_delta_phi_fl_deg=1.50e+00_offN=3.50e-04_offR=1.50e-03.csv')
#LeakDataFilePath = data_folder.joinpath('LeakData_a-195_redStk-70.0_dphi_deg=1.0_delta_phi_fl_deg=1.50e+00_offN=3.30e-04_offR=1.40e-03.csv')
#LeakDataFilePath = data_folder.joinpath('LeakData_a-195_redStk-70.0_dphi_deg=1.0_delta_phi_fl_deg=1.50e+00_offN=3.00e-04_offR=1.30e-03.csv')


for key in selected_OPs:
    OP = OPs[key]
    if 'HA' in OP['name']: #in case of humid air --> not yet implemented in PDSim
        Y = HAPropsSI('Y', 'T', OP['T1'], 'P', OP['p1'], 'R', 0.9)
        fluid = 'nitrogen[{0:.4f}]&water[{1:.4f}]'.format(1-Y, Y)
        OP.update(dict(fluid=fluid))
    p1 = OP['p1']/1000
    p2 = OP['p2']/1000
    inletState = State.State(OP['fluid'],{'T':OP['T1'],'P':p1})
    outletState = State.State(OP['fluid'],{'T':OP['T2'],'P':p2})
    if True:
        #delta_S = 1e-4 
    #for delta_S in [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4]:
    #for delta_S in [4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4]:
    #for delta_S in [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4, 15e-4, 20e-4, 25e-4]:
    # for delta_S in [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]:
    # for delta_S in [6e-4, 7e-4, 8e-4, 9e-4, 10e-4]:
        filename='Screw_'+OP['name']+'_kaltSpalt_angepasste_flowcoefficients'
        #filename='Screw_'+OP['name']+'_warmSpalt'
        screw1 = ScrewSpindle(num_lobes=2)

        screw1.set_base_geomdata(BaseGeomDataFilePath = GeomDataFilePath, V_dis_plenum=10, V_suc_plenum=10)
        screw1.set_operation_data(inletState, outletState, OP['n'])
        screw1.auto_add_CVs()
        screw1.auto_add_suction_discharge_tubes()

        #filename+='_leak' + '_deltaS_{0:.3e}'.format(delta_S)
        #screw1.set_leakage_geomdata(LeakGeomDataFilePath = GeomDataFilePath, delta_S=delta_S)
        screw1.set_leakage_geomdata(LeakGeomDataFilePath = LeakDataFilePath)
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
            # HDF5filename=filename+'_'+str_now+'.h5'
            HDF5filename=filename+'.h5'
    )