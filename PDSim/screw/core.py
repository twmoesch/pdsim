
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import splrep, splev, sproot
from scipy import optimize

from PDSim.core.containers import ControlVolume, Tube
from PDSim.flow.flow import FlowPath
from PDSim.core.core import PDSimCore
from PDSim.flow import flow_models

from PDSim.screw import screw_spindle_geo
from PDSim.screw.screw_spindle_geo import leakage_id
from PDSim.screw._screw import _ScrewSpindle

from CoolProp import State
from math import pi

class ScrewSpindle(PDSimCore, _ScrewSpindle):

    def __init__(self, **kwargs):
        PDSimCore.__init__(self)
        
        ## Define the geometry structure
        self.geo=screw_spindle_geo.geoVals()

        self.geo.num_lobes = kwargs.get('num_lobes', 2)
        self.geo.dtheta_chamb = pi/self.geo.num_lobes
        self.geo.V_suc_plenum = kwargs.get('V_suc_plenum', 1.0)
        self.geo.V_dis_plenum = kwargs.get('V_dis_plenum', 1.0)


    def __getstate__(self):
        """
        A function for preparing class instance for pickling
         
        Combine the dictionaries from the _Scroll base class and the Scroll
        class when pickling
        """
        py_dict = self.__dict__.copy()
        py_dict.update(_ScrewSpindle.__cdict__(self))
        return py_dict

    def __setstate__(self, d):
        """
        A function for unpacking class instance for unpickling
        """
        for k,v in d.items():
            setattr(self,k,v)
    

    def set_base_geomdata(self, BaseGeomDataFilePath:Path = None):
        ''' read volume curve and suction/discharge port areas and converts them to splines'''
        if BaseGeomDataFilePath != None:
            df_geomdata = pd.read_csv(BaseGeomDataFilePath,sep='\t',header=0)
            self.geo.theta_raw = df_geomdata['theta'].to_numpy()
            self.geo.V_raw = df_geomdata['V'].to_numpy()
            tck_V = splrep(df_geomdata['theta'].to_numpy(), df_geomdata['V'].to_numpy(), k=1)
            self.geo.dV_raw = splev(df_geomdata['theta'].to_numpy(),tck_V,der=1)
            self.geo.A_dis_ax_raw = df_geomdata['F_HD_ax'].to_numpy()
            self.geo.A_suc_ax_raw = df_geomdata['F_ND_ax'].to_numpy()
            self.geo.A_suc_rad_raw = df_geomdata['F_ND_rad'].to_numpy()

            #Maximum number compression chambers in existence and resulting min and max crank angles, respectively
            self.geo.num_chambers = int(np.ceil(abs(self.geo.theta_raw[0]) / self.geo.dtheta_chamb) + np.ceil(abs(self.geo.theta_raw[-1]) / self.geo.dtheta_chamb))
            self.geo.theta_min = -np.ceil(abs(self.geo.theta_raw[0]) / self.geo.dtheta_chamb) * self.geo.dtheta_chamb
            self.geo.theta_max = np.ceil(abs(self.geo.theta_raw[-1]) / self.geo.dtheta_chamb) * self.geo.dtheta_chamb

            #Suction and discharge angles
            tck_A_suc = splrep(df_geomdata['theta'].to_numpy(), df_geomdata['F_ND_ax'].to_numpy()+df_geomdata['F_ND_rad'].to_numpy() - 1e-12, k=1)
            tck_A_dis = splrep(df_geomdata['theta'].to_numpy(), df_geomdata['F_HD_ax'].to_numpy() - 1e-12, k=1)

            self.geo.theta_suc = sproot(tck_A_suc)[1]
            self.geo.theta_dis = sproot(tck_A_dis)[0]

    	    #Suction and discharge (built-in) volume
            self.geo.V_suc = splev(self.geo.theta_suc, tck_V)
            self.geo.V_dis = splev(self.geo.theta_dis, tck_V)

        else:
            raise NotImplementedError()
    
    def set_leakage_geomdata(self, LeakGeomDataFilePath:Path = None):
        if LeakGeomDataFilePath != None:
            df_geomdata = pd.read_csv(LeakGeomDataFilePath,sep='\t',header=0)
            self.geo.A_leak_housing_raw = df_geomdata['FL_5_A'].to_numpy()
            self.geo.A_leak_intermesh_external_raw = df_geomdata['FL_5_B'].to_numpy()
            self.geo.A_leak_radial_raw = df_geomdata['FL_5_C'].to_numpy()
            self.geo.A_leak_blowhole1_raw = df_geomdata['FL_5_D1'].to_numpy()
            self.geo.A_leak_blowhole2_raw  = df_geomdata['FL_5_D2'].to_numpy()
            self.geo.A_leak_intermesh_internal_raw = df_geomdata['FL_5_E'].to_numpy()
    
    def set_inj_geomdata(self, InjGeomDataFilePath:Path = None):
        if InjGeomDataFilePath != None:
            df_geomdata = pd.read_csv(InjGeomDataFilePath,sep='\t',header=0)
            self.geo.A_inj_raw = np.zeros(df_geomdata['theta'].to_numpy().size)
            i=0
            for col in df_geomdata.columns:
                i+=1
                if 'inj' in str(col):
                    self.geo.A_inj_raw[i] = df_geomdata[col].to_numpy()
            #self.geo.num_inj_tubes = i
    
    # def set_hx_geomdata(self, HXGeomDataFilePath:Path = None):
    #     return None

    def V_SC(self, theta):
        '''VdV function for suction plenum'''
        return self.geo.V_suc_plenum, 0
    
    def V_DC(self, theta):
        '''VdV function for discharge plenum'''
        return self.geo.V_dis_plenum, 0

    def A_suc(self, ichamb):
        '''simple function factory for suction area for each chamber'''
        def _A_suc(self, theta):
            return screw_spindle_geo.area_suction(theta, self.geo, ichamb)
        return _A_suc

    def A_dis(self, ichamb):
        '''simple function factory for discharge area for each chamber'''
        def _A_dis(self, theta):
            return screw_spindle_geo.area_discharge(theta, self.geo, ichamb)
        return _A_dis

    def A_inj(self, ichamb):
        '''simple function factory for injection area for each chamber'''
        def _A_inj(self, theta):
            return screw_spindle_geo.area_injection(theta, self.geo, ichamb)
        return _A_inj

    def VdV(self, ichamb):
        '''simple function factory for volume for each chamber'''
        def _VdV(self, theta):
            return screw_spindle_geo.VdV(theta, self.geo, ichamb)
        return _VdV

    def MdotFcn_leakage(self, ichamb, leakage_id):
        '''simple function factory for leakage mass flow'''
        def _MdotFcn_leakage(self, FP:FlowPath):
            return self.Leakage(FP, ichamb, leakage_id)
        return _MdotFcn_leakage
    
    def MdotFcn_injection(self, ichamb, upstream_key:str='INJ'):
        '''simple function factory for injection mass flow'''
        def _MdotFcn_injection(self, FP:FlowPath):
            return self.Injection(FP, ichamb, upstream_key)
        return _MdotFcn_injection

    def auto_add_CVs(self, inletState:State, outletState:State):
        """
        Adds all the control volumes for the screw spindle compressor.
        
        Parameters
        ----------
        inletState
            A :class:`State <CoolProp.State.State>` instance for the inlet to the scroll set.  Can be approximate
        outletState
            A :class:`State <CoolProp.State.State>` instance for the outlet to the scroll set.  Can be approximate
            
        Notes
        -----
        Uses the indices of 
        
        ============= ===================================================================
        CV            Description
        ============= ===================================================================
        ``sc``        Suction chamber (plenum)
        ``dc``        Discharge chamber (plenum)
        ``ci``        The i-th working chamber (i=1 for the chamber nearest the suction chamber)
        ============= ===================================================================
        """

        

        #Add suction and discharge plenum
        self.add_CV(ControlVolume(key='sc',initialState=inletState.copy(),
                VdVFcn=self.V_SC,becomes=['sc','c1']))

        self.add_CV(ControlVolume(key='dc',initialState=outletState.copy(),
                VdVFcn=self.V_DC,becomes='dc'))

        #Add working chambers (automatically recognizing chambers opened to suction / discharge side)
        for ichamb in range(self.geo.num_chambers) + 1:
            key = 'c'+str(ichamb); key_becomes = 'c'+str(ichamb + 1)

            # A_suc1 = screw_spindle_geo.area_suction(0, self.geo, ichamb)
            # A_suc2 = screw_spindle_geo.area_suction(self.geo.dtheta_chamb, self.geo, ichamb)
            # A_dis1 = screw_spindle_geo.area_discharge(0, self.geo, ichamb)
            # A_dis2 = screw_spindle_geo.area_discharge(self.geo.dtheta_chamb, self.geo, ichamb)
            A_suc1 = self.A_suc(ichamb)(0)
            A_suc2 = self.A_suc(ichamb)(self.geo.dtheta_chamb)
            A_dis1 = self.A_dis(ichamb)(0)
            A_dis2 = self.A_dis(ichamb)(self.geo.dtheta_chamb)

            if A_suc1!=0 or A_suc2!=0: #suction working chambers
                initial_state = inletState.copy()
            elif A_dis1!=0 or A_dis2!=0: #discharge working chambers
                initial_state = outletState.copy()
            else: # compression chambers 
                s1 = inletState.s
                rho1 = inletState.rho
                V1 = self.geo.V_suc
                V2 = self.VdV(ichamb)(0)

                rho2 = rho1 * V1 / V2
                temp = inletState.copy()
                def resid(T):
                    temp.update(dict(T=T, D=rho2))
                    return temp.s-s1
                optimize.root_scalar(resid, inletState.T)
                # Temp has now been updated
                initial_state=temp.copy()

            self.add_CV(ControlVolume(key=key,
                                      initialState=initial_state.copy(),
                                      VdVFcn=self.VdV(ichamb),
                                      #VdVFcn_kwargs={'alpha':alpha},
                                      #discharge_becomes=disc_becomes_c1,
                                      becomes=key_becomes))
        return None
    
    def auto_add_suction_discharge_tubes(self, **kwargs):

        # suction_is_closed = kwargs.get('suction_is_closed', False)

        # add inlet tube #TODO nochmal auf tatsächliche Geometrie anpassen
        self.add_tube(Tube(
            key1='inlet.1',
            key2='inlet.2',
            L=kwargs.get('L_suc_tube', 0.1),
            ID=kwargs.get('D_suc_tube', 0.5),
            mdot=self.mdot_guess,
            State1=self.inletState.copy(),
            fixed=1,
            TubeFcn=self.TubeCode,
            ))

        # add outlet tube #TODO nochmal auf tatsächliche Geometrie anpassen
        self.add_tube(Tube(
            key1='outlet.1',
            key2='outlet.2',
            L=kwargs.get('L_dis_tube', 0.1),
            ID=kwargs.get('D_dis_tube', 0.5),
            mdot=self.mdot_guess,
            State2=self.outletState.copy(),
            fixed=2,
            TubeFcn=self.TubeCode,
            ))
        return None

    def auto_add_leakage(self):
        for ichamb in range(self.geo.num_chambers) + 1:
            if ichamb>=3:
                self.add_flow(FlowPath(key1=ichamb-2,
                                   key2=ichamb,
                                   MdotFcn=self.MdotFcn_leakage(ichamb, leakage_id.HOUSING),
                                   )
                          )
            if ichamb>=2:
                self.add_flow(FlowPath(key1=ichamb-1,
                                   key2=ichamb,
                                   MdotFcn=self.MdotFcn_leakage(ichamb, leakage_id.BLOWHOLE),
                                   )
                          )
            if ichamb>=4:
                self.add_flow(FlowPath(key1=ichamb-3,
                                   key2=ichamb,
                                   MdotFcn=self.MdotFcn_leakage(ichamb, leakage_id.INTERMESH_EXT),
                                   )
                          )    
            if ichamb>=5:
                self.add_flow(FlowPath(key1=ichamb-4,
                                   key2=ichamb,
                                   MdotFcn=self.MdotFcn_leakage(ichamb, leakage_id.INTERMESH_INT),
                                   )
                          )
                self.add_flow(FlowPath(key1=ichamb-4,
                                   key2=ichamb,
                                   MdotFcn=self.MdotFcn_leakage(ichamb, leakage_id.RADIAL),
                                   )
                          )    
        return None
    
    def auto_add_injection(self, injState:State, **kwargs):

        D_inj = kwargs.get('D_inj', 0.01 * np.ones(self.geo.num_inj_tubes))
        L_inj = kwargs.get('L_inj', 0.05 * np.ones(self.geo.num_inj_tubes))


        for itube in range(self.geo.num_inj_tubes) + 1:
            self.add_tube(
                Tube(
                    key1='INJtube{}.1'.format(itube),
                    key2='INJtube{}.2'.format(itube),
                    L=L_inj[itube],
                    ID=D_inj[itube],
                    mdot=1e-6,
                    State1=injState.copy(),
                    fixed=1,
                    TubeFcn=self.TubeCode,
                    )
                ) 
            for ichamb in range(self.geo.num_chambers) + 1:
                for theta in np.linspace(0, self.geo.dtheta_chamb, 100):
                    if self.A_inj(ichamb)(theta) > 0:
                        self.add_flow(FlowPath(
                        key1 = 'INJtube{}.2'.format(itube), key2 = 'C{}'.format(ichamb), 
                        MdotFcn = self.MdotFcn_injection(ichamb, upstream_key='INJtube{}.2'.format(itube)),
                        #MdotFcn_kwargs=dict(itube = itube, ichamb = ichamb,)
                        )) 
                        break
        return None

# TODO: continue with:
# - heat transfer (area, coefficient, calcHT, callback)
# - step callback (discard small chambers to discharge)
# - lump_energy_balance_callback
# - attach_HDF5_annotations
# - TubeCode
# - Compressor_solve
# - leakage coefficient c0 handling
        

               

