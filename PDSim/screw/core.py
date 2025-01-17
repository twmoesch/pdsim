
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import splrep, splev, sproot, PPoly, BSpline
from scipy import optimize

from PDSim.core.containers import ControlVolume, Tube
from PDSim.flow.flow import FlowPath
from PDSim.core.core import PDSimCore
from PDSim.flow import flow_models
from PDSim.misc.datatypes import arraym

from PDSim.screw import screw_spindle_geo
from PDSim.screw._screw import _ScrewSpindle

from CoolProp import State
from math import pi

class ScrewSpindle(PDSimCore, _ScrewSpindle):

    def __init__(self, **kwargs):
        PDSimCore.__init__(self)
        
        ## Define the geometry structure
        # self.geo=screw_spindle_geo.geoVals()
        self.geo=screw_spindle_geo.geoVals(num_lobes=kwargs.get('num_lobes', 2))

        # self.geo.num_lobes = kwargs.get('num_lobes', 2)
        self.geo.dtheta_chamb = pi/self.geo.num_lobes
        # self.geo.V_suc_plenum = kwargs.get('V_suc_plenum', 1.0)
        # self.geo.V_dis_plenum = kwargs.get('V_dis_plenum', 1.0)

        self.__incl_injection__ = False
        self.__incl_leakage__ = False


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
    

    def set_base_geomdata(self, BaseGeomDataFilePath:Path = None, **kwargs):
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
            tck_A_suc = splrep(df_geomdata['theta'].to_numpy(), df_geomdata['F_ND_ax'].to_numpy()+df_geomdata['F_ND_rad'].to_numpy() - 1e-12, k=3)
            tck_A_dis = splrep(df_geomdata['theta'].to_numpy(), df_geomdata['F_HD_ax'].to_numpy() - 1e-12, k=3)

            # self.geo.theta_suc = sproot(tck_A_suc)[1]
            # self.geo.theta_dis = sproot(tck_A_dis)[0]

            self.geo.theta_suc = PPoly.from_spline(tck_A_suc).roots()[1]
            self.geo.theta_dis = PPoly.from_spline(tck_A_dis).roots()[0]

    	    #Suction and discharge (built-in) volume
            self.geo.V_suc = splev(self.geo.theta_suc, tck_V)
            self.geo.V_dis = splev(self.geo.theta_dis, tck_V)
            self.Vdisp = self.geo.V_suc * self.geo.num_lobes * 2

            self.geo.V_suc_plenum = kwargs.get('V_suc_plenum', self.geo.V_suc * 10)
            self.geo.V_dis_plenum = kwargs.get('V_dis_plenum', self.geo.V_dis * 10)
            self.geo.V_nan = self.geo.V_suc * 1e-3
            

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
            self.geo.theta_inj_raw = df_geomdata['theta'].to_numpy()
            self.geo.A_inj_raw = np.zeros(df_geomdata['theta'].to_numpy().size)
            i=0
            A_inj = np.zeros(df_geomdata['theta'].to_numpy().size)
            for col in df_geomdata.columns:
                if 'inj' in str(col):
                    i+=1
                    A_inj += df_geomdata[col].to_numpy()
                    #self.geo.A_inj_raw[i] = df_geomdata[col].to_numpy()
            self.geo.A_inj_raw = A_inj
            self.geo.num_inj_tubes = i
    
    def set_operation_data(self, inletState:State, outletState:State, n:float=12000/60):
        self.inletState = inletState
        self.outletState = outletState
        self.n = n
        self.omega = self.n * 2 * pi
        return None

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
        def _A_suc(theta):
            return screw_spindle_geo.area_suction(theta, self.geo, ichamb)
        return _A_suc

    def A_dis(self, ichamb):
        '''simple function factory for discharge area for each chamber'''
        def _A_dis(theta):
            return screw_spindle_geo.area_discharge(theta, self.geo, ichamb)
        return _A_dis

    def A_inj(self, ichamb):
        '''simple function factory for injection area for each chamber'''
        def _A_inj(theta):
            return screw_spindle_geo.area_injection(theta, self.geo, ichamb)
        return _A_inj

    def VdV(self, ichamb):
        '''simple function factory for volume for each chamber'''
        def _VdV(theta):
            return screw_spindle_geo.VdV(theta, self.geo, ichamb)[0:2]
        return _VdV

    def MdotFcn_leakage(self, ichamb, leakage_id):
        '''simple function factory for leakage mass flow'''
        def _MdotFcn_leakage(FP:FlowPath):
            return self.Leakage(FP, ichamb, leakage_id)
        return _MdotFcn_leakage
    
    def MdotFcn_injection(self, ichamb, upstream_key:str='INJ'):
        '''simple function factory for injection mass flow'''
        def _MdotFcn_injection(FP:FlowPath):
            return self.Injection(FP, ichamb, upstream_key)
        return _MdotFcn_injection

    def auto_add_CVs(self):
        """
        Adds all the control volumes for the screw spindle compressor.
        
        Parameters
        ----------
        inletState
            A :class:`State <CoolProp.State.State>` instance for the inlet to the screw compressor.  Can be approximate
        outletState
            A :class:`State <CoolProp.State.State>` instance for the outlet to the screw compressor.  Can be approximate
            
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
        self.add_CV(ControlVolume(key='sc',initialState=self.inletState.copy(),
                VdVFcn=self.V_SC,becomes=['sc','c1']))

        self.add_CV(ControlVolume(key='dc',initialState=self.outletState.copy(),
                VdVFcn=self.V_DC,becomes='dc'))

        #Add working chambers (automatically recognizing chambers opened to suction / discharge side)
        is_suction = True; is_discharge = False
        for ichamb in range(1, self.geo.num_chambers + 1, 1):

            key = 'c'+str(ichamb); key_becomes = 'c'+str(ichamb + 1)
            
                
            # A_suc1 = screw_spindle_geo.area_suction(0, self.geo, ichamb)
            # A_suc2 = screw_spindle_geo.area_suction(self.geo.dtheta_chamb, self.geo, ichamb)
            # A_dis1 = screw_spindle_geo.area_discharge(0, self.geo, ichamb)
            # A_dis2 = screw_spindle_geo.area_discharge(self.geo.dtheta_chamb, self.geo, ichamb)
            A_suc1 = self.A_suc(ichamb)(theta=0)
            A_suc2 = self.A_suc(ichamb)(theta=self.geo.dtheta_chamb)
            A_dis1 = self.A_dis(ichamb)(theta=0)
            A_dis2 = self.A_dis(ichamb)(theta=self.geo.dtheta_chamb)

            if A_suc1!=0 or A_suc2!=0: #suction working chambers
                initial_state = self.inletState.copy()
                is_suction = True
            elif A_dis1!=0 or A_dis2!=0 or is_discharge: #discharge working chambers
                initial_state = self.outletState.copy()
                is_discharge = True
            else: # compression chambers
                is_suction = False 
                s1 = self.inletState.s
                rho1 = self.inletState.rho
                V1 = self.geo.V_suc
                #VdV2 = self.VdV(ichamb)(0)
                V2, dV2 = self.VdV(ichamb)(0)

                if V2 <= self.geo.V_nan:
                    initial_state = self.outletState.copy()
                else:
                    rho2 = rho1 * V1 / V2
                    temp = self.inletState.copy()
                    def resid(T):
                        temp.update(dict(T=T, D=rho2))
                        return temp.s-s1
                    optimize.root_scalar(resid, x0=self.inletState.T)
                    # Temp has now been updated
                    initial_state=temp.copy()

            self.add_CV(ControlVolume(key=key,
                                      initialState=initial_state.copy(),
                                      VdVFcn=self.VdV(ichamb),
                                      #VdVFcn_kwargs={'alpha':alpha},
                                      #discharge_becomes=disc_becomes_c1,
                                      becomes=key_becomes))

            if is_suction:
                self.add_flow(FlowPath(key1='sc',key2=key,MdotFcn=self.Suction,MdotFcn_kwargs=dict(ichamb=ichamb)))
            if is_discharge:
                self.add_flow(FlowPath(key1=key,key2='dc',MdotFcn=self.Discharge,MdotFcn_kwargs=dict(ichamb=ichamb)))
        return None
    
    def auto_add_suction_discharge_tubes(self, **kwargs):

        L_suc = kwargs.get('L_suc_tube', 0.1)
        D_suc = kwargs.get('D_suc_tube', 0.5)
        L_dis = kwargs.get('L_dis_tube', 0.1)
        D_dis = kwargs.get('D_dis_tube', 0.5)

        self.mdot_guess = self.inletState.rho*self.Vdisp*self.omega/(2*pi)

        # add inlet tube #TODO nochmal auf tatsächliche Geometrie anpassen
        self.add_tube(Tube(
            key1='inlet.1',
            key2='inlet.2',
            L=L_suc,
            ID=D_suc,
            mdot=self.mdot_guess,
            State1=self.inletState.copy(),
            fixed=1,
            TubeFcn=self.TubeCode,
            ))

        # add outlet tube #TODO nochmal auf tatsächliche Geometrie anpassen
        self.add_tube(Tube(
            key1='outlet.1',
            key2='outlet.2',
            L=L_dis,
            ID=D_dis,
            mdot=self.mdot_guess,
            State2=self.outletState.copy(),
            fixed=2,
            TubeFcn=self.TubeCode,
            ))
        
        self.add_flow(FlowPath(key1='inlet.2',key2='sc',MdotFcn=self.SimpleFlow, \
                               MdotFcn_kwargs=dict(A=pi/4*D_suc**2)))
        self.add_flow(FlowPath(key1='dc',key2='outlet.1',MdotFcn=self.SimpleFlow, \
                               MdotFcn_kwargs=dict(A=pi/4*D_dis**2)))
        
        

        return None

    def auto_add_leakage(self):
        self.__incl_leakage__ = True
        for ichamb in range(1, self.geo.num_chambers + 1, 1) :
            if ichamb>=3:
                self.add_flow(FlowPath(key1='c' + str(ichamb-2),
                                   key2='c' + str(ichamb),
                                   # MdotFcn=self.MdotFcn_leakage(ichamb, screw_spindle_geo.HOUSING),
                                   MdotFcn=self.MdotFcn_leakage(ichamb, 0),
                                   )
                          )
            if ichamb>=2:
                self.add_flow(FlowPath(key1='c' + str(ichamb-1),
                                   key2= 'c' + str(ichamb),
                                   #MdotFcn=self.MdotFcn_leakage(ichamb, screw_spindle_geo.BLOWHOLE),
                                   MdotFcn=self.MdotFcn_leakage(ichamb, 4),
                                   )
                          )
            if ichamb>=4:
                self.add_flow(FlowPath(key1='c' + str(ichamb-3),
                                   key2='c' + str(ichamb),
                                   #MdotFcn=self.MdotFcn_leakage(ichamb, screw_spindle_geo.INTERMESH_EXT),
                                   MdotFcn=self.MdotFcn_leakage(ichamb, 3),
                                   )
                          )    
            if ichamb>=5:
                self.add_flow(FlowPath(key1='c' + str(ichamb-4),
                                   key2='c' + str(ichamb),
                                   #MdotFcn=self.MdotFcn_leakage(ichamb, screw_spindle_geo.INTERMESH_INT),
                                   MdotFcn=self.MdotFcn_leakage(ichamb, 2),
                                   )
                          )
                self.add_flow(FlowPath(key1='c' + str(ichamb-4),
                                   key2='c' + str(ichamb),
                                   # MdotFcn=self.MdotFcn_leakage(ichamb, screw_spindle_geo.RADIAL),
                                   MdotFcn=self.MdotFcn_leakage(ichamb, 1),
                                   )
                          )    
        return None
    
    def auto_add_injection(self, injState:State, **kwargs):
        self.__incl_injection__ = True
        D_inj = kwargs.get('D_inj', 0.01 * np.ones(self.geo.num_inj_tubes))
        L_inj = kwargs.get('L_inj', 0.05 * np.ones(self.geo.num_inj_tubes))


        for itube in range(self.geo.num_inj_tubes):
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
            for ichamb in range(1, self.geo.num_chambers + 1, 1):
                for theta in np.linspace(0, self.geo.dtheta_chamb, 100):
                    if self.A_inj(ichamb)(theta) > 0:
                        self.add_flow(FlowPath(
                        key1 = 'INJtube{}.2'.format(itube), key2 = 'c{}'.format(ichamb), 
                        MdotFcn = self.MdotFcn_injection(ichamb, upstream_key='INJtube{}.2'.format(itube)),
                        #MdotFcn_kwargs=dict(itube = itube, ichamb = ichamb,)
                        )) 
                        break
        return None

    def TubeCode(self,Tube,**kwargs):
        if abs(self.Tlumps[0])==np.inf or np.isnan(self.Tlumps[0]):
            T_wall = 300
        else:
            T_wall = self.Tlumps[0]
        Tube.Q = flow_models.IsothermalWallTube(Tube.mdot,
                                                Tube.State1,
                                                Tube.State2,
                                                Tube.fixed,
                                                Tube.L,
                                                Tube.ID,
                                                T_wall=T_wall,
                                                Q_add = Tube.Q_add,
                                                alpha = Tube.alpha
                                                )

    def mechanical_losses(self):
        """
        The mechanical losses in kW
        """
        return 1e-6
    
    def ambient_heat_transfer(self):
        """
        The ambient heat transfer for the compressor in kW
        
        Returns a positive value if heat is added to the compressor from the 
        ambient
        """
        return 1e-6


    def heat_transfer_callback(self, theta):
        """
        A callback used by PDSimCore.derivs to calculate the heat transfer
        to the gas in the working chamber.
        
        We return an arraym instance the same length as the number of CV in existence
        
        More code (a proper heat transfer model) could be included here, like 
        in PDSim.recip.core
        """
        return arraym([0.0]*len(self.CVs.exists_keys))
    
    def lump_energy_balance_callback(self):
        """
        A callback used in PDSimCore.solve to do the energy balance on the lump
        
        Note: we neglect heat transfer to the gas in the working chamber
        """

        #Mechanical losses are added to the lump
        self.Wdot_mechanical = self.mechanical_losses() #[kW]
        #Heat transfer between the shell and the ambient
        self.Qamb = self.ambient_heat_transfer() #[kW]
        return self.Wdot_mechanical + self.Qamb

    def step_callback(self,t,h,Itheta):
        """ A callback at each step """ 
        # This gets called at every step, or partial step
        self.theta = t

        V,dV=self.CVs.volumes(t)
        Vdict=dict(zip(self.CVs.exists_keys,V))

        disable = False

        #  Release chamber DC when it gets too small
        for ichamb in [self.geo.num_chambers-2,self.geo.num_chambers-1, self.geo.num_chambers]:
            kam = 'c'+str(ichamb)
            if self.CVs[kam].exists:
                if Vdict[kam]<=self.geo.V_nan*1.05:
                    #Build the volume vector using the old set of control volumes (pre-merge)
                    V,dV=self.CVs.volumes(t)
                    
                    if self.__hasLiquid__==False:
                        # Density
                        rho_kam=self.CVs[kam].State.rho
                        rhoDC=self.CVs['dc'].State.rho
                        # Internal energy
                        u_kam=self.CVs[kam].State.u
                        uDC=self.CVs['dc'].State.u
                        # Temperature
                        T_kam=self.CVs[kam].State.T
                        TDC=self.CVs['dc'].State.T
                        # Volumes
                        Vdict=dict(zip(self.CVs.exists_keys,V))
                        VDC=Vdict['dc']
                        
                        V_merged=Vdict[kam]+VDC
                        m_merged=rho_kam*Vdict[kam]+rhoDC*VDC
                        U_before=u_kam*rho_kam*Vdict[kam]+uDC*rhoDC*VDC
                        rho_merged=m_merged/V_merged
                        #guess the mixed temperature as a volume-weighted average
                        T_merged=(T_kam*Vdict[kam]+TDC*VDC)/V_merged
                        #Must conserve mass and internal energy (instantaneous mixing process)
                        
                        temp = self.CVs['dc'].State.copy()
                        def resid(T):
                            temp.update(dict(T=T,D=rho_merged))
                            return temp.u - U_before/m_merged
                            
                        T_u = optimize.newton(resid, T_merged)
                        
                        self.CVs['dc'].State.update({'T':T_u,'D':rho_merged})
                        U_after=self.CVs['dc'].State.u*self.CVs['dc'].State.rho*V_merged
                        
                        DeltaU=m_merged*(U_before-U_after)
                        if abs(DeltaU)>1e-5:
                            raise ValueError('Internal energy not sufficiently conserved in merging process')
                        
                        self.CVs[kam].exists=False
                        self.CVs['dc'].exists=True
                        
                        self.update_existence()
                        
                        #Re-calculate the CV
                        V,dV=self.CVs.volumes(t)
                        self.T[self.CVs.exists_indices,Itheta] = self.CVs.T
                        self.p[self.CVs.exists_indices,Itheta] = self.CVs.p
                        self.m[self.CVs.exists_indices,Itheta] = arraym(self.CVs.rho)*V
                        self.rho[self.CVs.exists_indices,Itheta] = arraym(self.CVs.rho)
                        
                    else:
                        raise NotImplementedError('no flooding yet')
                    disable=True 
        return disable,h

    def compressor_solve(self, save2file=True, showDiagrams=False, **kwargs):
        fileName=kwargs.get('HDF5filename', 'screw.h5')

        # compressor speed
        self.n = kwargs.get('n', 12000/60) # shaft speed in 1/min
        self.omega = self.n*2*pi
        
        # solver specifications
        self.EulerN = kwargs.get('EulerN',10000) # Steps for Euler Integration (Default: 7000)
        self.HeunN = kwargs.get('HeunN',10000) # Steps for Heun Integration (Default: 7000)
        self.RK45_eps = kwargs.get('RK45_eps',1e-8) # Tolerance for RK45 Integrator (Default: 1e-8)


        # connect all callbacks
        self.connect_callbacks(step_callback=self.step_callback,
                                endcycle_callback=self.endcycle_callback, # Provided by PDSimCore
                                heat_transfer_callback=self.heat_transfer_callback,
                                lumps_energy_balance_callback = self.lump_energy_balance_callback
                                )
        self.solve(
            key_inlet='inlet.1',
            key_outlet='outlet.2',
            solver_method = kwargs.get('solver_method', 'Euler'),
            OneCycle = kwargs.get('OneCycle', False),
            UseNR = False,
            cycle_integrator_options=dict(tmin=0.0,tmax=pi/self.geo.num_lobes),
            plot_every_cycle=kwargs.get('plot_every_cycle', False),
            eps_cycle = kwargs.get('eps_cycle', 0.001),
            eps_energy_balance = 0.01,
            max_number_of_steps = 100000,
            )

        del self.FlowStorage
        for attr, value in self.geo:
            setattr(self, attr, value)
        del self.geo
        try:
            del self.GapFlowLib
        except AttributeError:
            pass
        if save2file:
            from PDSim.misc.hdf5 import HDF5Writer
            h5 = HDF5Writer()
            h5.write_to_file(self, fileName)

        if showDiagrams:
            from PDSim.plot.plots import debug_plots
            debug_plots(self)

        return None

# TODO: continue with:
# - heat transfer (area, coefficient, calcHT, callback)
# - step callback (discard small chambers to discharge)

# - attach_HDF5_annotations
# - TubeCode
# - Compressor_solve
# - leakage coefficient c0 handling

# - update heat_transfer_callback    
# - update lump_energy_balance_callback

        

               

