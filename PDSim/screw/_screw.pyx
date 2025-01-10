from PDSim.flow import flow_models
cimport PDSim.flow.flow_models as flow_models

from PDSim.screw import screw_spindle_geo
cimport PDSim.screw.screw_spindle_geo as screw_spindle_geo


cdef class _ScrewSpindle(object):
    
    cpdef dict __cdict__(self):
        return dict(theta = self.theta, 
                    geo = self.geo,
                    HTC = self.HTC,
                    inc_injection = self.incl_injection,
                    incl_leakage = self.incl_leakage)

    cpdef double Suction(self, FlowPath FP, int ichamb):
        """
        Calculate the suction mass flow rate

        Parameters
        ----------
        FP : FlowPath
        ichamb : int
            number of chamber that is connected to the suction plenum
        
        """
        FP.A = screw_spindle_geo.area_suction(self.theta, self.geo, ichamb)
        try:
            return flow_models.IsentropicNozzle(FP.A,FP.State_up,FP.State_down)
        except ZeroDivisionError:
            return 0.0

    cpdef double Discharge(self, FlowPath FP, int ichamb):
        """
        Calculate the discharge mass flow rate
        
        Parameters
        ----------
        FP : FlowPath
        ichamb : int
            number of chamber that is connected to the discharge plenum
        """

        FP.A = screw_spindle_geo.area_discharge(self.theta, self.geo, ichamb)
        try:
            return flow_models.IsentropicNozzle(FP.A,FP.State_up,FP.State_down)
        except ZeroDivisionError:
            return 0.0
    
    cpdef double Leakage(self, FlowPath FP, int ichamb, leak_id id):
        """
        Calculate the leakage mass flow rate from a specific chamber

        Parameters
        ----------
        FP : FlowPath
        ichamb : int
            number of chamber that the leakage originates from
        id : leak_id
            leakage identifier (enum)       
        """

        FP.A = screw_spindle_geo.area_leak(self.theta, self.geo, ichamb, id)
        try:
            return flow_models.IsentropicNozzle(FP.A,FP.State_up,FP.State_down)
        except ZeroDivisionError:
            return 0.0
    
    cpdef double Injection(self, FlowPath FP, int ichamb, str upstream_key):
        """
        Calculate the injection mass flow rate into a specific chamber (w/o backflow)

        Parameters
        ----------
        FP : FlowPath
        ichamb : int
            number of chamber that the leakage originates from
        upstream_key: string
            Key for the side of the flow path that is considered to be "upstream"      
        """

        if FP.key_up == upstream_key:
            FP.A = screw_spindle_geo.area_injection(self.theta, self.geo, ichamb)
            try:
                return flow_models.LiquidNozzleFlow(FP.A,FP.State_up,FP.State_down, 1.0, 1e-10)
            except ZeroDivisionError:
                return 0.0
        else:
            return 0.0