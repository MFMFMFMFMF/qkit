# -*- coding: utf-8 -*-
"""
Measure Rabi oscillation by changing the amplitude of the control pulse.

The control pulse has a sin^2 envelope, while the readout pulse is square.
"""
import ast
import math
from typing import List, Tuple

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import rotate_opt, sin2

from qkit.measure.presto._base import Base

DAC_CURRENT = 32_000  # uA
config_0 = {
    "adc_mode": [1,1,1,1],
    "adc_fsample": [2,2,2,2],
    "dac_mode": [2,2,2,2],
    "dac_fsample": [2,2,2,2]}


IDX_LOW = 1_500
IDX_HIGH = 2_000


class SimplePulse_Digital(Base):

    def __init__(
        self,dict_param = {}
     ) -> None:
        
        self._default_vals = {
            'digital_delay':1e-9,
            'readout_freq' : 6e9,
            'num_averages' : 10,
            'readout_duration':200e-9,
            'sample_duration' : 200e-9,
            'sample_port' : 3,
            'digital_port': 1,
            'readout_sample_delay':200e-6,
            'wait_delay' : 50e-6,
            'drag': 0,
            'experiment_name': "0.h5",
            'store_arr':[None],
            't_arr' : [None]}
        
        
        for key,value in dict_param.items():
            if key  not in self._default_vals :
                print(key ,'is unnecessary')
        
        for key, value in self._default_vals.items():
            setattr(self, key, dict_param.get(key, value))
        
        self.converter_config = config_0

    def run(
        self,
        presto_address: str,
        presto_port: int = None,
        ext_ref_clk: bool = False,
    ) -> str:
        self.settings  = self.get_instr_dict()
        CONVERTER_CONFIGURATION = self.create_converter_config(self.converter_config)
        # Instantiate interface class
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **CONVERTER_CONFIGURATION,
        ) as pls:
            assert pls.hardware is not None

            pls.hardware.set_adc_attenuation(self.sample_port, 27.0)
            pls.hardware.configure_mixer(
                freq=self.readout_freq,
                in_ports=self.sample_port,
                sync=True,  # sync in next call
            )

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for frequencies


            # Setup lookup tables for amplitudes

            

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here



            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            # Readout
            #pls.reset_phase(T, self.readout_port)
            
            pls.output_digital_marker(T+self.digital_delay, self.readout_duration, self.digital_port)
            pls.store(T + self.readout_sample_delay)
            T += self.readout_duration
            # Wait for decay
            T += self.wait_delay


            # **************************
            # *** Run the experiment ***
            # **************************
            # repeat the whole sequence `nr_amps` times
            # then average `num_averages` times

            pls.run(
                period=T,
                repeat_count=1,
                num_averages=self.num_averages,
                print_time=True,
            )
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save(self.experiment_name)

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)
