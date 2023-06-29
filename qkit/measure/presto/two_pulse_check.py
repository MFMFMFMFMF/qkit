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


class SimplePulse(Base):
    '''
    experiment_simple_pulse = presto_basic_pulse_check.SimplePulse({ 'readout_freq' : 7.325e9,
                                        'num_averages' : 500000,
                                        'readout_amp' : 0.02,
                                        'readout_duration' : 500e-9,
                                        'sample_duration' : 900e-9,
                                        'sample_port' : 1,
                                        'readout_port': 1,
                                        'readout_sample_delay' : 200e-9,
                                        'wait_delay' : 50e-6})
    experiment_simple_pulse.experiment_name = 'filename'
    save_filename = experiment_simple_pulse.run(presto_address)
    '''
    def __init__(
        self,dict_param = {}
     ) -> None:
        
        self._default_vals = {
            'output_freq1' : 6e9,
            'output_freq2' : 6e9,
            'num_averages' : 10,
            'amp1':0.1,
            'amp2':0.,
            'phase1':0.,
            'phase2':0.,
            'duration1':200e-9,
            'duration2':200e-9,
            'sample_duration' : 200e-9,
            'sample_port' : 1,
            'output_port1': 1,
            'output_port2': 2,
            'delay2':0,
            'readout_sample_delay':0e-6,
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
        print_time: bool = True,
        print_save : bool = True
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
            pls.hardware.set_adc_attenuation(self.sample_port, 25.0)
            pls.hardware.set_dac_current(self.output_port1, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.output_port1, 0)
            pls.hardware.configure_mixer(
                freq=self.output_freq1,
                in_ports=self.sample_port,
                out_ports=self.output_port1,
                tune =True,
                sync=False, 
            )            
            pls.hardware.set_dac_current(self.output_port2, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.output_port2, 0)
            pls.hardware.configure_mixer(
                freq=self.output_freq2,
                out_ports=self.output_port2,
                tune =True,
                sync=True,  # sync in next call
            )
            
            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for frequencies
            pls.setup_freq_lut(
                output_ports=self.output_port1,
                group=0,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )
            
            pls.setup_freq_lut(
                output_ports=self.output_port2,
                group=0,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )
            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(
                output_ports=self.output_port1,
                group=0,
                scales=self.amp1,
            )
            pls.setup_scale_lut(
                output_ports=self.output_port2,
                group=0,
                scales=self.amp2,
            )

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            pulse1 = pls.setup_long_drive(
                output_port=self.output_port1,
                group=0,
                duration=self.duration1,
                amplitude=1.0*np.exp(1j*self.phase1),
                rise_time=0e-9,
                fall_time=0e-9,
            )
            pulse2 = pls.setup_long_drive(
                output_port=self.output_port2,
                group=0,
                duration=self.duration2,
                amplitude=1.0*np.exp(1j*self.phase2),
                rise_time=0e-9,
                fall_time=0e-9,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            # Readout
            pls.reset_phase(T, self.output_port1)
            pls.reset_phase(T, self.output_port2)
            
            pls.output_pulse(T, pulse1)
            pls.output_pulse(T+self.delay2, pulse2)
            pls.store(T + self.readout_sample_delay)
            T += max(self.duration1,self.duration2)
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
                print_time=print_time,
            )
            self.t_arr, self.store_arr = pls.get_store_data()
            self.store_arr = self.store_arr[0,0,:]

        return self.save(self.experiment_name,print_save=print_save)

    def save(self, save_filename: str = None,print_save:bool = True) -> str:
        return super().save(__file__, save_filename=save_filename,print_save = print_save)
