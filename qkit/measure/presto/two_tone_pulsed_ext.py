# -*- coding: utf-8 -*-
"""
Two-tone spectroscopy with Pulsed mode: sweep of pump frequency, with fixed pump power and fixed probe.
"""
import ast

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


class TwoTonePulsed(Base):
    def __init__(
        self,dict_param = {}
     ) -> None:
    
        
        self._default_vals = {
            'readout_freq' : 6e9,
            'num_averages' : 10,
            'readout_amp':0.1,
            'readout_duration':200e-9,
            'control_duration' : 300e-9,
            'sample_duration' : 200e-9,
            'sample_port' : 1,
            'control_digital_port':3,
            'readout_port': 1,
            'readout_sample_delay':200e-6,
            'wait_delay' : 50e-6,
            'digital_delay' : 200e-9,
            'drag': 0,
            'experiment_name': "0.h5",
            'control_freq_arr' : [None],
            '_control_freq_func' : 0,
            'store_arr':[None],
            't_arr' : [None],}
            
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
        print_time:bool = True,
        print_save:bool = True,
    ) -> str:
        self.settings  = self.get_instr_dict()
        CONVERTER_CONFIGURATION = self.create_converter_config(self.converter_config)
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **CONVERTER_CONFIGURATION,
        ) as pls:
            assert pls.hardware is not None

            pls.hardware.set_adc_attenuation(self.sample_port, 27.0)
            pls.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.configure_mixer(
                freq=self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
                sync=True,  # sync in next call
            )

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for frequencies
            pls.setup_freq_lut(
                output_ports=self.readout_port,
                group=0,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )
           

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(
                output_ports=self.readout_port,
                group=0,
                scales=self.readout_amp,
            )

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            readout_pulse = pls.setup_long_drive(
                output_port=self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1.0,
                amplitude_q=1.0,
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
            # Control pulse
            
            for ii in range(len(self.control_freq_arr)):
                self._control_freq_func(self.control_freq_arr[ii])
                
                pls.output_digital_marker(T, self.control_duration, self.control_digital_port)
                # Readout pulse starts right after control pulse
                T += self.control_duration + self.digital_delay
                pls.reset_phase(T, self.readout_port)
                
                pls.output_pulse(T, readout_pulse)
                # Sampling window
                pls.store(T + self.readout_sample_delay)
                # Move to next Rabi amplitude
                T += self.readout_duration
                # Wait for decay
                T += self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************
            # repeat the whole sequence `rabi_n` times
            # then average `num_averages` times
            pls.run(
                period=T,
                repeat_count=1,
                num_averages=self.num_averages,
                print_time=print_time,
            )
            self.t_arr, self.store_arr = pls.get_store_data()


        return self.save(self.experiment_name,print_save=print_save)

    def save(self, save_filename: str = None,print_save:bool = True) -> str:
        return super().save(__file__, save_filename=save_filename,print_save = print_save)
