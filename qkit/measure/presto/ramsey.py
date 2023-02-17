# -*- coding: utf-8 -*-
"""Measure the decoherence time T2 with a Ramsey (non-echo) experiment."""
import ast
from typing import List, Optional

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import format_precision, rotate_opt, sin2

from qkit.measure.presto._base import Base,project


DAC_CURRENT = 32_000  # uA
# CONVERTER_CONFIGURATION = {
    # "adc_mode": AdcMode.Mixed,
    # "adc_fsample": AdcFSample.G4,
    # "dac_mode": [DacMode.Mixed04, DacMode.Mixed02, DacMode.Mixed42, DacMode.Mixed02],
    # "dac_fsample": [DacFSample.G8, DacFSample.G6, DacFSample.G8, DacFSample.G6],
# }
config_0 = {
    "adc_mode": [1,1,1,1],
    "adc_fsample": [2,2,2,2],
    "dac_mode": [2,2,2,2],
    "dac_fsample": [2,2,2,2]}


IDX_LOW = 1_500
IDX_HIGH = 2_000


class Ramsey(Base):
    def __init__(
        self,dict_param = {}
     ) -> None: 
    
        
        self._default_vals = { 
            'readout_freq':6e9,
            'control_freq' : 4e9,
            'readout_amp' : 0.1,
            'readout_phase':0,
            'control_amp':0.5,
            'readout_duration' : 500e-9,
            'control_duration':200e-9,
            'match_duration' : 300e-9,
            'number_of_match':1,
            'delay_arr' : [0],
            'readout_port':1,
            'control_port':3,
            'sample_port':1,
            'wait_delay': 60e-6,
            'readout_match_delay':100e-9,
            'num_averages':100,
            'jpa_params':None,
            'drag':0.0,
            'experiment_name': "0.h5",
            'match_arr': [None]}
            
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
        save: bool = True,
        print_time: bool = True,
        print_save: bool = True,
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
            pls.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)
            pls.hardware.configure_mixer(
                freq=self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
                sync=False,  # sync in next call
            )
            pls.hardware.configure_mixer(
                freq=self.control_freq,
                out_ports=self.control_port,
                sync=True,  # sync here
            )
            if self.jpa_params is not None:
                pls.hardware.set_lmx(
                    self.jpa_params["pump_freq"],
                    self.jpa_params["pump_pwr"],
                    self.jpa_params["pump_port"],
                )
                pls.hardware.set_dc_bias(self.jpa_params["bias"], self.jpa_params["bias_port"])
                pls.hardware.sleep(1.0, False)

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for frequencies
            # we only need to use carrier 1
            pls.setup_freq_lut(
                output_ports=self.readout_port,
                group=0,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )
            pls.setup_freq_lut(
                output_ports=self.control_port,
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
            pls.setup_scale_lut(
                output_ports=self.control_port,
                group=0,
                # scales=control_amp,
                scales=1.0,
            )

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            readout_pulse = pls.setup_long_drive(
                output_port=self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1.0*np.exp(1j*self.readout_phase) ,
                rise_time=0e-9,
                fall_time=0e-9,
            )
            control_ns = int(
                round(self.control_duration * pls.get_fs("dac"))
            )  # number of samples in the control template
            control_envelope = sin2(control_ns, drag=self.drag)
            control_pulse = pls.setup_template(
                output_port=self.control_port,
                group=0,
                template=self.control_amp * control_envelope,
                template_q=self.control_amp * control_envelope if self.drag == 0.0 else None,
                envelope=True,
            )
            # Setup template matching
            ns_match = int(round(self.match_duration * pls.get_fs("adc")))
            templ_i = np.full(ns_match, 1+0j)
            templ_q = np.full(ns_match, 0+1j)
            match_i, match_q = pls.setup_template_matching_pair(
                input_port=self.sample_port,
                template1=templ_i,
                template2=templ_q,
                )
            dict_pulses = {}
            dict_pulses[0] = [match_i, match_q]
            if self.number_of_match>1:
                match_i_2, match_q_2 = pls.setup_template_matching_pair(
                    input_port=self.sample_port,
                    template1=templ_i,
                    template2=templ_q,
                    )
                dict_pulses[1] = [match_i_2, match_q_2]
            
            

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for delay in self.delay_arr:
                pls.reset_phase(T, self.readout_port)
                #pls.output_pulse(T, readout_pulse)
                #pls.store(T + self.readout_sample_delay)
                #for i in range(self.number_of_match):
                #    pls.match(T +self.readout_match_delay+ i*self.match_duration, dict_pulses[i%2])
                # first pi/2 pulse
                pls.reset_phase(T, self.control_port)
                pls.output_pulse(T, control_pulse)
                T += self.control_duration
                # wait 
                T += delay 
                # second pi/2 pulse
                pls.output_pulse(T, control_pulse)
                T += self.control_duration
                # Readout
                pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, readout_pulse)
                
                for i in range(self.number_of_match):
                    pls.match(T +self.readout_match_delay+ i*self.match_duration, dict_pulses[i%2])
                # Wait for decay
                T += self.wait_delay

            if self.jpa_params is not None:
                # adjust period to minimize effect of JPA idler
                idler_freq = self.jpa_params["pump_freq"] - self.readout_freq
                idler_if = abs(idler_freq - self.readout_freq)  # NCO at readout_freq
                idler_period = 1 / idler_if
                T_clk = int(round(T * pls.get_clk_f()))
                idler_period_clk = int(round(idler_period * pls.get_clk_f()))
                # first make T a multiple of idler period
                if T_clk % idler_period_clk > 0:
                    T_clk += idler_period_clk - (T_clk % idler_period_clk)
                # then make it off by one clock cycle
                T_clk += 1
                T = T_clk * pls.get_clk_T()

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(
                period=T,
                repeat_count=1,
                num_averages=self.num_averages,
                print_time=print_time,
            )
            match_arr_0 = np.array(pls.get_template_matching_data(dict_pulses[0]))
            if self.number_of_match>1:
                match_arr_1 = np.array(pls.get_template_matching_data(dict_pulses[1]))
                
                
                N1 = self.number_of_match//2
                N2 = self.number_of_match//2 + self.number_of_match%2
                #print(N2,N1,match_arr_0.shape[1]/N2)
                match_arr_0 = match_arr_0.reshape((2,int(match_arr_0.shape[1]/N2),N2))
                match_arr_1 = match_arr_1.reshape((2,int(match_arr_1.shape[1]/N1),N1))
                self.match_arr = (match_arr_0.mean(2)+match_arr_1.mean(2))/2
                
            else:
                self.match_arr = (match_arr_0)

            if self.jpa_params is not None:
                pls.hardware.set_lmx(0.0, 0.0, self.jpa_params["pump_port"])
                pls.hardware.set_dc_bias(0.0, self.jpa_params["bias_port"])
        if save:
            return self.save(self.experiment_name,print_save=print_save)

    def save(self, save_filename: str = None,print_save:bool = True) -> str:
        return super().save(__file__, save_filename=save_filename,print_save = print_save)