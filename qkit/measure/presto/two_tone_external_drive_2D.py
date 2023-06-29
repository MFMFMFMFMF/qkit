# -*- coding: utf-8 -*-
"""
Simple frequency sweep using the Lockin mode.
"""
import h5py
import numpy as np
import ast
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import lockin
from presto.utils import ProgressBar

from qkit.measure.presto._base import Base 

from matplotlib import *
from scipy.ndimage import gaussian_filter
import time
DAC_CURRENT = 32_000  # uA
# CONVERTER_CONFIGURATION_0 = {
    # "adc_mode": AdcMode.Mixed,
    # "adc_fsample": AdcFSample.G4,
    # "dac_mode": DacMode.Mixed04,
    # "dac_fsample": DacFSample.G8,
# }
config_0 = {
    "adc_mode": [1,1,1,1],
    "adc_fsample": [2,2,2,2],
    "dac_mode": [2,2,2,2],
    "dac_fsample": [4,4,4,4]}



class two_tone(Base):
    '''
    ###### Define the experiment as such :
    experiment = presto_sweep.Sweep({   'input_port' : 1,
                                    'output_port' : 1,
                                    'df' : 0.2e6,
                                    'freq_center' : 7.32e9,
                                    'freq_span' : 0.06e9,
                                    'amp' : 0.005,
                                    'dither' : True,
                                    'num_skip' : 0 ,
                                    'num_averages' : 3000}
                                    )
    #####  Define the data folder and launch it using : 
    experiment.experiment_name = 'filename.h5'
    save_filename = experiment.run(presto_address)
    '''
    def __init__(
        self,dict_param = {}
     ) -> None:
            

        self._default_vals = {
            'freq' : 6e9,
            'num_averages' : 10,
            'amp' : 0.1,
            'output_port' : 1,
            'input_port': 1,
            'dither': True,
            'experiment_name': "0.h5",
            'resp_arr':[None],
            'drive_frequ_arr' : [None],
            'bias_arr' : [None],
            '_bias_func': 0,
            '_frequ_drive_func' : 0 }
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
        with lockin.Lockin(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            force_config =True,
            **CONVERTER_CONFIGURATION,
        ) as lck:
            assert lck.hardware is not None

            self._bias_func(self.bias_arr[0])
            lck.hardware.set_adc_attenuation(self.input_port, 27.0)
            lck.hardware.set_dac_current(self.output_port, DAC_CURRENT)
            lck.hardware.set_inv_sinc(self.output_port, 0)

            # tune frequencies
            n_bias = len(self.bias_arr)
            self.resp_arr = np.zeros((n_bias,len(self.drive_frequ_arr)), np.complex128)
            lck.hardware.configure_mixer(
                freq=self.freq,
                in_ports=self.input_port,
                out_ports=self.output_port,
            )
            #lck.set_df(self.df)
            og = lck.add_output_group(self.output_port, 1)
            og.set_frequencies(0.0)
            og.set_amplitudes(self.amp)
            og.set_phases(0.0, 0.0)

            lck.set_dither(self.dither, self.output_port)
            ig = lck.add_input_group(self.input_port, 1)
            ig.set_frequencies(0.0)

            lck.apply_settings()

            t0 = time.time()
            for jj in range(len(self.bias_arr)):
                if jj>0:
                    self._bias_func(self.bias_arr[jj])
                
                lck.hardware.sleep(0.2, False)
                dt = time.time() - t0
                print(f"\r bias {round(self.bias_arr[jj],3)} , iteration {jj+1} over {n_bias} / should end in {round((dt*(n_bias-jj)/(jj+1))/60,3)} min ", end="")

                for ii in range(len(self.drive_frequ_arr)):
                    self._frequ_drive_func(self.drive_frequ_arr[ii])
                    lck.hardware.sleep(1e-3, False)
                    
                    _d = lck.get_pixels(self.num_averages, quiet=True)
                    data_i = _d[self.input_port][1][:, 0]
                    data_q = _d[self.input_port][2][:, 0]
                    data = data_i.real + 1j * data_q.real  # using zero IF

                    self.resp_arr[jj,ii] = np.mean(data[ :])

                    #pb.increment()

                #pb.done()

            # Mute outputs at the end of the sweep
            og.set_amplitudes(0.0)
            lck.apply_settings()
        
        return self.save(self.experiment_name)

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)
    
    
    def plot_load(self,name,ax,phaseroll = 780):
        expe = self.load(name)
        f = expe.freq_arr
        out = expe.resp_arr
        Sij = gaussian_filter(((out)/expe.amp )*np.exp(1j*f*phaseroll*1e-9),0.8)
        ax[0].plot(f/1e9,20*np.log10(abs(Sij)))
        ax[1].plot(f/1e9,np.unwrap(np.angle(Sij)))
        ax[0].set_xlabel('frequency (GHz)')
        ax[0].set_ylabel('|Sij| (dB)')
        ax[1].set_ylabel('angle(Sij) (rad)')
        ax[1].set_xlabel('frequency (GHz)')
        ax[2].plot(np.real(Sij),np.imag(Sij))