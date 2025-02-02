"""
Author: @Ziad (https://github.com/ZiadHatab)

example of LRM calibration using cpw simulated data from skrf.
"""

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from skrf.media import CPW, Coaxial

# my code
from lrm import LRM

if __name__ == '__main__':
    freq = rf.F(0.1, 100, 999, 'GHz')

    ## error boxes
    # 1.0 mm coaxial media for calibration error boxes
    coax1mm = Coaxial(freq, Dint=0.434e-3, Dout=1.0e-3, sigma=1e8, z0_port=50)

    # Realistic looking error networks.
    X = coax1mm.line(1, 'm', z0=58, name='X')
    Y = coax1mm.line(1.1, 'm', z0=40, name='Y')

    # Realistic looking switch terms
    gamma_f = coax1mm.delay_load(0.2, 21e-3, 'm', z0=60)
    gamma_r = coax1mm.delay_load(0.25, 16e-3, 'm', z0=56)

    ## cal standards
    # CPW media used for DUT and the calibration standards
    cpw = CPW(freq, w=40e-6, s=51e-6, ep_r=12.9, t=5e-6, rho=2e-8)

    # Lengths of the lines used in the calibration, units are in meters
    line_len = 0.3e-3
    line = cpw.line(line_len, 'm')
    
    # update error boxes to have same impedance interface with standards
    X = X**cpw.line(0, 'm')
    Y = cpw.line(0, 'm')**Y
    
    # short standard
    short = cpw.delay_short(10e-6, 'm')
    short = rf.two_port_reflect(short, short)
    
    # match standard
    match = cpw.match()
    match = rf.two_port_reflect(match, match)

    # Measured cal standards with switch termination
    line_meas  = rf.terminate(X**line**Y, gamma_f, gamma_r)
    short_meas = rf.terminate(X**short**Y, gamma_f, gamma_r)
    match_meas = rf.terminate(X**match**Y, gamma_f, gamma_r)
    
    # DUT
    # attenuator with mismatched feed lines
    dut_feed = cpw.line(2e-3, 'm')
    dut_feed.renormalize(60)
    impedance_transformer = cpw.impedance_mismatch(cpw.z0, 60)
    dut = cpw.attenuator(-5)**impedance_transformer**dut_feed**impedance_transformer.flipped()**cpw.attenuator(-5)
    
    dut_meas = rf.terminate(X**dut**Y, gamma_f, gamma_r)
    dut_meas.name = 'DUT'
    
    cal = LRM(line=line_meas, match=match_meas, reflect=short_meas, 
              reflect_est=-1, gamma_length=line_len*cpw.gamma,
              switch_term=[gamma_f, gamma_r])
    cal.run()

    # Apply calibration to the measured DUT
    dut_cal = cal.apply_cal(dut_meas)

    # plot the comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    # S11 magnitude
    dut.s11.plot_s_db(ax=ax1, label='Original')
    dut_cal.s11.plot_s_db(ax=ax1, label='Calibrated', linestyle='--')
    ax1.set_title('S11 Magnitude')
    ax1.set_ylim([-60, -20])
    ax1.set_xlim([0, freq.f[-1]])
    ax1.legend()
    
    # S21 magnitude
    dut.s21.plot_s_db(ax=ax2, label='Original')
    dut_cal.s21.plot_s_db(ax=ax2, label='Calibrated', linestyle='--')
    ax2.set_title('S21 Magnitude')
    ax2.set_ylim([-10.4, -10])
    ax2.set_xlim([0, freq.f[-1]])
    ax2.legend()
    
    # S11 phase
    dut.s11.plot_s_deg(ax=ax3, label='Original')
    dut_cal.s11.plot_s_deg(ax=ax3, label='Calibrated', linestyle='--')
    ax3.set_title('S11 Phase')
    ax3.set_ylabel(r'Phase ($\times \pi$)')
    ax3.set_yticks([-180, -90, 0, 90, 180])
    ax3.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
    ax3.set_xlim([0, freq.f[-1]])
    
    # S21 phase
    dut.s21.plot_s_deg(ax=ax4, label='Original')
    dut_cal.s21.plot_s_deg(ax=ax4, label='Calibrated', linestyle='--')
    ax4.set_title('S21 Phase')
    ax4.set_ylabel(r'Phase ($\times \pi$)')
    ax4.set_ylim([-180, 180])
    ax4.set_yticks([-180, -90, 0, 90, 180])
    ax4.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
    ax4.set_xlim([0, freq.f[-1]])
    plt.tight_layout()


    plt.show()
    # EOF