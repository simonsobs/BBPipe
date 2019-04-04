import numpy as np

from fgbuster.component_model import CMB

class Bandpass(object):
    def __init__(self, nu, dnu, bnu, bp_number, config, phi_nu=None):
        self.number = bp_number
        self.nu = nu
        self.bnu_dnu = bnu * nu**2 * dnu
        self.cmb_norm = 1./np.sum(CMB('K_RJ').eval(self.nu) * self.bnu_dnu)
        self.nu_mean = np.sum(CMB('K_RJ').eval(self.nu) * self.bnu_dnu * nu) * self.cmb_norm

        # Checking if we'll be sampling over bandpass systematics
        self.do_shift = False
        self.name_shift = None
        self.do_gain = False
        self.name_gain = None
        field = 'bandpass_%d' % bp_number
        try:
            d = config['systematics']['bandpasses'][field]
        except KeyError:
            d = {}
        for n, p in d.items():
            if p[0] == 'shift':
                self.do_shift = True
                self.name_shift = n
            if p[0] == 'gain':
                self.do_gain = True
                self.name_gain = n

        if phi_nu:
            from scipy.interpolate import interp1d
            nu_phi,phi=phi_nu
            phif=interp1d(nu_phi,phi,bounds_error=False,fill_value=0)
            phi_arr=phif(self.nu)
            self.phase = np.cos(phi_arr) + 1j * np.sin(phi_arr)
            self.bnu_dnu *= self.phase

    def convolve_sed(self, sed, params):
        dnu = 0.
        if self.do_shift:
            dnu = params[self.name_shift] * self.nu_mean

        conv_sed = np.sum(sed(self.nu + dnu) * self.bnu_dnu) * self.cmb_norm

        if self.do_gain:
            conv_sed *= params[self.name_gain]

        return conv_sed
