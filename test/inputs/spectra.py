from __future__ import print_function
import numpy as np
import inspect
import sys

class Component(object):
    def __init__(self, comp_name):
        """ SED must be in uK_RJ units.
        """
        self.comp_name = comp_name
        self.sed = globals()[comp_name]
        return

    def __call__(self, nu, args):
        """ Method to call the SED with whichi a given instance was initialized.

        Parameters
        ----------
        nu: float, or array_like(float)
            Frequency or list of frequencies, in GHz, at which to evaluate the
            SED.
        args:  tuple
            Tuple containing the positional parameters taken by the SED.
        """
        return self.sed(nu, *args)

    def get_description(self):
        print("Component SED name: ", self.comp_name)
        print(self.sed.__doc__, "\n --------------- \n")
        pass

    def get_parameters(self):
        """ Method to fetch the keywork arguments for the various components
        and return them. This is used to build a list of possible parameters
        that may be varied by MapLike.
        """
        if sys.version_info[0]>= 3:
            sig = inspect.signature(self.sed)
            pars = list(sig.parameters.keys())
        else :
            pars = list(inspect.getargspec(self.sed).args)
        return list(filter(lambda par: par not in ['nu', 'args', 'kwargs'], pars))


def cmb(nu):
    """ Function to compute CMB SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    """
    x = 0.0176086761 * nu
    ex = np.exp(x)
    sed = ex * (x / (ex - 1)) ** 2
    return sed


def syncpl(nu, nu_ref_s, beta_s):
    """ Function to compute synchrotron power law SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    beta_s: float
        Power law index in RJ units.

    Returns
    -------
    array_like(float)
        Synchroton SED relative to reference frequency.
    """
    x = nu / nu_ref_s
    sed = x ** beta_s
    return sed


def sync_curvedpl(nu, nu_ref_s, beta_s, beta_c):
    """ Function to compute curved synchrotron power law SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    beta_s: float
        Power law index in RJ units.
    beta_c: float
        Power law index curvature.

    Returns
    -------
    array_like(float)
        Synchroton SED relative to reference frequency.
    """
    x = nu / nu_ref_s
    sed = x ** (beta_s + beta_c * np.log(nu / nu_ref_s))
    return sed

def dustmbb(nu, nu_ref_d, beta_d, T_d):
    """ Function to compute modified blackbody dust SED.

    Parameters
    ----------
    nu: float or array_like(float)
        Freuency at which to calculate SED.
    nu_ref_d: float
        Reference frequency in GHz.
    beta_d: float
        Power law index of dust opacity.
    T_d: float
        Temperature of the dust.

    Returns
    -------
    array_like(float)
        SED of dust modified black body relative to reference frequency.
    """
    x_to = 0.0479924466 * nu / T_d
    x_from = 0.0479924466 * nu_ref_d / T_d
    sed = (nu / nu_ref_d) ** (1 + beta_d) * (np.exp(x_from) - 1) / (np.exp(x_to) - 1)
    return sed
