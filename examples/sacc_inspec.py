import numpy as np
import matplotlib.pyplot as plt
import sacc

#Read file
s=sacc.SACC.loadFromHDF("BK15.sacc")

#Print some information about the contents
s.printInfo()

#Check, for instance the shape of the data vector
dv=s.mean.vector
print("Data vector shape: ",s.mean.vector)

#And the covariance matrix
covar=s.precision.getCovarianceMatrix()
print("Covariance matrix shape: ",covar.shape)

#Now, in order to form the theory prediction you need
#information about the frequencies and scales that go into each
#element of the data vector.
#This is what SACC.sortTracers does for you:
for t1,t2,typ,ells,ndx in s.sortTracers() :
    #t1,t2
    #These are the indices of two Tracer objects stored in s.tracers.
    #Let's select them first
    tr1=s.tracers[t1]
    tr2=s.tracers[t2]

    #A Tracer object right now contains information about one of the two
    #sky maps that make up one of the power spectra. The information stored
    #in them is:

    #Name
    print("Maps: "+tr1.name+", "+tr2.name)

    #Type (in this case it just says that it's a spin-2 quantity)
    print("  - Types:"+tr1.type+" "+tr2.type)

    #Which experiment they correspond to (in case you care)
    print("  - Experiments:"+tr1.exp_sample+" "+tr2.exp_sample)

    #Their bandpasses. These are currently stored as two arrays (frequency and bandpass):
    #  Frequency: tr1.z <- in units of GHz
    #  Bandpasss: tr1.Nz <- in arbitrary units
    #(The naming is mostly inspired in what LSST cares about, which is redshift distributions,
    #but I'm pushing towards making this more generic).
    # If you want to check the bandpasses you could uncomment the following
    # plt.plot(tr1.z,tr1.Nz)
    # plt.plot(tr2.z,tr2.Nz)
    # plt.xlabel('Freq. [GHz]')
    # plt.xscale('log')
    # plt.show()
    print("  - Bandpass sizes: %d, %d"%(len(tr1.z),len(tr2.z)))

    #typ:
    #This contains information about the type of correlation this is. I.e. whether it's
    #EE, EB, BE or BB
    print("  - Correlation type: "+typ.decode('utf-8'))

    #ndx:
    #These are the indices of the data vector where the power spectra are stored
    print("  - Indices: ["+' , '.join(["%d"%i for i in ndx])+']')

    #ells:
    #This contains the nominal multipole values at which this specific power spectrum is sampled.
    print("  - Ells: ["+' , '.join(["%.1lf"%l for l in ells])+']')

    #The power spectrum values can be found by evaluating the data vector at ndx
    print("  - C_ells: ["+' , '.join(["%.1lf"%l for l in dv[ndx]])+']')

    #Finally, the bandpower window functions can be accessed through the
    windows=s.binning.windows[ndx]
    # Each window is an actual python object that basically contains two arrays: `ls` and `w`,
    # containing the window function for a given bandpower.
    # A Window object has a `convolve` method that allows you to convolve an input C_ell with the
    # window function. The input power spectrum must have the same size as `ls`.
    #
    # As an example, you can compute the width of the window (not that this is much use, but whatever)
    stds=np.array([np.sqrt(w.convolve(w.ls**2)-w.convolve(w.ls)**2) for w in windows])
    print("  - Ell widths: ["+' , '.join(["%.1lf"%l for l in stds])+']')
    # As a second example, let's first construct a fictitious power-law power spectrum.
    def power_model(l) :
        return (l/80.)**-1.5
    # First we sample it at all ells.
    # You could think of this as your model power spectrum having included all the foreground components etc.
    cell_hires=power_model(windows[0].ls)
    #Note that I'm cheating, because I know all window functions are sampled at the same ells. You can use the same trick too.
    # Now let's do the naive thing and evaluate it at the token ells:
    cell_wrong=power_model(ells)
    # Finally, let's do the proper thing and convolve the high-res power spectrum with the bandpower windows:
    cell_right=[w.convolve(cell_hires) for w in windows]
    # And let's see the differences:
    print("  - Wrong Cl: ["+' , '.join(["%.1lf"%cl for cl in cell_wrong])+']')
    print("  - Right Cl: ["+' , '.join(["%.1lf"%cl for cl in cell_right])+']')
