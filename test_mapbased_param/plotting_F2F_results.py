# plotting the results from BB pipe
# for the F2F @ Berkeley
# July 2019


import argparse
# from mpi4py import MPI
import os
# import subprocess
import random
import string
import glob
import numpy as np
import pylab as pl
from matplotlib import ticker
from matplotlib.ticker import LogLocator

def ticks_format(value, index):
    """
    get the value and returns the value as:
        integer: [0,99]
        1 digit float: [0.1, 0.99]
        n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:   
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))


####################################################################
## INPUT ARGUMENTS
def grabargs():

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_temp_files", type=str, help = "path to save temporary files, usually scratch at NERSC", default='/global/cscratch1/sd/josquin/SO_pipe/')
    parser.add_argument("--tag", type=str, help = "specific tag for a specific run, to avoid erasing previous results", default='')

    args = parser.parse_args()

    return args

######################################################
## MAIN FUNCTION
def main():

    args = grabargs()

    if args.tag == '':
        # give the list of relevant tags for the F2F 
        tag0 = 'knee1_newmask_apo10_nlb5_sens2'## QUEUEING [70%]
        tag1 = 'knee1_newmask_apo10_nlb5_r0p01' ## QUEUEING [30%]
        tag2 = 'wn_newmask_apo10_nlb5' ## DONE 
        tag3 = 'wn_newmask_apo10_nlb5_AL0p5' ## RUNNING ... [30%]
        tag4 = 'WN_newmask_apo10_nlb5_r0p01' ## DONE
        tag5 = 'wn_newmask_apo10_nlb5_no_dust_marg' ## DONE
        tag6 = 'knee1_newmask_apo10_nlb5' ## DONE
        tag7 = 'knee0_newmask_apo10_nlb5' ## DONE
        tag8 = 'WN_external_gaussian_sims' ## QUEUEING [0%]
        tags = [tag5, tag2, tag4, tag3, tag8]
        tag0 = 'outputs_knee2_apo8_nlb10_sens1_Nspec0_d0s0_new_cmb_sim_00000_00000'
        tag1 = 'outputs_knee2_apo8_nlb10_sens1_Nspec0_d1s1_new_cmb_sim_00000_00000'
        tag2 = 'outputs_knee1_apo8_nlb10_sens1_Nspec0_d0s0_new_cmb_sim_00000_00000'
        tag3 = 'outputs_knee1_apo8_nlb10_sens1_Nspec0_d1s1_new_cmb_sim_00000_00000'
        tag4 = 'outputs_knee0_apo8_nlb10_sens1_Nspec0_d0s0_new_cmb_sim_00000_00000'
        # tags = [tag7, tag6, tag1, tag0 ]
        # tags = [tag2]
    else:
        tags = [args.tag]

    ind_tag = 0
    from pylab import  cm
    colors_rainbow = cm.rainbow(np.linspace(0, 1, 5))
    factor = np.linspace(0.95,1.05,num=5)
    pl.figure()
    for tag_ in tags:
        for i in range(9):
            name_folder = 'outputs_'+tag_+'_00000_0000'+str(i)
            if not os.path.exists(os.path.join(args.path_to_temp_files,name_folder)):
                print 'DOWNLOADING FILES FROM CORI ... '
                os.system('mkdir '+os.path.join(args.path_to_temp_files,name_folder))
                os.system('scp -r josquin@cori.nersc.gov:/global/cscratch1/sd/josquin/SO_pipe/results_F2F_Jun19/'+name_folder+'/*txt '+os.path.join(args.path_to_temp_files,name_folder)+'/')
                # os.system('scp -r josquin@cori.nersc.gov:/global/cscratch1/sd/josquin/SO_pipe/results_F2F_Jun19/'+name_folder+'/*pdf '+os.path.join(args.path_to_temp_files,name_folder)+'/')

        # list all the output directories
        list_output_dir = glob.glob(os.path.join(args.path_to_temp_files,'outputs_'+tag_+'_0*'))
        # read the estimated_cosmo_params.txt in each directory 
        r_all = []
        Ad_all = []
        sigma_all = []
        sigma_Ad_all = []
        for dir_ in list_output_dir:
            print tag_
            print dir_
            print np.loadtxt(os.path.join(args.path_to_temp_files,dir_,'estimated_cosmo_params.txt'))
            if 'no_dust' not in tag_:
                # print tag_
                # np.loadtxt(os.path.join(args.path_to_temp_files,dir_,'estimated_cosmo_params.txt'))
                r_, sigma_, Ad_, sigma_Ad_ = np.loadtxt(os.path.join(args.path_to_temp_files,dir_,'estimated_cosmo_params.txt'))
                Ad_all.append(Ad_)
                sigma_Ad_all.append(sigma_Ad_)
            else:
                r_, sigma_= np.loadtxt(os.path.join(args.path_to_temp_files,dir_,'estimated_cosmo_params.txt'))
            r_all.append(r_)
            
            sigma_all.append(sigma_)

        # pl.figure()
        bins = np.logspace(-3.5,-1.5, 20)*factor[ind_tag]
        pl.plot(1e-6, 1e-6, color=colors_rainbow[ind_tag], linewidth=4.0, alpha=0.8, label=tag_)
        # bins = np.linspace(0.00001,0.01, 50)
        # pl.hist( r_all, bins, color='DarkGray', histtype='step', linewidth=4.0, alpha=0.8, label='measured r, \n'+str(round(np.mean(r_all), ndigits=5))+' $\pm$ '+str(round(np.std(r_all), ndigits=5)))
        pl.hist( r_all, bins, color=colors_rainbow[ind_tag], histtype='step', linestyle=':', linewidth=4.0, alpha=0.8, label='measured r, \n'+str(round(np.mean(r_all), ndigits=5))+' $\pm$ '+str(round(np.std(r_all), ndigits=5)))
        # pl.hist( sigma_all, bins, color='DarkOrange', histtype='step', linewidth=4.0, alpha=0.8, label='sigma(r), \n'+str(round(np.mean(sigma_all), ndigits=5))+' $\pm$ '+str(round(np.std(sigma_all), ndigits=5)))
        pl.hist( sigma_all, bins, color=colors_rainbow[ind_tag], histtype='step', linewidth=4.0, alpha=0.8, label='sigma(r), \n'+str(round(np.mean(sigma_all), ndigits=5))+' $\pm$ '+str(round(np.std(sigma_all), ndigits=5)))
        # pl.savefig(os.path.join(args.path_to_temp_files,'histogram_measured_r_and_sigma_'+tag_+'.pdf'))
        ind_tag+=1
    pl.xlim([np.min(bins), np.max(bins)])
    ax = pl.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size':10}, ncol=1, labelspacing=1.0)
    frame = legend.get_frame()
    frame.set_edgecolor('white')
    legend.get_frame().set_alpha(0.3)

    subs = [ 1.0, 2.0, 5.0 ]  
    ax.xaxis.set_major_locator( ticker.LogLocator( subs=subs ) )
    ax.xaxis.set_minor_locator( ticker.LogLocator( subs=subs ) ) #set the ticks position
    ax.xaxis.set_major_formatter( ticker.NullFormatter() )   # remove the major ticks
    ax.xaxis.set_minor_formatter( ticker.FuncFormatter(ticks_format) )
    # ax.yaxis.set_major_locator( ticker.LogLocator( subs=subs ) )
    # ax.yaxis.set_minor_locator( ticker.LogLocator( subs=subs ) ) #set the ticks position
    # ax.yaxis.set_major_formatter( ticker.NullFormatter() )   # remove the major ticks
    # ax.yaxis.set_minor_formatter( ticker.FuncFormatter(ticks_format) )

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.xaxis.get_minor_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_minor_ticks():
        tick.label.set_fontsize(16)

    pl.xlabel('tensor-to-scalar ratio', fontsize=16)
    pl.ylabel('number of simulations', fontsize=16)
    pl.xscale('log')
    pl.savefig(os.path.join(args.path_to_temp_files,'histogram_measured_r_and_sigma_WN.pdf'))
    pl.show()
        # pl.close()
    

    exit()

######################################################
## MAIN CALL
if __name__ == "__main__":
    main( )
