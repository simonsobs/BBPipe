# This code prepares a BB pipeline yml 
# and a configuration file to launch
# a bbpipe run for the map-based 
# component separation pipeline 


import argparse
from mpi4py import MPI
import os
import subprocess
import random
import string
import glob
import numpy as np
import pylab as pl

######################################################################################################
# MPI VARIABLES
comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.rank
barrier=comm.barrier
root=0

######################################################################################################
## JUST GENERATING A RANDOM STRING ;) 
rand_string = ''*10
if rank==0:
	rand_string = ''.join( random.choice(string.ascii_uppercase + string.digits) for _ in range(10) )
barrier()
rand_string = comm.bcast( rand_string, root=0 )

######################################################################################################
## INPUT ARGUMENTS
def grabargs():

    parser = argparse.ArgumentParser()

    parser.add_argument("--Nsims", type=int, help = "number of CMB + noise simulations", default=1)
    parser.add_argument("--sensitivity_mode", type=int, help = "SO V3 sensitivity mode", default=1)
    parser.add_argument("--knee_mode", type=int, help = "SO V3 1/f knee mode", default=1)
    parser.add_argument("--ny_lf", type=float, help = "SO V3 low frequency integration time", default=1.0)
    parser.add_argument("--noise_option", type=str, help = "option for the noise generator", default='white_noise')
    parser.add_argument("--dust_marginalization", type=bool, help = "marginalization of the cosmo likelihood over a dust template", default=True)
    parser.add_argument("--path_to_temp_files", type=str, help = "path to save temporary files, usually scratch at NERSC", default='/global/cscratch1/sd/josquin/SO_pipe/')
    parser.add_argument("--tag", type=str, help = "specific tag for a specific run, to avoid erasing previous results", default=rand_string)
    parser.add_argument("--r_input", type=float, help = "input r value to be assumed", default=0.000)

    args = parser.parse_args()

    return args

######################################
#### distribution of patches over multiprocessing 
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

######################################
#### TEST.YML
def generate_pipe_yml(id_tag, path_to_temp_files='./'):
    global_string = '''
modules: test_mapbased_param

# The launcher to use
# These are defined in bbpipe/sites
launcher: local

# The list of stages to run and the number of processors
# to use for each.
stages:
    - name: BBMapSim 
      nprocess: 1
    - name: BBMapParamCompSep 
      nprocess: 1
    - name: BBClEstimation
      nprocess: 1
    - name: BBREstimation
      nprocess: 1

# Definitions of where to find inputs for the overall pipeline.
# Any input required by a pipeline stage that is not generated by
# a previous stage must be defined here.  They are listed by tag.
inputs:
    binary_mask: /global/cscratch1/sd/josquin/SO_sims/mask_04000.fits
    norm_hits_map: /global/homes/j/josquin/SIMONS_OBS/BBPipe/test_mapbased_param/norm_nHits_SA_35FOV_G.fits
    Cl_BB_lens: /global/homes/j/josquin/SIMONS_OBS/BBPipe/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits
    Cl_BB_prim_r1: /global/homes/j/josquin/SIMONS_OBS/BBPipe/test_mapbased_param/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits

# Overall configuration file 
config: '''+os.path.join(path_to_temp_files,'config_'+id_tag+'.yml')+'''

# If all the outputs for a stage already exist then do not re-run that stage
resume: False

# Put all the output files in this directory:
output_dir: '''+os.path.join(path_to_temp_files,'outputs_'+id_tag)+'''

# Put the logs from the individual stages in this directory:
log_dir: '''+os.path.join(path_to_temp_files,'logs')+'''

# Put the log for the overall pipeline infrastructure in this file:
pipeline_log: '''+os.path.join(path_to_temp_files,'log'+id_tag+'.txt')+'''
    '''

    text_file = open(os.path.join(path_to_temp_files, "test_"+id_tag+".yml"), "w")
    text_file.write( global_string )
    text_file.close()

    return global_string

######################################
#### CONFIG.YML
def generate_config_yml(id_tag, sensitivity_mode=1, knee_mode=1, ny_lf=1.0, \
				noise_option='white_noise', dust_marginalization=True, path_to_temp_files='./',\
					r_input=0.000):
    global_string = '''
global:
    frequencies: [27,39,93,145,225,280]
    fsky: 0.1
    nside: 512
    lmin: 30
    lmax: 200
    nlb: 9
    custom_bins: True

BBMapSim:
    sensitivity_mode: '''+str(sensitivity_mode)+'''
    knee_mode: '''+str(knee_mode)+'''
    ny_lf: '''+str(ny_lf)+'''
    cmb_model: 'c1'
    dust_model: 'd1'
    sync_model: 's1'
    tag: 'SO_sims'
    noise_option: \''''+str(noise_option)+'''\'

BBMapParamCompSep:
    nside_patch: 0
    smart_multipatch: False

BBClEstimation:
    aposize: 10.0
    apotype: 'C1'
    purify_b: True
    Cls_fiducial: './test_mapbased_param/Cls_Planck2018_lensed_scalar.fits'

BBREstimation:
    r_input: '''+str(r_input)+'''
    A_lens: 1.0
    dust_marginalization: '''+str(dust_marginalization)+'''
    ndim: 2
    nwalkers: 500
    '''

    text_file = open(os.path.join(path_to_temp_files, "config_"+id_tag+".yml"), "w")
    text_file.write( global_string )
    text_file.close()
    
    return

######################################################
## MAIN FUNCTION
def main():

    args = grabargs()

    if args.r_input!=0.0:
    	print('you should be careful with r!=0')
    	print('have you changed the CMB simulator accordingly?')
    	exit()

    simulations_split = []
    if rank == 0 :
        print('Nsims = ', args.Nsims)
        print('size = ', size)
        simulations_split = chunkIt(range(args.Nsims), size)
        print('simulations_split = ', simulations_split)
    barrier()
	simulations_split = comm.bcast( simulations_split, root=0 )
    
    ####################
    for sim in simulations_split[rank]:
        id_tag_rank = format(rank, '05d')
        id_tag_sim = format(sim, '05d')
        id_tag = args.tag+'_'+id_tag_rank+'_'+id_tag_sim
        # create test.yml
        generate_pipe_yml(id_tag, path_to_temp_files=args.path_to_temp_files)
        # create config.yml
        generate_config_yml(id_tag, sensitivity_mode=args.sensitivity_mode, knee_mode=args.knee_mode,\
                ny_lf=args.ny_lf, noise_option=args.noise_option, dust_marginalization=args.dust_marginalization,\
                path_to_temp_files=args.path_to_temp_files, r_input=args.r_input)
        # submit call 
        print("subprocess call = ", "/global/homes/j/josquin/.local/cori/3.6-anaconda-5.2/bin/bbpipe", os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml"))
        p = subprocess.call("/global/homes/j/josquin/.local/cori/3.6-anaconda-5.2/bin/bbpipe "+os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml"), shell=True)

    ####################
    barrier()
    # grab all results and analyze them
    if rank ==0 :
        # list all the output directories
        list_output_dir = glob.glob(os.path.join(args.path_to_temp_files,'outputs_'+args.tag+'*'))
        # read the estimated_cosmo_params.txt in each directory 
        r_all = []
        sigma_all = []
        for dir_ in list_output_dir:
            r_, sigma_ = np.loadtxt(os.path.join(args.path_to_temp_files,dir_,'estimated_cosmo_params.txt'))
            r_all.append(r_)
            sigma_all.append(sigma_)

        pl.figure()
        pl.hist( r_all, 20, color='DarkGray', histtype='step', linewidth=4.0, alpha=0.8, label='measured r, '+str(np.mean(r_all))+' +/- '+str(np.std(r_all)))
        pl.hist( sigma_all, 20, color='DarkOrange', histtype='step', linewidth=4.0, alpha=0.8, label='sigma(r), '+str(np.mean(sigma_all))+' +/- '+str(np.std(sigma_all)))
        legend=pl.legend()
        frame = legend.get_frame()
        frame.set_edgecolor('white')
        legend.get_frame().set_alpha(0.3)
        pl.xlabel('tensor-to-scalar ratio', fontsize=16)
        pl.xlabel('number of simulations', fontsize=16)
        pl.savefig(os.path.join(args.path_to_temp_files,'histogram_measured_r_and_sigma_'+args.tag+'.pdf'))
        pl.close()
    
    barrier()
    
    exit()

######################################################
## MAIN CALL
if __name__ == "__main__":
    main( )
