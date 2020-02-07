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
# import time

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
    parser.add_argument("--nside", type=int, help = "resolution of maps for the analysis", default=512)
    parser.add_argument("--nside_patch", type=int, help = "patch nside for a multipatch approach", default=0)
    parser.add_argument("--sensitivity_mode", type=int, help = "SO V3 sensitivity mode", default=1)
    parser.add_argument("--knee_mode", type=int, help = "SO V3 1/f knee mode", default=1)
    parser.add_argument("--ny_lf", type=float, help = "SO V3 low frequency integration time", default=1.0)
    parser.add_argument("--noise_option", type=str, help = "option for the noise generator", default='white_noise')
    parser.add_argument("--dust_model", type=str, help = "PySM dust model", default='d1')
    parser.add_argument("--sync_model", type=str, help = "PySM sync model", default='s1')
    parser.add_argument("--dust_marginalization", action='store_true',help = "marginalization of the cosmo likelihood over a dust template", default=False)
    parser.add_argument("--sync_marginalization", action='store_true', help = "marginalization of the cosmo likelihood over a sync template", default=False)
    parser.add_argument("--path_to_temp_files", type=str, help = "path to save temporary files, usually scratch at NERSC", default='/global/cscratch1/sd/josquin/SO_pipe/')
    parser.add_argument("--path_to_bbpipe", type=str, help = "path to bbpipe binary", default="/global/homes/j/josquin/.local/cori/3.6-anaconda-5.2/bin/bbpipe")
    parser.add_argument("--path_to_binary_mask", type=str, help = "path to binary mask", default="/global/cscratch1/sd/josquin/SO_sims/norm_nHits_SA_35FOV_G_nside512_binary.fits")
    parser.add_argument("--path_to_norm_hits", type=str, help = "path to normed hits map", default="/global/homes/j/josquin/SIMONS_OBS/BBPipe/test_mapbased_param/norm_nHits_SA_35FOV_G_nside512.fits")
    parser.add_argument("--path_to_ClBBlens", type=str, help = "path to lensed BB angular spectrum", default="/global/homes/j/josquin/SIMONS_OBS/BBPipe/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits")
    parser.add_argument("--path_to_ClBBprim", type=str, help = "path to primordial BB angular spectrum, assuming r=1", default="/global/homes/j/josquin/SIMONS_OBS/BBPipe/test_mapbased_param/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits")
    parser.add_argument("--tag", type=str, help = "specific tag for a specific run, to avoid erasing previous results", default=rand_string)
    parser.add_argument("--apotype", type=str, help = "apodization type", default='C1')
    parser.add_argument("--aposize", type=float, help = "apodization size", default=8.0)
    parser.add_argument("--r_input", type=float, help = "input r value to be assumed", default=0.000)
    parser.add_argument("--AL_input", type=float, help = "input r value to be assumed", default=1.000)
    parser.add_argument("--include_stat_res", action='store_true', help = "estimating and including statistical residuals in the analysis", default=False)
    parser.add_argument("--AL_marginalization", action='store_true',help = "marginalization of the cosmo likelihood over A_lens (lensing BB amplitude)", default=False)
    parser.add_argument("--cmb_sim_no_pysm", action='store_true', help = "perform the CMB simulation with synfast, outside pysm", default=False)
    parser.add_argument("--no_inh", action='store_true', help = "do not generate inhomogeneous noise", default=False)
    parser.add_argument("--nlb", type=int, help = "number of bins", default=512)
    parser.add_argument("--Nspec", type=int, help = "number of slices through the Bd PySM map", default=0)
    parser.add_argument("--mask_apo", type=str, help = "path to apodized mask", default='')
    parser.add_argument("--external_sky_sims", type=str, help = "path to the external, sky simulated, noise-free maps", default='')
    parser.add_argument("--Nsims_bias", type=int, help = "number of simulations performed to estimate the noise bias", default=100)
    parser.add_argument("--extra_apodization", action='store_true', help = "perform an extra apodization of the mask, prior to rescaling by Nhits", default=False)

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
def generate_pipe_yml(id_tag, path_to_temp_files='./', 
        path_to_binary_mask='', path_to_norm_hits='', path_to_ClBBlens='',
        path_to_ClBBprim=''):
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
    # binary_mask: /global/cscratch1/sd/josquin/SO_sims/mask_04000.fits
    binary_mask: '''+path_to_binary_mask+'''
    norm_hits_map: '''+path_to_norm_hits+'''
    Cl_BB_lens: '''+path_to_ClBBlens+'''
    Cl_BB_prim_r1: '''+path_to_ClBBprim+'''

# Overall configuration file 
config: '''+os.path.join(path_to_temp_files,'config_'+id_tag+'.yml')+'''

# If all the outputs for a stage already exist then do not re-run that stage
resume: True

# Put all the output files in this directory:
output_dir: '''+os.path.join(path_to_temp_files,'outputs_'+id_tag)+'''

# Put the logs from the individual stages in this directory:
log_dir: '''+os.path.join(path_to_temp_files,'logs'+id_tag)+'''

# Put the log for the overall pipeline infrastructure in this file:
pipeline_log: '''+os.path.join(path_to_temp_files,'log'+id_tag+'.txt')+'''
    '''

    text_file = open(os.path.join(path_to_temp_files, "test_"+id_tag+".yml"), "w")
    text_file.write( global_string )
    text_file.close()

    return

######################################
#### CONFIG.YML
def generate_config_yml(id_tag, sensitivity_mode=1, knee_mode=1, ny_lf=1.0, \
				noise_option='white_noise', dust_marginalization=True, 
                sync_marginalization=True, path_to_temp_files='./', r_input=0.000, AL_input=1.000,\
                apotype='C2', aposize=10.0, include_stat_res=False, AL_marginalization=False,\
                cmb_sim_no_pysm=False, no_inh=False, nside=512, nside_patch=0, nlb=9, external_sky_sims='',
                Nspec=0.0, Nsims_bias=100, dust_model='d1', sync_model='s1', extra_apodization='False', mask_apo='',):
    '''
    function generating the config file
    '''

    ndim = 1
    if dust_marginalization: ndim += 1
    if sync_marginalization: ndim += 1
    if AL_marginalization: ndim += 1

    global_string = '''
global:
    frequencies: [27,39,93,145,225,280]
    fsky: 0.1
    nside: '''+str(nside)+'''
    lmin: 30
    lmax: 500
    nlb: '''+str(nlb)+'''
    custom_bins: True
    noise_option: \''''+str(noise_option)+'''\'
    include_stat_res: '''+str(include_stat_res)+'''
    r_input: '''+str(r_input)+'''
    A_lens:  '''+str(AL_input)+'''
    no_inh: '''+str(no_inh)+'''
    Nspec: '''+str(Nspec)+'''
    sensitivity_mode: '''+str(sensitivity_mode)+'''
    knee_mode: '''+str(knee_mode)+'''
    ny_lf: '''+str(ny_lf)+'''


BBMapSim:
    cmb_model: 'c1'
    dust_model: \''''+str(dust_model)+'''\'
    sync_model: \''''+str(sync_model)+'''\'
    tag: 'SO_sims'
    cmb_sim_no_pysm: '''+str(cmb_sim_no_pysm)+'''
    external_sky_sims: \''''+str(external_sky_sims)+'''\'

BBMapParamCompSep:
    nside_patch: '''+str(nside_patch)+'''
    smart_multipatch: False

BBClEstimation:
    aposize:  '''+str(aposize)+'''
    apotype:  \''''+str(apotype)+'''\'
    purify_b: True
    Cls_fiducial: './test_mapbased_param/Cls_Planck2018_lensed_scalar.fits'
    Nsims_bias:  '''+str(Nsims_bias)+'''
    extra_apodization: '''+str(extra_apodization)+'''
    mask_apo: \''''+mask_apo+'''\'

BBREstimation:
    dust_marginalization: '''+str(dust_marginalization)+'''
    sync_marginalization: '''+str(sync_marginalization)+'''
    AL_marginalization: '''+str(AL_marginalization)+'''
    ndim: '''+str(ndim)+'''
    nwalkers: 500
    include_stat_res: True
    '''

    text_file = open(os.path.join(path_to_temp_files, "config_"+id_tag+".yml"), "w")
    text_file.write( global_string )
    text_file.close()
    
    return

######################################################
## MAIN FUNCTION
def main():
    '''
    main function
    '''
    args = grabargs()

    # check for output directory
    if not os.path.exists(args.path_to_temp_files):
        os.makedirs(args.path_to_temp_files)

    # if args.r_input!=0.0:
    #     print('you should be careful with r!=0')
    #     print('have you changed the CMB simulator accordingly?')
    #     exit()

    simulations_split = []
    if rank == 0 :
        print('Nsims = ', args.Nsims)
        print('size = ', size)
        simulations_split = chunkIt(list(range(args.Nsims)), size)
        print('simulations_split = ', simulations_split)
        print(simulations_split)
    barrier()
    simulations_split = comm.bcast( simulations_split, root=0 )

    ####################
    for sim in simulations_split[rank]:
        id_tag_rank = format(rank, '05d')
        id_tag_sim = format(sim, '05d')
        id_tag = args.tag+'_'+id_tag_rank+'_'+id_tag_sim
        # create test.yml
        generate_pipe_yml(id_tag, path_to_temp_files=args.path_to_temp_files, 
            path_to_binary_mask=args.path_to_binary_mask, path_to_norm_hits=args.path_to_norm_hits, 
            path_to_ClBBlens=args.path_to_ClBBlens, path_to_ClBBprim=args.path_to_ClBBprim)
        # create config.yml
        generate_config_yml(id_tag, sensitivity_mode=args.sensitivity_mode, knee_mode=args.knee_mode,\
                ny_lf=args.ny_lf, noise_option=args.noise_option, dust_marginalization=args.dust_marginalization,\
                sync_marginalization=args.sync_marginalization,\
                path_to_temp_files=args.path_to_temp_files, r_input=args.r_input, AL_input=args.AL_input,\
                apotype=args.apotype, aposize=args.aposize, include_stat_res=args.include_stat_res,\
                AL_marginalization=args.AL_marginalization, cmb_sim_no_pysm=args.cmb_sim_no_pysm,\
                no_inh=args.no_inh, nside=args.nside, nside_patch=args.nside_patch, nlb=args.nlb,\
                external_sky_sims=args.external_sky_sims, Nspec=args.Nspec, Nsims_bias=args.Nsims_bias,\
                dust_model=args.dust_model, sync_model=args.sync_model, extra_apodization=args.extra_apodization,\
                mask_apo=args.mask_apo)
        # submit call 
        # time.sleep(10*rank)
        print("subprocess call = ", args.path_to_bbpipe,  os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml"))
        # p = subprocess.call("/global/homes/j/josquin/.local/cori/3.6-anaconda-5.2/bin/bbpipe "+os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml"), shell=True, stdout=subprocess.PIPE)
        # p = subprocess.Popen("/global/homes/j/josquin/.local/cori/3.6-anaconda-5.2/bin/bbpipe "+os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml"), shell=True, stdout=subprocess.PIPE)
        # p.communicate()[0]z
        # p.wait()
        # p = subprocess.check_output("/global/homes/j/josquin/.local/cori/3.6-anaconda-5.2/bin/bbpipe "+os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml"))
        p = os.system( args.path_to_bbpipe+' '+os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml"))

        #         p = os.system( args.path_to_bbpipe+' '+os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml --dry-run > log_"+str(rank)+".txt"))

        #         fin = open("log_"+str(rank)+".txt", "rt")
        #         fout = open("batch"+str(rank)+".txt", "wt")

        #         fout.write("#!/bin/bash\n\
        # #SBATCH -N 1\n\
        # #SBATCH -C haswell\n\
        # #SBATCH -q regular\n\
        # #SBATCH -J test_BBpipe\n\
        # #SBATCH -t 00:30:00\n\
        # #OpenMP settings:\n\
        # export OMP_NUM_THREADS=1\n\
        # export OMP_PLACES=threads\n\
        # export OMP_PROC_BIND=spread\n")

        #         for line in fin.readlines():
        #             if line != '\n':
        #                 fout.write('srun -n 1 -c 1 '+line)
        #             else: fout.write(line)
        #         fin.close()
        #         fout.close()

        #         exit()




    ####################
    barrier()
    # grab all results and analyze them
    if rank ==0 :
        # list all the output directories
        list_output_dir = glob.glob(os.path.join(args.path_to_temp_files,'outputs_'+args.tag+'*'))
        # read the estimated_cosmo_params.txt in each directory 
        r_all = []
        Ad_all = []
        sigma_all = []
        sigma_Ad_all = []
        for dir_ in list_output_dir:
            estimated_parameters = np.loadtxt(os.path.join(args.path_to_temp_files,dir_,'estimated_cosmo_params.txt'))
            if args.dust_marginalization: 
                r_, sigma_, Ad_, sigma_Ad_=estimated_parameters
                sigma_Ad_all.append(sigma_Ad_)
                Ad_all.append(Ad_)            
            else: r_, sigma_=estimated_parameters
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
