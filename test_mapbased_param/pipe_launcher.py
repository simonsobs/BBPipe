# This code prepares a BB pipeline yml 
# and a configuration file to launch
# a bbpipe run for the map-based 
# component separation pipeline 


import argparse
import os
import subprocess
import random
import string
import glob
import numpy as np
import pylab as pl
import sys
import healpy as hp

######################################################################################################
# MPI VARIABLES
try:
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.rank
    barrier=comm.barrier
    root=0
    mpi = True
except ModuleNotFoundError:
    # Error handling
    mpi = False
    rank=0
    pass

######################################################################################################
## JUST GENERATING A RANDOM STRING ;) 
rand_string = ''*10
if rank==0:
    rand_string = ''.join( random.choice(string.ascii_uppercase + string.digits) for _ in range(10) )
if mpi: rand_string = comm.bcast( rand_string, root=0 )


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
    parser.add_argument("--apotype", type=str, help = "apodization type", default='C2')
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
    parser.add_argument("--external_noise_sims", type=str, help = "path to the external, simulated noise maps", default='')
    parser.add_argument("--external_binary_mask", type=str, help = "path to the external binary mask", default='')
    parser.add_argument("--external_noise_cov", type=str, help = "path to the external noise covariance", default='')
    parser.add_argument("--bmodes_template", type=str, help = "path to an estimated B-modes template, for delensing purposes", default='')
    parser.add_argument("--Nsims_bias", type=int, help = "number of simulations performed to estimate the noise bias", default=100)
    parser.add_argument("--extra_apodization", action='store_true', help = "perform an extra apodization of the mask, prior to rescaling by Nhits", default=False)
    parser.add_argument("--fixed_delta_beta_slicing", action='store_true', help = "when Nspec!=0, regions are defined by constant delta(Bd)", default=False)
    parser.add_argument("--North_South_split", action='store_true', help = "when Nspec!=0, regions are defined by being in the Galactic North or South", default=False)
    parser.add_argument("--bandpass", action='store_true', help = "include non-zero bandpasses in the component separation", default=False)
    parser.add_argument("--instrument", type=str, help = "specifies the instrument bbpipe should consider, SO, CMBS4, etc.", default='SO')
    parser.add_argument("--path_to_dust_template", type=str, help = "path to e.g. PySM dust template", default="/global/cscratch1/sd/josquin/SO_sims/dust_beta.fits")
    parser.add_argument("--pixel_based_noise_cov", action='store_true', help = "pixel-based estimation of the noise covariance matrix", default=False)
    parser.add_argument("--highpass_filtering", action='store_true', help = "high-pass filtering of raw frequency maps to whiten the noise prior to spectral indices estimation", default=False)
    parser.add_argument("--harmonic_comp_sep", action='store_true', help = "performing the estimation of spectral indices in harmonic space", default=False)
    parser.add_argument("--combined_directory", type=str, help = "user provides a directory containing foregrounds, cmb, noise but needs to create a combined directory", default='')
    parser.add_argument("--common_beam_correction",  help = "if not 0, correct for beam-convolution the input simulations, and convolve with this common beam (in arcmin)", default=0.0)
    parser.add_argument("--effective_beam_correction", action='store_true', help = "correct the power spectra by the effective Bl", default=False)
    parser.add_argument("--Nico_noise_combination", action='store_true', help = "Perform the combination of white with one over f noise", default=False)
    parser.add_argument("--force_histogram", action='store_true', help = "compute histogram although all jobs are not run", default=False)
    parser.add_argument("--sky_type", type=str, help = "type of sky input Gaussian, d0s0 or d1s1", default="d0s0")
    parser.add_argument("--time", type=str, help = "duration of the submitted job", default="01:00:00")

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
def generate_config_yml(id_tag, sensitivity_mode=1, knee_mode=1, ny_lf=1.0,
				noise_option='white_noise', dust_marginalization=True, 
                sync_marginalization=True, path_to_temp_files='./', r_input=0.000, AL_input=1.000,
                apotype='C2', aposize=10.0, include_stat_res=False, AL_marginalization=False,
                cmb_sim_no_pysm=False, no_inh=False, nside=512, nside_patch=0, nlb=9,
                external_sky_sims='', external_noise_sims='',
                external_binary_mask='', external_noise_cov='',
                Nspec=0.0, Nsims_bias=100, dust_model='d1', sync_model='s1', extra_apodization='False', 
                mask_apo='',bmodes_template='', fixed_delta_beta_slicing=False, North_South_split=False, 
                instrument='SO',
                frequencies=[27,39,93,145,225,280], bandpass=False, path_to_dust_template='',
                pixel_based_noise_cov=False, highpass_filtering=False, harmonic_comp_sep=False,
                common_beam_correction=0.0, effective_beam_correction=False, combined_directory='',
                Nico_noise_combination=False, isim=0):
    '''
    function generating the config file
    '''

    ndim = 1
    if dust_marginalization: ndim += 1
    if sync_marginalization: ndim += 1
    if AL_marginalization: ndim += 1

    global_string = '''
global:
    frequencies: '''+str(frequencies)+'''
    nside: '''+str(nside)+'''
    lmin: 30
    lmax: '''+str(int(2*nside))+'''
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
    bmodes_template: \''''+str(bmodes_template)+'''\'
    instrument: \''''+str(instrument)+'''\'
    bandpass: '''+str(bandpass)+'''
    Nsims_bias:  '''+str(Nsims_bias)+'''
    common_beam_correction: '''+str(common_beam_correction)+'''
    effective_beam_correction: '''+str(effective_beam_correction)+'''

BBMapSim:
    cmb_model: 'c1'
    dust_model: \''''+str(dust_model)+'''\'
    sync_model: \''''+str(sync_model)+'''\'
    tag: 'SO_sims'
    cmb_sim_no_pysm: '''+str(cmb_sim_no_pysm)+'''
    external_sky_sims: \''''+str(external_sky_sims)+'''\'
    external_noise_sims: \''''+str(external_noise_sims)+'''\'
    combined_directory: \''''+str(combined_directory)+'''\'
    external_binary_mask: \''''+str(external_binary_mask)+'''\'
    external_noise_cov: \''''+str(external_noise_cov)+'''\'
    pixel_based_noise_cov: '''+str(pixel_based_noise_cov)+'''
    Nico_noise_combination: '''+str(Nico_noise_combination)+'''
    isim:  '''+str(isim)+'''

BBMapParamCompSep:
    nside_patch: '''+str(nside_patch)+'''
    smart_multipatch: False
    fixed_delta_beta_slicing: '''+str(fixed_delta_beta_slicing)+'''
    North_South_split: '''+str(North_South_split)+'''
    path_to_dust_template: \''''+str(path_to_dust_template)+'''\'
    highpass_filtering: '''+str(highpass_filtering)+'''
    harmonic_comp_sep: '''+str(harmonic_comp_sep)+'''

BBClEstimation:
    aposize:  '''+str(aposize)+'''
    apotype:  \''''+str(apotype)+'''\'
    purify_b: True
    Cls_fiducial: './test_mapbased_param/Cls_Planck2018_lensed_scalar.fits'
    extra_apodization: '''+str(extra_apodization)+'''
    mask_apo: \''''+mask_apo+'''\'

BBREstimation:
    dust_marginalization: '''+str(dust_marginalization)+'''
    sync_marginalization: '''+str(sync_marginalization)+'''
    AL_marginalization: '''+str(AL_marginalization)+'''
    ndim: '''+str(ndim)+'''
    nwalkers: 500
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

    if mpi:
        simulations_split = []
        if rank == 0 :
            print('Nsims = ', args.Nsims)
            print('size = ', size)
            simulations_split = chunkIt(list(range(args.Nsims)), size)
            print('simulations_split = ', simulations_split)
            print(simulations_split)
        barrier()
        simulations_split = comm.bcast( simulations_split, root=0 )
    else:
        simulations_split = [range(args.Nsims)]

    if args.instrument == 'SO':
        frequencies = [27, 39, 93, 145, 225, 280]
    elif args.instrument == 'CMBS4':
        frequencies = [20, 30, 40, 85, 95, 145, 155, 220, 270]
    else:
        print('I do not know this instrumental configuration')
        sys.exit()

    if args.combined_directory != '':

        if not os.path.exists(args.combined_directory): os.mkdir(args.combined_directory)

        print('looking for simulations on disk and organizing them (e.g. combining CMB+foregrounds)')
        list_of_sky_sim_folders = glob.glob(os.path.join(args.external_sky_sims, '*'))
        if not list_of_sky_sim_folders:
            print('the sky sim folder you provided looks empty!')
            sys.exit()

        list_of_noise_sim_folders = glob.glob(os.path.join(args.external_noise_sims, '*'))
        if not list_of_noise_sim_folders and not args.Nico_noise_combination:
            print('the noise sim folder you provided looks empty!')
            sys.exit()
        if args.Nico_noise_combination:
            print('we will combine white and one over f noise in map_simulator')
            list_of_noise_sim_folders = ['']*args.Nsims

        list_of_combined_directories = [] 
        for i_sim in range(args.Nsims):
            print('creating following repo: ', os.path.join(args.combined_directory, str(i_sim).zfill(4)))
            list_of_combined_directories.append(os.path.join(args.combined_directory, str(i_sim).zfill(4)))
            for f in frequencies:

                if not os.path.isfile(os.path.join(args.combined_directory, str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_comb_'+str(i_sim).zfill(4)+'.fits')):

                    if os.path.isfile(os.path.join(args.combined_directory, str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_comb_'+str(i_sim).zfill(4)+'.fits')): continue

                    if args.sky_type == 'Gaussian':
                        dust = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'FG_20201207/gaussian/foregrounds/dust/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_dust_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None)
                        synch = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'FG_20201207/gaussian/foregrounds/synch/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_synch_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None)
                    elif args.sky_type == 'd0s0':
                        dust = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'FG_20201207/realistic/d0s0/foregrounds/dust/SO_SAT_'+str(f)+'_dust_d0s0*.fits'))[0], field=None)
                        synch = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'FG_20201207/realistic/d0s0/foregrounds/synch/SO_SAT_'+str(f)+'_synch_d0s0*.fits'))[0], field=None)
                    elif args.sky_type == 'd1s1':
                        dust = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'FG_20201207/realistic/d1s1/foregrounds/dust/SO_SAT_'+str(f)+'_dust_d1s1*.fits'))[0], field=None)                    
                        synch = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'FG_20201207/realistic/d1s1/foregrounds/synch/SO_SAT_'+str(f)+'_synch_d1s1*.fits'))[0], field=None)
                    elif args.sky_type == 'dmsm':
                        dust = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'FG_20201207/realistic/dmsm/foregrounds/dust/SO_SAT_'+str(f)+'_dust_dmsm*.fits'))[0], field=None)                    
                        synch = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'FG_20201207/realistic/dmsm/foregrounds/synch/SO_SAT_'+str(f)+'_synch_dmsm*.fits'))[0], field=None)

                    cmb = hp.read_map( glob.glob(os.path.join(args.external_sky_sims, 'CMB_r0_20201207/cmb/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_cmb_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None) 
                    comb = dust + synch + cmb
                    if not os.path.exists(os.path.join(args.combined_directory, str(i_sim).zfill(4))): os.mkdir(os.path.join(args.combined_directory, str(i_sim).zfill(4)))
                    hp.write_map(os.path.join(args.combined_directory, str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_comb_'+str(i_sim).zfill(4)+'.fits'), comb)
    else:
        list_of_combined_directories = [args.external_sky_sims]
        list_of_noise_sim_folders = [args.external_noise_sims]
    print('rank = ', rank, ' and sim_splits = ', simulations_split[rank])
    print('#'*10)

    ####################
    for sim in simulations_split[rank]:
        id_tag_rank = format(rank, '05d')
        id_tag_sim = format(sim, '05d')
        id_tag = args.tag+'_'+id_tag_rank+'_'+id_tag_sim
        print('id_tag = ', id_tag)
        # create test.yml
        generate_pipe_yml(id_tag, path_to_temp_files=args.path_to_temp_files, 
            path_to_binary_mask=args.path_to_binary_mask, path_to_norm_hits=args.path_to_norm_hits, 
            path_to_ClBBlens=args.path_to_ClBBlens, path_to_ClBBprim=args.path_to_ClBBprim)
        # create config.yml
        print(list_of_noise_sim_folders[sim])
        generate_config_yml(id_tag, sensitivity_mode=args.sensitivity_mode, knee_mode=args.knee_mode,\
                ny_lf=args.ny_lf, noise_option=args.noise_option, dust_marginalization=args.dust_marginalization,\
                sync_marginalization=args.sync_marginalization,\
                path_to_temp_files=args.path_to_temp_files, r_input=args.r_input, AL_input=args.AL_input,\
                apotype=args.apotype, aposize=args.aposize, include_stat_res=args.include_stat_res,\
                AL_marginalization=args.AL_marginalization, cmb_sim_no_pysm=args.cmb_sim_no_pysm,\
                no_inh=args.no_inh, nside=args.nside, nside_patch=args.nside_patch, nlb=args.nlb,\
                external_sky_sims=args.external_sky_sims, external_noise_sims=list_of_noise_sim_folders[sim], \
                external_binary_mask=args.external_binary_mask, external_noise_cov=args.external_noise_cov, \
                Nspec=args.Nspec, Nsims_bias=args.Nsims_bias,\
                dust_model=args.dust_model, sync_model=args.sync_model, extra_apodization=args.extra_apodization,\
                mask_apo=args.mask_apo, bmodes_template=args.bmodes_template, fixed_delta_beta_slicing=args.fixed_delta_beta_slicing,\
                North_South_split=args.North_South_split,\
                instrument=args.instrument, frequencies=frequencies, bandpass=args.bandpass, \
                path_to_dust_template=args.path_to_dust_template,\
                pixel_based_noise_cov=args.pixel_based_noise_cov, highpass_filtering=args.highpass_filtering, \
                harmonic_comp_sep=args.harmonic_comp_sep, common_beam_correction=args.common_beam_correction,\
                effective_beam_correction=args.effective_beam_correction, combined_directory=list_of_combined_directories[sim],
                Nico_noise_combination=args.Nico_noise_combination, isim=sim)

        # submit call 
        print("subprocess call = ", args.path_to_bbpipe,  os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml"))

        # if output directory does not exist, then create it
        if not os.path.exists(os.path.join(args.path_to_temp_files,'outputs_'+id_tag)):
             os.mkdir(os.path.join(args.path_to_temp_files,'outputs_'+id_tag))

        # the following lines are generating and submitting a bash job
        p = os.system( args.path_to_bbpipe+' '+os.path.join(args.path_to_temp_files, "test_"+id_tag+".yml --dry-run > log_"+id_tag+".txt"))

        fin = open("log_"+id_tag+".txt", "rt")
        fout = open("batch_"+id_tag+".sh", "wt")

        fout.write("#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH -C haswell\n\
#SBATCH -q regular\n\
#SBATCH -J test_BBpipe\n\
#SBATCH -t "+args.time+"\n\
\n")

        for line in fin.readlines():
            if line != '\n':
                fout.write('srun -n 1 -c 1 '+line)
            # else: fout.write(line)
        fin.close()
        fout.close()

        # submit the job if the final products have not be produced already
        if os.path.isfile(os.path.join(args.path_to_temp_files,'outputs_'+id_tag,'estimated_cosmo_params.txt')):
            print('this has already been computed! '+os.path.join(args.path_to_temp_files,'outputs_'+id_tag,'estimated_cosmo_params.txt'))
        elif args.force_histogram:
            print(' force histogram option = not submitting the job ')
            continue
        else:
            p = os.system('sbatch batch_'+id_tag+".sh")

    ####################
    if mpi: barrier()
    # grab all results and analyze them
    if rank ==0 :
        # list all the output directories
        list_output_dir = glob.glob(os.path.join(args.path_to_temp_files,'outputs_'+args.tag+'*'))
        # read the estimated_cosmo_params.txt in each directory 
        r_all = []
        Ad_all = []
        AL_all = []
        sigma_all = []
        sigma_Ad_all = []
        sigma_AL_all = []
        for dir_ in list_output_dir:
            try:
                estimated_parameters = np.loadtxt(os.path.join(args.path_to_temp_files,dir_,'estimated_cosmo_params.txt'))
            except IOError:
                if args.force_histogram:
                    continue
                else:
                    print('this is a missing file: ', os.path.join(args.path_to_temp_files,dir_,'estimated_cosmo_params.txt'))
                    sys.exit()
            if args.dust_marginalization: 
                r_, sigma_, Ad_, sigma_Ad_=estimated_parameters
                sigma_Ad_all.append(sigma_Ad_)
                Ad_all.append(Ad_) 
            elif args.AL_marginalization:
                r_, AL_, sigma_, sigma_AL_=estimated_parameters
                sigma_AL_all.append(sigma_AL_)
                AL_all.append(AL_) 
            else: r_, sigma_=estimated_parameters
            r_all.append(r_)
            sigma_all.append(sigma_)

        # saving all products 
        np.save(os.path.join(args.path_to_temp_files,'r_all_'+args.tag), r_all)
        np.save(os.path.join(args.path_to_temp_files,'sigma_all_'+args.tag), sigma_all)
        np.save(os.path.join(args.path_to_temp_files,'AL_all_'+args.tag), AL_all)
        np.save(os.path.join(args.path_to_temp_files,'sigma_AL_all_'+args.tag), sigma_AL_all)

        # if args.AL_marginalization:
            # f, ax = pl.subplots(2, 2, sharey=True)
        f, ax = pl.subplots(1, 2, sharey=True)
        # else:
            # f, ax = pl.subplots(1, 2, sharey=True)
            # f, ax = pl.subplots(1, 1)

        ax[0].set_title('r = '+str(round(np.mean(r_all),5))+' +/- '+str(round(np.std(r_all),5)), fontsize=10)
        ax[0].hist( r_all, 40, color='DarkGray', histtype='step', linewidth=3.0, alpha=0.8)
        ax[0].axvline(x=0.0, color='r', linestyle='--', alpha=0.8, linewidth=2.0)
        ax[0].axvline(x=np.mean(r_all), color='DarkGray', linestyle='--', alpha=0.8, linewidth=2.0)
        # ax[0,1].set_title('sigma(r), '+str(np.mean(sigma_all))+' +/- '+str(np.std(sigma_all)))
        # ax[0,1].hist( sigma_all, 20, color='DarkOrange', histtype='step', linewidth=4.0, alpha=0.8)
        # legend=pl.legend()
        # frame = legend.get_frame()
        # frame.set_edgecolor('white')
        # legend.get_frame().set_alpha(0.3)
        # pl.xscale('log')
        ax[0].set_xlabel('tensor-to-scalar ratio', fontsize=12)
        ax[0].set_ylabel('# of sims', fontsize=12)
        # ax[0,1].set_xlabel('tensor-to-scalar ratio', fontsize=12)
        # ax[0,1].set_ylabel('# of sims', fontsize=12)
        # pl.close()
        
        if args.AL_marginalization:
            # pl.figure()
            ax[1].set_title('$A_L$ = '+str(round(np.mean(AL_all),5))+' +/- '+str(round(np.std(AL_all),5)), fontsize=10)
            ax[1].hist( AL_all, 40, color='DarkGray', histtype='step', linewidth=3.0, alpha=0.8)#, label='r = '+str(np.mean(AL_all))+' +/- '+str(np.std(AL_all)))
            ax[1].axvline(x=1.0, color='r', linestyle='--', alpha=0.8, linewidth=2.0)
            ax[1].axvline(x=np.mean(AL_all), color='DarkGray', linestyle='--', alpha=0.8, linewidth=2.0)
            # ax[1,1].hist( sigma_AL_all, 20, color='DarkOrange', histtype='step', linewidth=4.0, alpha=0.8, label='sigma(r), '+str(np.mean(sigma_AL_all))+' +/- '+str(np.std(sigma_AL_all)))
            # pl.xscale('log')
            # legend=pl.legend()
            # frame = legend.get_frame()
            # frame.set_edgecolor('white')
            # legend.get_frame().set_alpha(0.3)
            ax[1].set_xlabel(r'$A_{\rm lens}$', fontsize=12)
            ax[1].set_ylabel('# of sims', fontsize=12)
            # ax[1,1].set_xlabel(r'$A_{\rm lens}$', fontsize=12)
            # ax[1,1].set_ylabel('number of simulations', fontsize=12)
            # pl.savefig(os.path.join(args.path_to_temp_files,'histogram_measured_AL_and_sigma_'+args.tag+'.pdf'))
            # pl.close()
        
        # for ax_ in ax:
        # for i in range(len(ax)):
            # for j in range(ax.shape[1]):
            # ax[i].set_xscale('log')

        f.savefig(os.path.join(args.path_to_temp_files,'histogram_measured_r_and_sigma_'+args.tag+'.pdf'))
        pl.close()

    if mpi: barrier()
    
    exit()

######################################################
## MAIN CALL
if __name__ == "__main__":
    main( )
