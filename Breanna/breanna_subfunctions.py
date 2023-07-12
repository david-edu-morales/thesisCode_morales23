# %%
import numpy as np
import pandas as pd

import os
import os.path
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

import warnings
import pickle
import time
import glob

import csv

warnings.simplefilter(action='ignore', category=UserWarning)                                          # suppress warnings related to older versions of some packages that we need to use to run flopy
warnings.simplefilter(action='ignore', category=RuntimeWarning)                                       # suppress warnings related to older versions of some packages that we need to use to run flopy
# %%
start = time.time()

# stakeholder='env'                                                                                      # choose from ag/town/env
# stage='combine'                                                                                        # choose from random/reseed/combined/afterdata
# prefix = stakeholder + '_' + stage + '_'      #generates filename prefix
# folder = 6.22

def output_directory(folder):     # use this to define where your output files reside
    os.chdir('c:\\Users\\moral\\OneDrive\\UofA\\2022-2023\\Research\\thesisCode_morales23\\Breanna\\{}_output'.format(folder))       # 244 model rerun
 
def figure_directory(folder):     # use this to define where to put your figure files 
     os.chdir('c:\\Users\\moral\\OneDrive\\UofA\\2022-2023\\Research\\thesisCode_morales23\\Breanna\\{}_figures'.format(folder))

def comparison_directory(folder):
    os.chdir('c:\\Users\\moral\\OneDrive\\UofA\\2022-2023\\Research\\thesisCode_morales23\\Breanna\\{}_likelihood'.format(folder))
    
# displaycolumn=40                                                                                       # column for head and drawdown analysis
# displayrow=30                                                                                         # row for head and drawdown analysis
# strdisplaycolumn=35                                                                                   # column for streamflow analysis

# minmismatch=0.05                                                                                      # don't consider mismatches that are less than this value - avoids unrealistically high likelihoods
#                                                                                                       # define criteria to meet to qualify as behavioral ... if a model PASSES this test it is behavioral
# in_time_sequence =       [1,1,1]                                                                      # 0=ntna, 1=ytna, 2=ytya ... enter a number for each criterion applied
# in_basis_sequence =      [0,0,1]                                                                      # see list below
# in_comparison_sequence = [1,0,1]                                                                      # 0 = greater than, 1 = less than
# in_limit_sequence =      [2000,10,3]                                                                  # value defining behavioral response
# in_column_sequence =     [15,15,15]                                                                   # column of observation point for basis 2 or 3  - must have a value for every criterion, even if not used
# in_row_sequence =        [15,15,15]                                                                   # row of observation point for basis 2 or 3 - must have a value for every criterion, even if not used
                  
# behavioral_criteria = [in_time_sequence, in_basis_sequence, in_comparison_sequence, in_limit_sequence, in_column_sequence, in_row_sequence]
# comparison_directory(folder)
# np.save(prefix + 'behavioral_criteria',behavioral_criteria) #save behavioral criteria to a file
# output_directory(folder)

# define_mocs = True                                                                                    # define criteria to meet to qualify as an MOC ... if a model PASSES this test it is a model of concern (MOC)
# if stakeholder=='town':
#     moc_time_sequence =       [2]                                                                       # 0=ntna, 1=ytna, 2=ytya ... enter a number for each criterion applied
#     moc_basis_sequence =      [4]                                                                       # see list below
#     moc_comparison_sequence = [0]                                                                       # 0 = greater than, 1 = less than
#     moc_limit_sequence =      [0.5]                                                                   # value defining behavioral response
#     moc_column_sequence =     [37]                                                                     # column of observation point for basis 2 or 3  - must have a value for every criterion, even if not used
#     moc_row_sequence =        [20]                                                                     # row of observation point for basis 2 or 3 - must have a value for every criterion, even if not used
# elif stakeholder=='ag':
#     moc_time_sequence =       [2]                                                                       # 0=ntna, 1=ytna, 2=ytya ... enter a number for each criterion applied
#     moc_basis_sequence =      [3]                                                                       # see list below
#     moc_comparison_sequence = [1]                                                                       # 0 = greater than, 1 = less than
#     moc_limit_sequence =      [68]                                                                   # value defining behavioral response
#     moc_column_sequence =     [13]                                                                     # column of observation point for basis 2 or 3  - must have a value for every criterion, even if not used
#     moc_row_sequence =        [11]                                                                     # row of observation point for basis 2 or 3 - must have a value for every criterion, even if not used
# elif stakeholder=='env':
#     moc_time_sequence =       [2]                                                                       # 0=ntna, 1=ytna, 2=ytya ... enter a number for each criterion applied
#     moc_basis_sequence =      [2]                                                                       # see list below
#     moc_comparison_sequence = [1]                                                                       # 0 = greater than, 1 = less than
#     moc_limit_sequence =      [50]                                                                   # value defining behavioral response
#     moc_column_sequence =     [38]                                                                     # column of observation point for basis 2 or 3  - must have a value for every criterion, even if not used
#     moc_row_sequence =        [25]                                                                     # row of observation point for basis 2 or 3 - must have a value for every criterion, even if not used
    
# we should add the ability to have an MOC criterion in other than layer 1!!
# add criterion for flow reduction

#   BASES FOR DETERMINATION OF BEHAVIORAL MODEL
#       0 = max streamflow along stream
#       1 = min groundwater depth over first layer
#       2 = streamflow at specified location
#       3 = head at specified location
#       4 = drawdown at specified location
                                                                                                      # use empty brackets if no data available, each value must contain the same number of inputs, separate multiple data points by commas
# moc_criteria = [moc_time_sequence, moc_basis_sequence, moc_comparison_sequence, moc_limit_sequence, moc_column_sequence, moc_row_sequence]
# comparison_directory(folder)
# np.save(prefix + 'moc_criteria',moc_criteria) #save moc criteria to a file
# output_directory(folder)
'''
read_true_data=True                                                                                   # True to read in the observations from truth_heads_ss_ytna.npy and truth_flows_ss_ytna.npy
if read_true_data==True:
    output_directory(folder)
    trueheads_ss_ytna=np.load('truth_heads_ss_ytna.npy')[0][:][:]
    trueflows_ss_ytna=np.load('truth_strflow_ss_ytna.npy')[:]

usedata = True                                                                                        # set to true to use available data to calculate likelihoods, false to set models as equally likely
data_time_sequence = [1,1]                                                                            # 0 = ntna, 1 = ytna, 2 = ytya
data_basis_sequence =[1,1]                                                                            # identify head or flow observation
data_layer_sequence = [0,0]                                                                           # layer of head observation - enter zero for flow
data_column_sequence = [35,20]                                                                         # column of head or flow observation
data_row_sequence = [20,30]                                                                           # row of head or flow observation
data_value_sequence = np.zeros((np.shape(data_row_sequence)))
for ii in np.arange(np.shape(data_row_sequence)[0]):                                                              
    if data_basis_sequence[ii]==1:
        data_value_sequence[ii]=trueheads_ss_ytna[data_row_sequence[ii],data_column_sequence[ii]]     # Retrieve head data in top layer from the 'truth' model
    elif data_basis_sequence[ii]==0:
        data_value_sequence[ii]=trueflows_ss_ytna[data_column_sequence[ii]][1]                        # Retrieve flow data in stream from the 'truth' model
        
#   TYPES OF OBSERVATION DATA CONSIDERED
#       0 = streamflow at specified location
#       1 = head at specified location

if usedata==True:
    num_data=np.shape(data_time_sequence)[0]
else:
    num_data=0

eliminate_lowL_models=True                                                                            # True to eliminate low Likelihood models from the ensemble - for now, always set to true, but set number to eliminate to zero to turn off
lowLcut_percent=10                                                                                    # remove this percent of models with the lowest L values
lowLcut_number=0                                                                                    # don't remove any more than this number of models, no matter how low their L value, despite above limit
lowLlimit=0.05                                                                                        # don't remove models with an L higher than this, despite above limits
lowLecho=True                                                                                         # True to list the low L models, False to suppress (in case you have a lot!)

#Export files needed for multi-stakeholder comparisons:
#To import, use np.load()
comparison_directory(folder)               #change directory to current stakeholder and stage
np.save(prefix + 'data_layer', data_layer_sequence)
np.save(prefix + 'data_row', data_row_sequence)
np.save(prefix + 'data_column', data_column_sequence)
output_directory(folder)'''

# use the following to control which analyses are completed - may be useful when running partial analyses for many models
# run_sections=3

# 0 = through identifying behavioral models and calculating model likelihoods
# 1 = identify models of concern
# 2 - calculate discriminatory index
# 3 - particle capture

# %%
#===========READINMODELS FUNCTION==================================================================
def readInModels(folder):
    output_directory(folder); mydir = os.getcwd()             # set the current directory to output folder
    file_list = glob.glob(mydir + "/m*_parvals")        # only consider output model files

    runnumbers   = []                                   # list to hold model names
    t_runnumbers = []

    for i in file_list:                                 # loop through file_list
        i = os.path.basename(i)                         # isolate filename from path
        if i[16] == '_' and i[16:26] != '_heads_pod':   # only select S.S. models
            runnumbers.append(i[0:16])
        elif i[16:26] == '_heads_pod':                  # only select tran models
            t_runnumbers.append(i[0:16])
    return runnumbers, t_runnumbers
'''
files = os.listdir(os.curdir)
runnumbers, t_runnumbers = readInModels(files)
'''
# %%
#==========RUN_PARVALS FUNCTION====================================================================
def run_parvals(runnumbers):
    Nmodels = np.shape(runnumbers)[0]

    run_params=np.zeros((Nmodels,7))                                                      # prepare an array to store parameter values that differ among model runs
    for i in np.arange(Nmodels):                                                          # loop over models to read parameter values from parvals file - not used for now, but may be
        with open (runnumbers[i]+'_parvals', 'rb') as fp:
            [nrow,ncol,delr,delc,Lx,Ly,nlay,ztop,crop,fNWc,well1,well2,
             recharge_ratio,return_loc,rNWc, K_horiz, Kzratio_low, Sy, 
             R1, ET1, ETratio_riparian, Kratio_stream] = pickle.load(fp)
            run_params[i,0]=K_horiz
            run_params[i,1]=Kzratio_low
            run_params[i,2]=Sy
            run_params[i,3]=R1
            run_params[i,4]=ET1
            run_params[i,5]=ETratio_riparian
            run_params[i,6]=Kratio_stream
    return run_params, [nrow,ncol,delr,delc,Lx,Ly,nlay,ztop,crop,fNWc,well1,well2,
            recharge_ratio,return_loc,rNWc, K_horiz, Kzratio_low, Sy, 
            R1, ET1, ETratio_riparian, Kratio_stream]


# %%
# #====================================================================================
# files = os.listdir(os.curdir)
# runnumbers, t_runnumbers = readInModels(files)
# run_params, [nrow,ncol,delr,delc,Lx,Ly,nlay,ztop,crop,fNWc,well1,well2,
#             recharge_ratio,return_loc,rNWc, K_horiz, Kzratio_low, Sy, 
#             R1, ET1, ETratio_riparian, Kratio_stream] = run_parvals(runnumbers)
# %%
#============IMPORTMODELRESULTS=======================================================
def modelResults(runnumbers, ztop, return_loc):
    Nmodels = len(runnumbers)
    scenario = ('ntna','ytna','ytya')

    #==================H E A D S=====================================================
    # initialize arrays to accept all heads for each ss run
    scenario_list = []
    for s in scenario:
        heads_arr = np.zeros((Nmodels, 50, 50))
        for i in range(Nmodels):
            # use head values from top layer
            heads = np.load(runnumbers[i]+'_heads_ss_'+s+'.npy')[0,:,:]
            # remove the corners and other negative values because they confuse std deviation
            heads[heads<0] = 0
            heads_arr[i] = heads
        scenario_list.append(heads_arr)

    allheads_ss = dict(zip(scenario, scenario_list))    # compile list into dictionary

    # remove the corners as they confused the standard deviation calculation (NOTE: this step was moved into the for loop above)
    # allheads_ss[s][allheads_ss[s]<0]=0

    #===================B U D G E T S===============================================
    # initialize arrays to accept all budgets for each ss run
    scenario_list = []
    for s in scenario:
        budget_arr = np.zeros((Nmodels, 16))
        for i in range(Nmodels):
            budget=np.load(runnumbers[i]+'_budget_ss_'+s+'.npy')
            for j in range(16):
                budget_arr[i][j] = budget[j][1]
        scenario_list.append(budget_arr)

    allbudgets_ss = dict(zip(scenario,scenario_list))              

    # budget components are:
    # 0: STORAGE_IN         4: RECHARGE_IN          8: CONSTANT_HEAD_OUT    12: STREAM_LEAKAGE_OUT
    # 1: CONSTANT_HEAD_IN   5: STREAM_LEAKAGE_IN    9: WELLS_OUT            13: TOTAL_OUT
    # 2: WELLS_IN           6: TOTAL_IN             10: ET_OUT              14: IN-OUT
    # 3: ET_IN              7: STORAGE_OUT          11: RECHARGE_OUT        15: PERCENT_DISCREPANCY

    #==================DEPTH TO WATER TABLE===================================
    scenario_dwt = []; scenario_maxdwt = []
    for s in scenario:
        # initialize arrays to receive water table depth values
        dwt_arr = np.zeros((Nmodels,50,50)); maxdwt_arr = np.zeros(Nmodels)
        # loop through all models to calculate depth to water table
        for i in np.arange(Nmodels):
            tempvar = allheads_ss[s][i,:,:]-ztop
            dwt_arr[i] = tempvar
            maxdwt_arr[i] = np.max(tempvar)
        # append array of recorded values to list, ordered by scenario
        scenario_dwt.append(dwt_arr); scenario_maxdwt.append(maxdwt_arr)

    # compile results into dictionaries
    alldwt_ss = dict(zip(scenario, scenario_dwt)); maxdwt_ss = dict(zip(scenario, scenario_maxdwt))

    #==================D R A W D O W N========================================
    # define drawdown for each model at each location in top layer from ytna and ytya
    dd=allheads_ss['ytna']-allheads_ss['ytya']                                                                 

    #==================F L O W S==============================================
    scenario_flows = []
    for s in scenario:
        # initialize array to receive flow values
        flow_arr = np.zeros((Nmodels, 49))
        # loop through all models to extract data from files
        for i in range(Nmodels):
            flows = np.load(runnumbers[i]+'_strflow_ss_'+s+'.npy')
            flow = []
            for tup in range(len(flows)):
                flow.append(flows[tup][1])
            flow_arr[i] = flow
        flow_arr = np.delete(flow_arr, return_loc, axis=1)
        # append array of recorded flows to list, ordered by scenario
        scenario_flows.append(flow_arr)
    #compile results into dictionary
    allflows_ss = dict(zip(scenario, scenario_flows))
        
    #==================LEAKAGE================================================
    value    = np.zeros((Nmodels,49))
    allleaks_ss = dict.fromkeys(scenario,value)

    # for s in scenario:
    #     for i in range(Nmodels):
    #         leaks=np.load(runnumbers[i]+'_strleak_ss_'+s+'.npy')
    #         leak = []
    #         for tup in range(len(leaks)):
    #             leak.append(leaks[tup][1])
    #         allleaks_ss[s][i][:][:] = leak

    scenario_leakage = []
    for s in scenario:
        # initialize array to receive leakage values
        leak_arr = np.zeros((Nmodels,49))
        # loop through all models to extract data from files
        for i in range(Nmodels):
            leaks = np.load(runnumbers[i]+'_strleak_ss_'+s+'.npy')
            leak = []
            for tup in range(len(leaks)):
                leak.append(leaks[tup][1])
            leak_arr[i] = leak
        # append array of recorded leaks to list, ordered by scenario
        scenario_leakage.append(leak_arr)
    # compile results into dictionary
    allleaks_ss = dict(zip(scenario, scenario_leakage))

    #===================PARTICLE TRACKING======================================
    # read this file to set dimensions of arrays in next lines
    testepts=np.load(runnumbers[0]+'_epts_ss_ytna.npy')

    # initialize arrays to accept all particle data for each ss run
    allepts_ss_ntna=np.zeros((np.shape(runnumbers)[0],np.shape(testepts)[0],np.shape(testepts)[1]))       
    allepts_ss_ytna=np.zeros((np.shape(runnumbers)[0],np.shape(testepts)[0],np.shape(testepts)[1]))     
    allepts_ss_ytya=np.zeros((np.shape(runnumbers)[0],np.shape(testepts)[0],np.shape(testepts)[1]))     

    counter=-1
    for i in runnumbers:                                       
        counter=counter+1                               # count the models as loaded to form joint array with all models' results

        epts=np.load(i+'_epts_ss_ntna.npy')             # load output for each model in ensemble
        for j in np.arange(np.shape(testepts)[0]):      
            allepts_ss_ntna[counter,j,0]=epts[j][8]     # start point column (all particles assumed to start in top layer, as recharge)
            allepts_ss_ntna[counter,j,1]=epts[j][7]     # start point row 
            allepts_ss_ntna[counter,j,2]=epts[j][20]    # end point column 
            allepts_ss_ntna[counter,j,3]=epts[j][19]    # end point row 
            allepts_ss_ntna[counter,j,4]=epts[j][4]     # end point time 

        epts=np.load(i+'_epts_ss_ytna.npy')  
        for j in np.arange(np.shape(testepts)[0]):
            allepts_ss_ytna[counter,j,0]=epts[j][8]            
            allepts_ss_ytna[counter,j,1]=epts[j][7]            
            allepts_ss_ytna[counter,j,2]=epts[j][20]           
            allepts_ss_ytna[counter,j,3]=epts[j][19]           
            allepts_ss_ytna[counter,j,4]=epts[j][4]

        epts=np.load(i+'_epts_ss_ytya.npy')  
        for j in np.arange(np.shape(testepts)[0]):
            allepts_ss_ytya[counter,j,0]=epts[j][8]            
            allepts_ss_ytya[counter,j,1]=epts[j][7]            
            allepts_ss_ytya[counter,j,2]=epts[j][20]           
            allepts_ss_ytya[counter,j,3]=epts[j][19]          
            allepts_ss_ytya[counter,j,4]=epts[j][4]       

    # clear temporary variables
    del heads                                                                                            
    del tempvar
    del flow
    del flows
    del leak
    del leaks
    del testepts
    del counter
    del epts

    return [allheads_ss, allbudgets_ss, alldwt_ss, allflows_ss, allleaks_ss, maxdwt_ss,
                allepts_ss_ntna, allepts_ss_ytna, allepts_ss_ytya, dd]

# %%
def nonbehavioralModels(runnumbers, allheads_ss, maxdwt_ss, allflows_ss, dd, behavioral_criteria):
    scenario = ('ntna', 'ytna', 'ytya')
    in_time_seq       = behavioral_criteria[0]; in_basis_seq = behavioral_criteria[1]
    in_comparison_seq = behavioral_criteria[2]; in_limit_seq = behavioral_criteria[3]
    in_column_seq     = behavioral_criteria[4]; in_row_seq   = behavioral_criteria[5]


    Nmodels = np.shape(runnumbers)[0]

    cullmodels=np.zeros((Nmodels,50))                   # store values used to assess (non)behavioural and likelihood to check process
    cullmodels_counter=-1

    # determine number of criteria to that have been applied
    # if 'in_time_seq' in globals():
    #     num_criteria=np.shape(in_time_seq)[0]
    num_criteria=np.shape(in_time_seq)[0]

    if num_criteria>0:
        print('Assessing (non)behavioral criteria')

        # loop over bases for discrimination of models of concern or (non)behavioral models
        for ii in np.arange(num_criteria):          
            cullmodels_counter += 1
            print('Assessing criterion',ii)

            # establish the time sequence (scenario) for each criterion
            in_time=in_time_seq[ii]
            s = scenario[in_time]           # set value for scenario ('ntna','ytna','ytya')

            # set criteria values for basis for consideration, greater/less than, and value limit
            in_basis=in_basis_seq[ii] 
            in_comparison=in_comparison_seq[ii]
            in_limit=in_limit_seq[ii]

            # set column and row values for observation
            in_column=in_column_seq[ii]
            in_row=in_row_seq[ii]

            if in_basis==0:                                         # store maximum flow for each model
                in_metric=np.zeros((np.shape(allflows_ss[s])[0]))
                in_metric=np.max(allflows_ss[s], axis=1)

            elif in_basis==1:                                       # store minimum flow for each model
                in_metric=np.zeros((np.shape(allflows_ss[s])[0]))  
                in_metric=maxdwt_ss[s]

            elif in_basis==2:                                       # store flow at specified location for each model
                in_metric=np.zeros((np.shape(allflows_ss[s])[0]))    
                for j in np.arange(np.shape(allflows_ss[s])[0]):
                    in_metric[j]=allflows_ss[s][j][in_column]

            elif in_basis==3:                                       # store flow at specified location for each model
                in_metric=np.zeros((np.shape(allheads_ss[s])[0]))
                for j in np.arange(np.shape(allheads_ss[s])[0]):
                    in_metric[j]=allheads_ss[s][j][in_row][in_column]

            else:
                in_metric=np.zeros((np.shape(dd)[0]))               # store flow at specified location for each model
                if in_time<=1:
                    for j in np.arange(np.shape(dd)[0]):                
                        in_metric[j]=dd[j][in_row][in_column]               
                else:
                    for j in np.arange(np.shape(allheads_ss['ytna'])[0]):                
                        in_metric[j]=dd[j][in_row][in_column]

            cullmodels[:,cullmodels_counter]=in_metric                      # store data used to check (non)behavioral status

        del num_criteria                                                    # clear temporary variables
        del in_time
        del in_basis
        del in_comparison
        del in_limit
        del in_column
        del in_row
        del in_metric

    else:
        print('No (non)behavioral criteria listed')

    return cullmodels, cullmodels_counter
    
# %%
def compileLikelihoodData(cullmodels, cullmodels_counter, dict_L_criteria, num_data, allflows_ss, allheads_ss, scenario, Nmodels, prefix):
    startdata = cullmodels_counter + 1                                  # start of data for likelihood estimation
    holdfordataworth=np.zeros((num_data,Nmodels))                       # array to hold comparison values
    if num_data > 0:
        # Loop through all models to determine extract comparison value to evaluate model likelihood
        for jj in range(Nmodels):
            # Loop through all evaluation options
            for ii in range(num_data):
                in_basis = dict_L_criteria['basis'][ii]                 # basis for determining Likelihood
                in_row   = dict_L_criteria['row'][ii]                   # cell row for comparison
                in_col   = dict_L_criteria['column'][ii]                # cell column for comparison
                in_time  = dict_L_criteria['time'][ii]                  # time sequence (scenario) for criterion
                s = scenario[in_time]                                   # set value for scenario ('ntna','ytna','ytya')
                # extract comparison value based on basis (flow/head)
                if in_basis == 0:                                       # streamflow, basis = 0
                    data2check = allflows_ss[s][jj][in_row][in_col]
                else:                                                   # head, basis = 1
                    data2check = allheads_ss[s][jj][in_row][in_col]
                # append comparison value according to above criteria to cullmodels array
                cullmodels[jj, startdata+ii] = data2check
                # record comparison value to array
                holdfordataworth[ii,jj] = data2check
        # save array of recorded comparison values
        np.save(prefix + 'holdfordataworth', holdfordataworth)
        
    else:
        dummy = 0
        del dummy
    
    return startdata, cullmodels, cullmodels_counter

# %%
#==================U S E  T R U E  D A T A===============================================
def useTrueData(dict_L_criteria, folder):
    # set up variable values
    Ncomparisons = len(dict_L_criteria['time'])     # determine number of comparisons
    data_basis   = dict_L_criteria['basis']         # basis for comparison (0: streamflow, 1: head)
    data_row     = dict_L_criteria['row']           # cell row for comparison value 
    data_column  = dict_L_criteria['column']        # cell column for comparison value
    # switch to output directory to interact with truth model files
    output_directory(folder)
    # set truth model values to objects
    trueheads_ss_ytna=np.load('truth_heads_ss_ytna.npy')[0][:][:]           # load truth heads file
    trueflows_ss_ytna=np.load('truth_strflow_ss_ytna.npy')[:]               # load truth flows file 
    # create array to record truth values                                      
    data_value = np.zeros(Ncomparisons)                                                           
    # Record truth value for each comparison depending on basis
    for ii in range(Ncomparisons):                                                              
        if data_basis[ii] == 1:
            data_value[ii]=trueheads_ss_ytna[data_row[ii],data_column[ii]]     # Retrieve head data in top layer from the 'truth' model
        elif data_basis[ii] == 0:
            data_value[ii]=trueflows_ss_ytna[data_column[ii]][1]               # Retrieve flow data in stream from the 'truth' model
    
    return data_value, Ncomparisons

# %%
#==================A S S E S S  N O N B E H A V I O R A L  M O D E L S===================
def assess_nonBehavioral(dict_B_criteria, rmse, cullmodels):
    # extract the number of nonbehavioral criteria to assess each model
    NBcriteria = len(dict_B_criteria['comparison'])
    # loop through each criterion to determine if each model falls within behavioral limits
    for i in range(NBcriteria):
        comparison = dict_B_criteria['comparison'][i]       # 0: greater than, 1: less than
        limit      = dict_B_criteria['limit'][i]            # the limiting value
        # assign extreme rmse value to nonbehavioral models
        if comparison == 0:
            rmse[cullmodels[:,i] <= limit] = 1.2345e9
        else:
            rmse[cullmodels[:,i] >= limit] = 1.2345e9

    return rmse

# %%
#==================C A L C U L A T E  M O D E L  L I K E L I H O O D=====================
def calculateModelLikelihood(dict_B_criteria, truth_value, cullmodels, cullmodels_counter, num_L_criteria, startdata, Nmodels, useTrueData_flag):
    rmse=np.zeros(Nmodels)                  # use model rmse to flag nonbehavioral models below
    L = 1/Nmodels                           # set likelihoods equal by default, keep if there are no data to compare

    # determine model specific model likelihood when flag is on
    if useTrueData_flag == True:
        # set counter to first truth value
        cullmodels_counter = cullmodels_counter + num_L_criteria
        # set mismatch value to zero. iteratively summed with each 
        mmsqsum = 0
        
        for i in range(num_L_criteria):
            cullmodels_counter += 1                     # advance the counter +1
            # extract simulated value across all models for comparison
            simValue = cullmodels[:, i+startdata]
            # square mismatch for first datapoint      
            mmsq = ((simValue-truth_value[i])**2)
            # store square mistmatches across all models for later checking
            cullmodels[:, cullmodels_counter] = mmsq
            # sum square mismatches
            mmsqsum += mmsq
        # calculate root mean sqaure mismatch    
        rmse = (mmsqsum/(i+1))**0.5

        cullmodels_counter += 1                         # advance counter to record rmse
        cullmodels[:, cullmodels_counter] = rmse

        rmse = assess_nonBehavioral(dict_B_criteria, rmse, cullmodels)
        
        cullmodels_counter += 1
        # store rmse after culling non-behavioral models
        cullmodels[:, cullmodels_counter] = rmse
        # invert rmse (1st step to calculating likelihood)
        Ltemp = 1/rmse
        L = Ltemp/np.sum(Ltemp)
        cullmodels_counter += 1
        cullmodels[:, cullmodels_counter] = L

        sorted_L_behavioral=np.sort(L)[::-1]
    
    else:
        rmse = assess_nonBehavioral(dict_B_criteria, rmse, cullmodels)
        sorted_L_behavioral = L

        # this portion of code is to specify Likelihoods of models based on their mismatch.
        # in Ty's code, there is an else statement that bypasses this specified likelihoods, but still
        # sets nonbehavioral models to incredibly high rmse. 
        # First, make a function that handles nonbehavioral models and stack it inside of the likelihood function.

    # sorted_L_behavioral=np.sort(L)[::-1]

    return rmse, cullmodels_counter, cullmodels, L, sorted_L_behavioral

# %%
#==================M O D E L  B E H A V I O R============================================
def modelBehavior(runnumbers, rmse):
    # convert list of model names into an array for masking
    runnumbers_arr = np.array(runnumbers)
    # boolean masks for (non)behavioral lists
    behavioral_mask    = np.ma.less(rmse, 1.2345e9)
    nonbehavioral_mask = np.ma.greater_equal(rmse, 1.2345e9)
    # lists of indices for (non)behavioral models
    behavioral_idx    = np.where(behavioral_mask)[0].tolist()
    nonbehavioral_idx = np.where(nonbehavioral_mask)[0].tolist()
    # lists of (non)behavioral model names
    behavioral_models = runnumbers_arr[behavioral_mask].tolist()
    nonbehavioral_models = runnumbers_arr[nonbehavioral_mask].tolist()

    return [behavioral_idx, behavioral_models, nonbehavioral_idx, nonbehavioral_models]

# %%
#==================B E H A V I O R  S T A T I S T I C S==================================
def behaviorStats (modelBehavior, runnumbers, dict_B_criteria, cullmodels, Nmodels):
    counter = -1
    # store values for later plotting
    holdplotx=[]; holdploty=[]
    holdleftlimit=[]; holdrightlimit=[]; holdplottype=[]
    # determine number of criteria that have been applied
    num_criteria = len(dict_B_criteria['time'])
    # unpack list of model idx and names
    behavioral_idx    = modelBehavior[0]; behavioral_models = modelBehavior[1]
    nonbehavioral_idx = modelBehavior[2]; nonbehavioral_models = modelBehavior[3]

    if num_criteria > 0:
        # loop over bases for discrimination of MOCs or (non)behavioral models
        for ii in range(num_criteria):
            in_comparison = dict_B_criteria['comparison'][ii]
            # sort metric in decreasing order for plotting
            sorted_in_metric = np.sort(cullmodels[:,ii])[::-1]

            # save results to plot later
            plottype = 0; holdplottype.append(plottype)     # distinguish behavioral (0) and MOC (1) plots
            x_range = np.arange(Nmodels)                    # store ordinal numbers from 0 to Nmodels
            holdplotx.append(x_range)
            holdploty.append(sorted_in_metric)              # hold sorted values of the metric used for assessment, H to L
            # store x & y positions for plotting limits
            if in_comparison == 0:                          # 'greater than' indicates behavioral
                holdleftlimit.append(len(behavioral_idx))
                holdrightlimit.append(Nmodels)
            elif in_comparison == 1:                        # 'less than' indicates behavioral
                holdleftlimit.append(0)
                holdrightlimit.append(Nmodels-len(behavioral_idx))
        # assess prevalance of each parameter for 'in' and 'not in' groups
        if len(nonbehavioral_idx) > 0:
            tempvar = []
            b_meanid = np.zeros(7); b_varid  = np.zeros(7)
            # count over variable parameter positions, 9 through 15
            for i in np.arange(9,16):
                # temporary variable to hold values for a single parameter
                tempvar = np.zeros(len(behavioral_idx))
                counter = -1
                # record parameter value for each behavioral model
                for j in behavioral_idx:
                    counter += 1
                    # extract parameter value from model name (str)
                    tempvar[counter] = int(runnumbers[j][i])
                # calculate the mean of the parameter values from behavioral models
                b_meanid[i-9] = np.mean(tempvar)
                # calculate the STD of the parameter values from behavioral models
                b_varid[i-9]  = np.std(tempvar)
            # repeat above process for nonbehavioral models
            tempvar = []
            nonb_meanid = np.zeros(7); nonb_varid  = np.zeros(7)
            for i in np.arange(9,16):
                tempvar = np.zeros(len(nonbehavioral_idx))
                counter = -1
                for j in nonbehavioral_idx:
                    counter += 1
                    tempvar[counter] = int(runnumbers[j][i])
                nonb_meanid[i-9] = np.mean(tempvar)
                nonb_varid[i-9]  = np.std(tempvar)
            nonb_meanid = np.round(nonb_meanid*1000)/1000
            nonb_varid  = np.round(nonb_varid*1000)/1000

            # pack (non)behavioral statistics into dictionary for ease of recall
            stats = [b_meanid, b_varid, nonb_meanid, nonb_varid]
            keys  = ['b_mean', 'b_var', 'nonb_mean', 'nonb_var']
            dict_B_stats = dict(zip(keys, stats))
            # pack plotting information into dictionary for ease of recall
            hold_list = [holdplotx, holdploty, holdleftlimit, holdrightlimit, holdplottype]
            plot_keys = ['x', 'y', 'l_limit', 'r_limit', 'type']
            dict_plotHolds = dict(zip(plot_keys, hold_list))

        else:
            print('All models are behavioral')
    else:
        print('No (non)behavioral criteria listed.')

    return dict_B_stats, dict_plotHolds

# %%
#==================C U L L  L O W  L I K E L I H O O D  M O D E L S======================
def cull_lowL_models(L, dict_lowL_options, rmse, modelBehavior, cullmodels, cullmodels_counter, Nmodels):
    # set number of nonbehavioral models for cutting
    Nnonbehavioral_models = len(modelBehavior[2])
    lowL_limit = dict_lowL_options['limit']
    lowL_cutNumber = dict_lowL_options['number']
    # sorted_L  = np.sort(L)[::-1]
    # sort model indices by likelihood, increasing
    Lcut_ids  = np.argsort(L)
    Lcut_vals = np.sort(L)
    # find number of models represented by % to remove
    lowL_cutPercent = dict_lowL_options['percent']
    number_to_remove = int(Nmodels * lowL_cutPercent/100) + Nnonbehavioral_models
    # don't cut models with L above low limit
    checkLimit = np.sum(Lcut_vals < lowL_limit)
    number_to_remove = min(checkLimit, number_to_remove)
    number_to_remove = min(number_to_remove, lowL_cutNumber)
    # don't allow number to remove to pass number list
    Lcut_ids = Lcut_ids[0:number_to_remove]
    rmse[Lcut_ids] = 1.23456e9
    Ltemp = 1/rmse
    L = Ltemp/np.sum(Ltemp)

    cullmodels_counter = cullmodels_counter + 1
    # store data used to check (non)behavioral status
    cullmodels[:, cullmodels_counter] = L

    return cullmodels, cullmodels_counter, L, Lcut_ids, rmse

# UNCOMMENT THE FOLLOWING TO DELETE RESULTS RATHER THAN JUST SETTING TO VERY LOW L
#     excludemodels1=rmse==1.23456e9
#     excludemodels1=1*excludemodels
#     excludemodels2=(np.arange(np.shape(runnumbers)[0]))
#     excludemodels=excludemodels2[excludemodels1==1]
#     for i in np.arange(np.shape(excludemodels)[0]):                                             # delete non-behavioral models
#         del runnumbers[excludemodels[i]]
#         allheads_ss_ntna = np.delete(allheads_ss_ntna, excludemodels[i],axis=0)                 # remove calculation rows for excluded models
#         allheads_ss_ytna = np.delete(allheads_ss_ytna, excludemodels[i],axis=0)
#         allheads_ss_ytya = np.delete(allheads_ss_ytya, excludemodels[i],axis=0)
#         allflows_ss_ntna = np.delete(allflows_ss_ntna, excludemodels[i],axis=0)
#         allflows_ss_ytna = np.delete(allflows_ss_ytna, excludemodels[i],axis=0)
#         allflows_ss_ytya = np.delete(allflows_ss_ytya, excludemodels[i],axis=0)
#         allleaks_ss_ntna = np.delete(allleaks_ss_ntna, excludemodels[i],axis=0)
#         allleaks_ss_ytna = np.delete(allleaks_ss_ytna, excludemodels[i],axis=0)
#         allleaks_ss_ytya = np.delete(allleaks_ss_ytya, excludemodels[i],axis=0)
#         allepts_ss_ntna = np.delete(allepts_ss_ntna, excludemodels[i],axis=0)            
#         dd = np.delete(dd, excludemodels[i],axis=0)     
#         run_params = np.delete(run_params, excludemodels[i],axis=0)                             # remove model names from list, too 

# %%
def calculateUtility(moc_limit, moc_comparison, metric):
    # set variables
    u_factor = 0.8                      # set utility variance
    u_var    = u_factor * moc_limit     # calculate variance of utility threshold
    u_range  = 2 * u_var                # calculate range of utility thresholds
    u_LL     = moc_limit - u_var        # lower threshold of utility

    # first utility
    u_norm = (metric - u_LL)/u_range

    if moc_comparison == 0:
        utility = 1 - u_norm
    if moc_comparison == 1:
        utility = u_norm
    
    utility[utility < 0] = 0
    utility[utility > 1] = 1
    
    return utility
# %%
def assess_MOCs(dict_MOC_crit, cullmodels, cullmodels_counter, allflows_ss, allheads_ss, maxdwt_ss, dd, dict_plotHolds, scenario, Nmodels):
    moc_total = np.zeros(Nmodels)
    # determine number of criteria to apply
    num_moc_criteria = len(dict_MOC_crit['time'])
    # unpack plotting records
    holdplotx      = dict_plotHolds['x']
    holdploty      = dict_plotHolds['y']
    holdplottype   = dict_plotHolds['type']
    holdleftlimit  = dict_plotHolds['l_limit']
    holdrightlimit = dict_plotHolds['r_limit']
    # loop over bases for discrimination of MOCs or (non)behavioral models
    for ii in range(num_moc_criteria):
        print('Assessing model of concern criterion:', ii)
        # unpack MOC-defining criteria
        moc_time  = dict_MOC_crit['time'][ii]; moc_basis  = dict_MOC_crit['basis'][ii]
        moc_row   = dict_MOC_crit['row'][ii] ; moc_column = dict_MOC_crit['column'][ii]
        moc_limit = dict_MOC_crit['limit'][ii]   ; moc_comparison = dict_MOC_crit['comparison'][ii]
        # establish the time sequence (scenario) for each criterion
        s = scenario[moc_time]
        metric  = np.zeros(Nmodels)

        # extract the metric value depending on the basis of MOC comparison
        if moc_basis == 0:
            metric = np.max(allflows_ss[s], axis=1)
        elif moc_basis == 1:
            metric = maxdwt_ss[s]
        elif moc_basis == 2:
            metric = allflows_ss[s][:,moc_column]
        elif moc_basis == 3:
            metric = allheads_ss[s][:, moc_row, moc_column]
        else:
            metric = dd[:, moc_row, moc_column]
        # calculate utility
        utility = calculateUtility(moc_limit, moc_comparison, metric)
        # advance counter +1
        cullmodels_counter=cullmodels_counter+1
        # store data used to check (non)behavioral status
        cullmodels[:,cullmodels_counter]= metric
        # advance counter +1
        cullmodels_counter=cullmodels_counter+1
        # store calculated utility values
        cullmodels[:,cullmodels_counter] = utility
        # advance counter +1
        cullmodels_counter=cullmodels_counter+1
        # boolean mask to identify MOCs. NOTE: default moc_comparison = 0 ... 1 signifies an MOC, 0 an other model
        cullmodels[:,cullmodels_counter] = 1 * (metric > moc_limit)                         
        if moc_comparison == 1:
            cullmodels[:,cullmodels_counter] = 1 - cullmodels[:,cullmodels_counter]
        # count number of mocs
        moc_total = moc_total + cullmodels[:,cullmodels_counter] 
        nummocs = np.sum(cullmodels[:,cullmodels_counter])
        # sort metric in decreasing order over all models
        sorted_moc_metric=np.sort(metric)[::-1]

        # save results to plot later
        plottype=1
        
        holdplottype.append(plottype)
        # store ordinal numbers from 0 to number of criteria used for asssessing models
        holdplotx.append(np.arange(Nmodels))
        # hold sorted values of the metric used for assessment, H to L
        holdploty.append(sorted_moc_metric)
        # greater than indicates behavioral
        if moc_comparison==0:
            holdleftlimit.append(nummocs)   # model zero presumably behavioral
            holdrightlimit.append(Nmodels)
        elif moc_comparison==1:
            holdleftlimit.append(0)
            holdrightlimit.append(Nmodels-nummocs)

        # advance counter +1
        cullmodels_counter = cullmodels_counter + 1
        # update record
        cullmodels[:,cullmodels_counter] = moc_total
        # # output this file to check all analyses to this point manually
        # np.savetxt(prefix + "cullmodels.csv", cullmodels, delimiter=",")

    # pack plotting information into dictionary for ease of recall
    hold_list = [holdplotx, holdploty, holdleftlimit, holdrightlimit, holdplottype]
    plot_keys = ['x', 'y', 'l_limit', 'r_limit', 'type']
    dict_plotHolds = dict(zip(plot_keys, hold_list))

    return nummocs, moc_total, cullmodels, cullmodels_counter, dict_plotHolds 

# %%
#==================M O C  B E H A V I O R================================================
def mocBehavior(runnumbers, moc_total):
    # Generate list of MOC names and indices
    # convert list of model names into an array for masking
    runnumbers_arr = np.array(runnumbers)
    # boolean masks for MOCs
    moc_mask    = np.ma.equal(moc_total, 1)
    nonmoc_mask = np.ma.equal(moc_total, 0)
    # lists of indices for (non)MOCs
    moc_idx    = np.where(moc_mask)[0].tolist()
    nonmoc_idx = np.where(nonmoc_mask)[0].tolist()
    # lists of (non)MOC names
    moc_models    = runnumbers_arr[moc_mask].tolist()
    nonmoc_models = runnumbers_arr[nonmoc_mask].tolist()

    return [moc_idx, moc_models, nonmoc_idx, nonmoc_models]

# %%
#==================M O C  S T A T I S T I C S============================================
def mocStats(mocBehavior, runnumbers):
    # unpack list of (non)MOC idx and names
    moc_idx    = mocBehavior[0]; moc_models    = mocBehavior[1]
    nonmoc_idx = mocBehavior[2]; nonmoc_models = mocBehavior[3]

    if len(nonmoc_models) > 0:
        tempvar = []
        moc_meanid = np.zeros(7); moc_varid  = np.zeros(7)
        # counter over variable parameter positions, 9 through 15
        for i in np.arange(9, 16):
            # temporary variable to hold values for a single parameter
            tempvar = np.zeros(len(moc_idx))
            counter = -1
            for j in (moc_models):
                counter += 1
                # extract parameter value from model name (str)
                tempvar[counter] = int(runnumbers[counter][i])
            # calculate the mean of the parameter values from moc models
            moc_meanid[i-9] = np.mean(tempvar)
            # calculate the STD of the parameter values from moc models
            moc_varid[i-9]  = np.std(tempvar)
        moc_meanid = np.round(moc_meanid*1000)/1000
        # repeat above process for nonMOC models
        tempvar = []
        nonmoc_meanid = np.zeros(7); nonmoc_varid = np.zeros(7)
        for i in np.arange(9, 16):
            tempvar = np.zeros(len(nonmoc_idx))
            counter = -1
            for j in (nonmoc_models):
                counter += 1
                tempvar[counter] = int(runnumbers[counter][i])
            nonmoc_meanid[i-9] = np.mean(tempvar)
            nonmoc_varid[i-9]  = np.std(tempvar)
        nonmoc_meanid = np.round(nonmoc_meanid * 1000)/1000

        moc_stats = [moc_meanid, moc_varid, nonmoc_meanid, nonmoc_varid]
        moc_keys  = ['moc_mean', 'moc_var', 'nonmoc_mean', 'nonmoc_var']
        dict_MOC_stats = dict(zip(moc_keys, moc_stats))

    else:
        print('No models of concern')

    return dict_MOC_stats

# %%
#==================L I K E L I H O O D  S T A T I S T I C S==============================
# calculate the likelihood weighted mean and standard deviation, defaults to normal mean and std if no data available
def Lstats(Lin,datain):                                                                           
    tempout = []
    # differentiate head and streamflow by dimensionality
    if np.ndim(datain) == 3:                                                        # if ndim = 3, dealing with head data
        # set up temporary array to store mean over all data
        tempmeanmatrix = np.zeros((np.shape(datain)[1],np.shape(datain)[2]))
        # set up temporary array to store standard deviation over all data
        tempstdmatrix  = np.zeros((np.shape(datain)[1],np.shape(datain)[2]))
    elif np.ndim(datain) == 2:                                                      # if ndim = 2, dealing with flow data
        # temp array to store mean over all data
        tempmeanmatrix = np.zeros((np.shape(datain)[1]))
        # temp array to store standard deviation over all data
        tempstdmatrix  = np.zeros((np.shape(datain)[1]))
    # kill the program if the data has neither 2 nor 3 dimentions
    else:
        print('problem with an input file')
        return
    # Loop through each element in array (subsurface or stream depending on datain)
    for j in np.arange(np.shape(datain)[1]):
        # treat 2 and 3 dimensional differently
        if np.ndim(datain)==3:
            # move along grid columns for each row (j)
            for k in np.arange(np.shape(datain)[2]):
                # extract head data for grid cell at row (j) and column (k) across all models
                tempvar=datain[:,j,k]
                # calculate L-weighted mean over all models at each location 
                tempmeanval=np.sum(tempvar*Lin)/np.sum(Lin)
                # record L-weighted mean for each location in j x k matrix
                tempmeanmatrix[j,k]=tempmeanval
                # repeat L-weighted mean
                meanremove=np.repeat(tempmeanval,np.shape(tempvar)[0])
                # calculate L-weighted standard deviation over all models at each location
                tempstdmatrix[j,k]=(np.sum(Lin*(tempvar-meanremove)**2)/(np.sum(Lin)*
                    (np.shape(Lin)[0]-1)/np.shape(Lin)[0]))**0.5
        # Code block to handle streamflow data
        else:
            # extract streamflow data for grid cell at column (j) across all models
            tempvar=datain[:,j]
            # calculate L-weighted mean over all models at each location
            tempmeanval=np.sum(tempvar*Lin)/np.sum(Lin)
            # record L-weighted mean for each location in 2-d matrix
            tempmeanmatrix[j]=tempmeanval
            # repeat L-weighted mean
            meanremove=np.repeat(tempmeanval,np.shape(tempvar)[0])
            # calculated L-weighted standard deviation over all models at each location
            tempstdmatrix[j]=(np.sum(Lin*(tempvar-meanremove)**2)/(np.sum(Lin)*
                    (np.shape(Lin)[0]-1)/np.shape(Lin)[0]))**0.5
    # export matric of L-weighted means over all models at each point
    tempout.append(tempmeanmatrix)
    # export matrix of L-weighted standard deviation over all models at each point
    tempout.append(tempstdmatrix)

    # clear temporary variables
    del tempmeanval
    del meanremove
    del tempvar
    del tempmeanmatrix
    del tempstdmatrix
    
    return tempout

# %%
def run_Lstats(L, allheads_ss, allflows_ss, allleaks_ss, dd, scenario, Lstat_flag, mocBehavior, Nmodels):
    bestmodels=np.argsort(-L)
    MLmodelID=bestmodels[0]

    moc_idx = mocBehavior[0]
    nonmoc_idx = mocBehavior[2]

    # Set screen to all models for all instances
    if Lstat_flag == 0:
        screen = np.arange(Nmodels)
    # When run_sections = 0, the script should only calculate Lstats for models, not MOCs. This resets screen to MOC indices
    elif Lstat_flag > 0:
        screen = moc_idx
    # Re-run this function when testing for MOCs, but the screen now looks at non-MOC models
    elif Lstat_flag < 0:
        screen = nonmoc_idx

    list_headmeans = []; list_headvar = []; list_headML = []
    list_flowmeans = []; list_flowvar = []; list_flowML = []
    list_leakmeans = []; list_leakvar = []; list_leakML = []
    # step through scenarios ('ntna', 'ytna', 'ytya')
    for s in scenario:
        # calculate the likelihood statistics across models for specified scenario (HEAD)
        tempout_head = Lstats(L[screen], allheads_ss[s][screen])
        # assign the statistics 
        temp_headmeans = tempout_head[0]          ; list_headmeans.append(temp_headmeans)
        temp_headvar   = tempout_head[1]          ; list_headvar.append(temp_headvar)
        temp_headML    = allheads_ss[s][MLmodelID]; list_headML.append(temp_headML)
        # calculate the likelihood statistics across models for specified scenario (FLOW)
        tempout_flow = Lstats(L[screen], allflows_ss[s][screen])
        # assign statistics
        temp_flowmeans = tempout_flow[0]          ; list_flowmeans.append(temp_flowmeans)
        temp_flowvar   = tempout_flow[1]          ; list_flowvar.append(temp_flowvar)
        temp_flowML    = allflows_ss[s][MLmodelID]; list_flowML.append(temp_flowML)
        # calculate the likelihood statistics across models for specified scenario (LEAK)
        tempout_leak = Lstats(L[screen], allleaks_ss[s][screen])
        # assign statistics
        temp_leakmeans = tempout_leak[0]          ; list_leakmeans.append(temp_leakmeans)
        temp_leakvar   = tempout_leak[1]          ; list_leakvar.append(temp_leakvar)
        temp_leakML    = allleaks_ss[s][MLmodelID]; list_leakML.append(temp_leakML)

    headmeans_ss = dict(zip(scenario, list_headmeans)); headvar_ss = dict(zip(scenario, list_headvar)); headML_ss = dict(zip(scenario, list_headML))
    flowmeans_ss = dict(zip(scenario, list_flowmeans)); flowvar_ss = dict(zip(scenario, list_flowvar)); flowML_ss = dict(zip(scenario, list_flowML))
    leakmeans_ss = dict(zip(scenario, list_leakmeans)); leakvar_ss = dict(zip(scenario, list_leakvar)); leakML_ss = dict(zip(scenario, list_leakML))

    # Calculate likelihood statistics for drawdown. Since it is not different for each scenario, it is calc'd once
    tempout_dd = Lstats(L, dd)
    ddmean = tempout_dd[0]
    ddvar  = tempout_dd[1]

    return [headmeans_ss, headvar_ss, headML_ss], [flowmeans_ss, flowvar_ss, flowML_ss], [leakmeans_ss, leakvar_ss, leakML_ss], [ddmean, ddvar], MLmodelID

# %%
def run_overlap(allheads_ss, allflows_ss, allleaks_ss, mocBehavior, L, scenario):
    # unpack (non)MOC indices from list
    moc_idx = mocBehavior[0]; nonmoc_idx = mocBehavior[2]
    # define (non)MOC likelihood using indices
    moc_L = L[moc_idx]; nonmoc_L = L[nonmoc_idx]
    # define list to record overlap for each scenario
    heads_overlap = []
    # step through scenarios ('ntna','ytna','ytya')
    for s in scenario:
        # set grid length from the shape of the head layer
        gridLength = np.shape(allheads_ss[s])[1]
        # create a square array to store overlap
        overlap = np.zeros((gridLength, gridLength))
        # split heads arrays by (non)MOC idx
        moc_data = allheads_ss[s][moc_idx]
        non_data = allheads_ss[s][nonmoc_idx]
        # loop through grid rows
        for i in np.arange(gridLength):
            # loop through colums along ith row
            for j in np.arange(gridLength):
                # initialize arrays for MOCs and other models
                moc_mask = np.zeros(len(moc_idx))
                non_mask = np.zeros(len(nonmoc_idx))
                # find lower maximum value between all (non)MOCs
                maxcutoff = min(max(moc_data[:,i,j]), max(non_data[:,i,j]))
                # find higher minimum value between all (non)MOCs
                mincutoff = max(min(moc_data[:,i,j]), min(non_data[:,i,j]))
                # find MOCs with predictions WITHIN overlap
                moc_mask = 1 - (1 * (moc_data[:,i,j] > maxcutoff) + 1 * (moc_data[:,i,j] < mincutoff))
                # find nonMOCs with predictions WITHIN overlap
                non_mask = 1 - (1 * (non_data[:,i,j] > maxcutoff) + 1 * (non_data[:,i,j] < mincutoff))
                # sum likelihoods of models in overlap
                overlap[i,j] = np.sum(moc_L * moc_mask) + np.sum(nonmoc_L * non_mask)
                # an overalp of 1 generally occurs because all models give the same value
                # ...a fixed head boundary or cut out corner
                overlap[overlap == 1] = np.nan
        
        heads_overlap.append(overlap)
    
    flows_overlap = []
    for s in scenario:
        gridLength = np.shape(allflows_ss[s])[1]
        overlap    = np.zeros(gridLength)
        moc_data   = allflows_ss[s][moc_idx]
        non_data   = allflows_ss[s][nonmoc_idx]

        for i in np.arange(gridLength):
            moc_mask = np.zeros(len(moc_idx))                                                                    # initialize arrays for MOCs and other models
            non_mask = np.zeros(len(nonmoc_idx))

            maxcutoff = min(max(moc_data[:,i]), max(non_data[:,i]))                                                     # find lower maximum value between all MOCs and all other models    
            mincutoff = max(min(moc_data[:,i]), min(non_data[:,i]))                                                     # find higher minimum value between all MOCs and all other models 
            moc_mask  = 1 - (1 * (moc_data[:,i] >= maxcutoff) + 1 * (moc_data[:,i] <= mincutoff))                                      # find MOCs with predictions WITHIN overlap
            non_mask  = 1 - (1 * (non_data[:,i] >= maxcutoff) + 1 * (non_data[:,i] <= mincutoff))                                                 # find other models with predictions WITHIN overlap
            overlap[i] = np.sum(moc_L * moc_mask) + np.sum(nonmoc_L * non_mask)  
        
        flows_overlap.append(overlap)
    
    leaks_overlap = []
    for s in scenario:
        gridLength = np.shape(allleaks_ss[s])[1]
        overlap    = np.zeros(gridLength)
        moc_data   = allleaks_ss[s][moc_idx]
        non_data   = allleaks_ss[s][nonmoc_idx]

        for i in np.arange(gridLength):
            moc_mask = np.zeros(len(moc_idx))                                                                    # initialize arrays for MOCs and other models
            non_mask = np.zeros(len(nonmoc_idx))

            maxcutoff = min(max(moc_data[:,i]), max(non_data[:,i]))                                                     # find lower maximum value between all MOCs and all other models    
            mincutoff = max(min(moc_data[:,i]), min(non_data[:,i]))                                                     # find higher minimum value between all MOCs and all other models 
            moc_mask  = 1 - (1 * (moc_data[:,i] >= maxcutoff) + 1 * (moc_data[:,i] <= mincutoff))                                      # find MOCs with predictions WITHIN overlap
            non_mask  = 1 - (1 * (non_data[:,i] >= maxcutoff) + 1 * (non_data[:,i] <= mincutoff))                                                 # find other models with predictions WITHIN overlap
            overlap[i] = np.sum(moc_L * moc_mask) + np.sum(nonmoc_L * non_mask)
            
        leaks_overlap.append(overlap)                                              
              
    heads_overlap = dict(zip(scenario, heads_overlap))
    flows_overlap = dict(zip(scenario, flows_overlap))
    leaks_overlap = dict(zip(scenario, leaks_overlap))

    return heads_overlap, flows_overlap, leaks_overlap

# %%
def discriminatoryIndex(nonhead_Lstats, mochead_Lstats, heads_overlap, scenario):
    # unpack likelihood statistics
    non_headmeans_ss = nonhead_Lstats[0]; non_headvar_ss = nonhead_Lstats[1]
    moc_headmeans_ss = mochead_Lstats[0]; moc_headvar_ss = mochead_Lstats[1]
    # define lists to record discriminatory information
    meandiff_s = []; sumvar_s = []; di_std_s = []; di_overlap_s = []

    for s in scenario:
        meandiff   = np.abs(non_headmeans_ss[s] - moc_headmeans_ss[s])
        sumvar     = non_headvar_ss[s] + moc_headvar_ss[s]
        sumvar[sumvar < 1e-10] = np.nan
        di_std     = meandiff/sumvar
        di_overlap = np.abs(meandiff) * (1-heads_overlap[s])

        meandiff_s.append(meandiff); sumvar_s.append(sumvar)
        di_std_s.append(di_std)    ; di_overlap_s.append(di_overlap)

    meandiff_ss = dict(zip(scenario, meandiff_s)); sumvar_ss    = dict(zip(scenario, sumvar_s))
    di_std_ss   = dict(zip(scenario, di_std_s))  ; di_overlap_ss = dict(zip(scenario, di_overlap_s)) 

    return meandiff_ss, sumvar_ss, di_std_ss, di_overlap_ss

# %%
def particleCapture(nrow, ncol, well1, well2, fNWc, strrow, L, runnumbers, allepts_ss_ntna, allepts_ss_ytna, allepts_ss_ytya):
    maxLid=np.argsort(-L)[0]

    strcapgrid=np.zeros((3,nrow,ncol))                                                                # initiate array to store starting locations of particles that end in stream
    w1capgrid=np.zeros((3,nrow,ncol))                                                                 # initiate array to store starting locations of particles that end in town well
    w2capgrid=np.zeros((3,nrow,ncol))                                                                 # initiate array to store starting locations of particles that end in ag well
    maxLw1capgrid=np.zeros((3,nrow,ncol))
    farmcappermodel=np.zeros(np.shape(runnumbers))
    streamcappermodel=np.zeros(np.shape(runnumbers))
    for k in np.arange(3):                                                                            # loop over ntna, ytna, ytya
        for i in np.arange(np.shape(runnumbers)[0]):                                                  # loop over all models in ensemble                 
            for j in np.arange(np.shape(allepts_ss_ntna)[1]):                                         # loop over all particles
                if k==0:
                    exloc=int(allepts_ss_ntna[i,j,3])                                                 # ending column   
                    eyloc=int(allepts_ss_ntna[i,j,2])                                                 # ending row
                    sxloc=int(allepts_ss_ntna[i,j,1])                                                 # starting column
                    syloc=int(allepts_ss_ntna[i,j,0])                                                 # starting row
                elif k==1:
                    exloc=int(allepts_ss_ytna[i,j,3])
                    eyloc=int(allepts_ss_ytna[i,j,2])
                    sxloc=int(allepts_ss_ytna[i,j,1])
                    syloc=int(allepts_ss_ytna[i,j,0])
                else:
                    exloc=int(allepts_ss_ytya[i,j,3])
                    eyloc=int(allepts_ss_ytya[i,j,2])
                    sxloc=int(allepts_ss_ytya[i,j,1])
                    syloc=int(allepts_ss_ytya[i,j,0])
                if exloc==well1[1] and eyloc==well1[2]:                                               # identify particles that end in well1
                    # add code to determine how many farm particles town well captures for each model                    
                    if k==2 and sxloc>=fNWc[1] and sxloc<=fNWc[1] +1:                                 # determine if sxloc and syloc for this particle originate on farm, layer 0                        
                        if syloc>=fNWc[0] and syloc<=fNWc[0] +1:
                            farmcappermodel[i]=farmcappermodel[i]+1                                   # if so, add one to tally for this model
                    w1capgrid[k,sxloc,syloc]=w1capgrid[k,sxloc,syloc]+L[i]                            # tally likelihood of model associated with particles captured from each grid cell over all models
                    if k==2 and i==maxLid:
                        maxLw1capgrid[k,sxloc,syloc]=1
                if exloc==well2[1] and eyloc==well2[2]:      
                    w2capgrid[k,sxloc,syloc]=w2capgrid[k,sxloc,syloc]+L[i]     
                if exloc==strrow:     
                    strcapgrid[k,sxloc,syloc]=strcapgrid[k,sxloc,syloc]+L[i]     
                    if k==2 and sxloc>=fNWc[1] and sxloc<=fNWc[1] +1:                                 # determine if sxloc and syloc for this particle originate on farm, layer 0                        
                        if syloc>=fNWc[0] and syloc<=fNWc[0] +1:
                            streamcappermodel[i]=streamcappermodel[i]+1                                   # if so, add one to tally for this model

    return strcapgrid, w1capgrid, w2capgrid

# %%


