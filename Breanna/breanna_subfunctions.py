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

stakeholder='env'                                                                                      # choose from ag/town/env
stage='combine'                                                                                        # choose from random/reseed/combined/afterdata
prefix = stakeholder + '_' + stage + '_'      #generates filename prefix

def output_directory():     # use this to define where your output files reside
    os.chdir('c:\\Users\\moral\\OneDrive\\UofA\\2022-2023\\Research\\thesisCode_morales23\\Breanna\\100model_output')       # 244 model rerun
 
def figure_directory():     # use this to define where to put your figure files 
     os.chdir('c:\\Users\\moral\\OneDrive\\UofA\\2022-2023\\Research\\thesisCode_morales23\\Breanna\\100model_figures')

def comparison_directory():
    os.chdir('c:\\Users\\moral\\OneDrive\\UofA\\2022-2023\\Research\\thesisCode_morales23\\Breanna\\100model_likelihood')
    
displaycolumn=40                                                                                       # column for head and drawdown analysis
displayrow=30                                                                                         # row for head and drawdown analysis
strdisplaycolumn=35                                                                                   # column for streamflow analysis

minmismatch=0.05                                                                                      # don't consider mismatches that are less than this value - avoids unrealistically high likelihoods
                                                                                                      # define criteria to meet to qualify as behavioral ... if a model PASSES this test it is behavioral
in_time_sequence =       [1,1,1]                                                                      # 0=ntna, 1=ytna, 2=ytya ... enter a number for each criterion applied
in_basis_sequence =      [0,0,1]                                                                      # see list below
in_comparison_sequence = [1,0,1]                                                                      # 0 = greater than, 1 = less than
in_limit_sequence =      [2000,10,3]                                                                  # value defining behavioral response
in_column_sequence =     [15,15,15]                                                                   # column of observation point for basis 2 or 3  - must have a value for every criterion, even if not used
in_row_sequence =        [15,15,15]                                                                   # row of observation point for basis 2 or 3 - must have a value for every criterion, even if not used
                  
behavioral_criteria = [in_time_sequence, in_basis_sequence, in_comparison_sequence, in_limit_sequence, in_column_sequence, in_row_sequence]
comparison_directory()
np.save(prefix + 'behavioral_criteria',behavioral_criteria) #save behavioral criteria to a file
output_directory()

define_mocs = True                                                                                    # define criteria to meet to qualify as an MOC ... if a model PASSES this test it is a model of concern (MOC)
if stakeholder=='town':
    moc_time_sequence =       [2]                                                                       # 0=ntna, 1=ytna, 2=ytya ... enter a number for each criterion applied
    moc_basis_sequence =      [4]                                                                       # see list below
    moc_comparison_sequence = [0]                                                                       # 0 = greater than, 1 = less than
    moc_limit_sequence =      [0.5]                                                                   # value defining behavioral response
    moc_column_sequence =     [37]                                                                     # column of observation point for basis 2 or 3  - must have a value for every criterion, even if not used
    moc_row_sequence =        [20]                                                                     # row of observation point for basis 2 or 3 - must have a value for every criterion, even if not used
elif stakeholder=='ag':
    moc_time_sequence =       [2]                                                                       # 0=ntna, 1=ytna, 2=ytya ... enter a number for each criterion applied
    moc_basis_sequence =      [3]                                                                       # see list below
    moc_comparison_sequence = [1]                                                                       # 0 = greater than, 1 = less than
    moc_limit_sequence =      [68]                                                                   # value defining behavioral response
    moc_column_sequence =     [13]                                                                     # column of observation point for basis 2 or 3  - must have a value for every criterion, even if not used
    moc_row_sequence =        [11]                                                                     # row of observation point for basis 2 or 3 - must have a value for every criterion, even if not used
elif stakeholder=='env':
    moc_time_sequence =       [2]                                                                       # 0=ntna, 1=ytna, 2=ytya ... enter a number for each criterion applied
    moc_basis_sequence =      [2]                                                                       # see list below
    moc_comparison_sequence = [1]                                                                       # 0 = greater than, 1 = less than
    moc_limit_sequence =      [50]                                                                   # value defining behavioral response
    moc_column_sequence =     [38]                                                                     # column of observation point for basis 2 or 3  - must have a value for every criterion, even if not used
    moc_row_sequence =        [25]                                                                     # row of observation point for basis 2 or 3 - must have a value for every criterion, even if not used
    
# we should add the ability to have an MOC criterion in other than layer 1!!
# add criterion for flow reduction

#   BASES FOR DETERMINATION OF BEHAVIORAL MODEL
#       0 = max streamflow along stream
#       1 = min groundwater depth over first layer
#       2 = streamflow at specified location
#       3 = head at specified location
#       4 = drawdown at specified location
                                                                                                      # use empty brackets if no data available, each value must contain the same number of inputs, separate multiple data points by commas
moc_criteria = [moc_time_sequence, moc_basis_sequence, moc_comparison_sequence, moc_limit_sequence, moc_column_sequence, moc_row_sequence]
comparison_directory()
np.save(prefix + 'moc_criteria',moc_criteria) #save moc criteria to a file
output_directory()

read_true_data=True                                                                                   # True to read in the observations from truth_heads_ss_ytna.npy and truth_flows_ss_ytna.npy
if read_true_data==True:
    output_directory()
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
comparison_directory()               #change directory to current stakeholder and stage
np.save(prefix + 'data_layer', data_layer_sequence)
np.save(prefix + 'data_row', data_row_sequence)
np.save(prefix + 'data_column', data_column_sequence)
output_directory()

# use the following to control which analyses are completed - may be useful when running partial analyses for many models
run_sections=3

# 0 = through identifying behavioral models and calculating model likelihoods
# 1 = identify models of concern
# 2 - calculate discriminatory index
# 3 - particle capture

# %%
#===========READINMODELS FUNCTION==================================================================
def readInModels():
    output_directory(); mydir = os.getcwd()             # set the current directory to output folder
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

    for s in scenario:
        for i in range(Nmodels):
            leaks=np.load(runnumbers[i]+'_strleak_ss_'+s+'.npy')
            leak = []
            for tup in range(len(leaks)):
                leak.append(leaks[tup][1])
            allleaks_ss[s][i][:][:] = leak

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
def nonbehavioralModels(runnumbers, allheads_ss, maxdwt_ss, allflows_ss, dd):
    global num_criteria
    scenario = ('ntna', 'ytna', 'ytya')


    Nmodels = np.shape(runnumbers)[0]

    cullmodels=np.zeros((Nmodels,50))                   # store values used to assess (non)behavioural and likelihood to check process
    cullmodels_counter=-1

    # determine number of criteria to that have been applied
    if 'in_time_sequence' in globals():
        num_criteria=np.shape(in_time_sequence)[0]

    if num_criteria>0:
        print('Assessing (non)behavioral criteria')

        # loop over bases for discrimination of models of concern or (non)behavioral models
        for ii in np.arange(num_criteria):          
            cullmodels_counter += 1
            print('Assessing criterion',ii)

            # establish the time sequence (scenario) for each criterion
            in_time=in_time_sequence[ii]
            s = scenario[in_time]           # set value for scenario ('ntna','ytna','ytya')

            # set criteria values for basis for consideration, greater/less than, and value limit
            in_basis=in_basis_sequence[ii] 
            in_comparison=in_comparison_sequence[ii]
            in_limit=in_limit_sequence[ii]

            # set column and row values for observation
            in_column=in_column_sequence[ii]
            in_row=in_row_sequence[ii]

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
def useTrueData(dict_L_criteria):
    # set up variable values
    Ncomparisons = len(dict_L_criteria['time'])     # determine number of comparisons
    data_basis   = dict_L_criteria['basis']         # basis for comparison (0: streamflow, 1: head)
    data_row     = dict_L_criteria['row']           # cell row for comparison value 
    data_column  = dict_L_criteria['column']        # cell column for comparison value
    # switch to output directory to interact with truth model files
    output_directory()
    # set truth model values to objects
    trueheads_ss_ytna=np.load('truth_heads_ss_ytna.npy')[0][:][:]           # load truth heads file
    trueflows_ss_ytna=np.load('truth_strflow_ss_ytna.npy')[:]               # load truth flows file 
    # create awwary to record truth values                                      
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
    
    else:
        rmse = assess_nonBehavioral(dict_B_criteria, rmse, cullmodels)

        # this portion of code is to specify Likelihoods of models based on their mismatch.
        # in Ty's code, there is an else statement that bypasses this specified likelihoods, but still
        # sets nonbehavioral models to incredibly high rmse. 
        # First, make a function that handles nonbehavioral models and stack it inside of the likelihood function.

    sorted_L_behavioral=np.sort(L)[::-1]

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

    return behavioral_idx, behavioral_models, nonbehavioral_idx, nonbehavioral_models

# %%
