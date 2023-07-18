# %%
import os

import matplotlib.pyplot as plt
import numpy as np

def paretoFront(rmse, utility):
    # Pair rmse and utility values into tuples, sorted by rmse
    sorted_data = sorted(zip(rmse,utility))
    # Unzip sorted tuples into two rmse-sorted lists
    sorted_rmse, sorted_utility = zip(*sorted_data)

    # Define lists that will record pareto front values
    pareto_util = [sorted_utility[0]]
    pareto_rmse = [sorted_rmse[0]]

    for i in range(len(sorted_utility)):
        if sorted_utility[i] >= pareto_util[-1]:
            pareto_util.append(sorted_utility[i])
            pareto_rmse.append(sorted_rmse[i])
    
    return pareto_rmse, pareto_util

# %%
og_filepath = 'c:\\Users\\moral\\OneDrive\\UofA\\2022-2023\\Research\\thesisCode_morales23\\Breanna'
likelihood_filepath = '\\6.22_likelihood'

os.chdir(og_filepath+likelihood_filepath)

# %%
# ================== CREATE THREE PLOT FIGURE FOR ALL TRUTH MODELS ======================
# =======================================================================================
ten_best_flag = False

# adds error or removes it depending on above flag
downsampled_flag = False
error_flag = True

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# Select between full/downsampled error analyses
if downsampled_flag == False:
    rmse_col = 64; util_col = 68; like_col = 66; errorfile = 'Error30'; obs = 'full'
else:
    rmse_col = 16; util_col = 20; like_col = 18; errorfile = 'Error2'; obs = 'downsampled'

# pack stakeholder names/marker options into tuple
tuple_sh = ('env', 'town', 'ag')
tuple_mk = ( 'ro',   'gv', 'bd')
tuple_pa = ('ro-',  'gv-','bd-')

# rmse_col = 16; util_col = 20; like_col = 18; errorfile = 'Error2'
# rmse_col = 64; util_col = 68; like_col = 18; errorfile = 'Error30'

# create figure and select settings
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
fig.set_facecolor('whitesmoke')
# plt.suptitle('Pareto front for all stakeholder across truth models', fontsize=18)
fig.supylabel('Utility',fontweight='bold', fontsize=18)

# run through all truth models
for j, ax in enumerate(axs):
    truth_sh = tuple_sh[j]    

# run through all stakeholder perspectives for each truth model
    for i, sh in enumerate(tuple_sh):
        # Load files and define sh cullmodels depending on truth model and error
        env_cullmodels = np.loadtxt("env_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                    delimiter=",",
                                    dtype=float)
        town_cullmodels = np.loadtxt("town_{}{}-{}_cullmodels.csv".format(neg, errorfile, truth_sh),
                                    delimiter=",",
                                    dtype=float)
        ag_cullmodels = np.loadtxt("ag_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                    delimiter=",",
                                    dtype=float)
        
        # pack cullmodels into tuple
        tuple_cull = (env_cullmodels, town_cullmodels, ag_cullmodels)
        # create dictionary of stakeholder cullmodels with their names as keys
        dict_cullmodels = dict(zip(tuple_sh, tuple_cull))

        # select marker setting for data and pareto front
        mk = tuple_mk[i]; pa = tuple_pa[i]

        # pull values from dictionaries
        rmse    = dict_cullmodels[sh][:, rmse_col]
        utility = dict_cullmodels[sh][:, util_col]

        # select 10 best models (optional)
        if ten_best_flag == True:
            sorted_data = sorted(zip(rmse, utility))
            rmse, utility = zip(*sorted_data[:20])

        # Pareto front
        pareto_rmse, pareto_util = paretoFront(rmse, utility)
        # plot coordinates
        ax.plot(rmse, utility, mk, label=sh)
        # plot pareto front
        ax.plot(pareto_rmse, pareto_util, pa)
        # plot settings
        ax.set_title('{} {} truth model {}'.format(obs, truth_sh, err), fontsize=18)
        ax.grid(True); ax.set_ylim(-0.05,1.05)
        if j == 1:
            ax.legend(loc='center left', fontsize=18)
        ax.invert_xaxis(); ax.tick_params(labelsize=15)

# only one legend and x label
ax.set_xlabel('RMSE', fontweight='bold', fontsize=18) 

plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

# %%
# ============================== CUMULATIVE LIKELIHOOD RMSE/UTILITY GRAPHS ==============
# =======================================================================================
# determine if analysis includes error
error_flag = True
downsampled_flag = False

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# Select between full/downsampled error analyses
if downsampled_flag == False:
    rmse_col = 64; util_col = 68; like_col = 66; errorfile = 'Error30'; obs = 'full'
else:
    rmse_col = 16; util_col = 20; like_col = 18; errorfile = 'Error2'; obs = 'downsampled'

# pack stakeholder names/marker options into tuple
tuple_sh = ('env', 'town', 'ag')    # sh name
tuple_mk = ( 'r-',   'g-', 'b-')    # sh marker options
tuple_pa = ('ro-',  'gv-','bd-')    # pareto marker options

# create figure and select settings
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
fig.set_facecolor('whitesmoke')
# plt.suptitle('Cumulative likelihood of outcomes across truth models', fontsize=18)
fig.supylabel('Cumulative Likelihood',fontweight='bold', fontsize=18)

# run through all truth models
for j, ax in enumerate(axs):
    truth_sh = tuple_sh[j]    

# run through all stakeholder perspectives for each truth model
    for i, sh in enumerate(tuple_sh):
        # Load files and define sh cullmodels depending on truth model and error
        env_cullmodels = np.loadtxt("env_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                    delimiter=",",
                                    dtype=float)
        town_cullmodels = np.loadtxt("town_{}{}-{}_cullmodels.csv".format(neg, errorfile, truth_sh),
                                    delimiter=",",
                                    dtype=float)
        ag_cullmodels = np.loadtxt("ag_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                    delimiter=",",
                                    dtype=float)
        
        # pack cullmodels into tuple
        tuple_cull = (env_cullmodels, town_cullmodels, ag_cullmodels)
        # create dictionary of stakeholder cullmodels with their names as keys
        dict_cullmodels = dict(zip(tuple_sh, tuple_cull))

        # select marker setting for data and pareto front
        mk = tuple_mk[i]; pa = tuple_pa[i]

        # pull values from dictionaries
        likelihood = dict_cullmodels[sh][:, like_col]
        utility    = dict_cullmodels[sh][:, util_col]

        # prep data for cumulative likelihood distribution function
        sorted_indices    = np.argsort(utility)
        sorted_likelihood = likelihood[sorted_indices]
        sorted_utility    = utility[sorted_indices]

        # Calculate cumulative likelihood
        cumulative_likelihood = np.cumsum(sorted_likelihood) / np.sum(sorted_likelihood)
        # add MOC threshold limit
        x = np.ones(10) * 0.5
        y = np.linspace(0, 1, endpoint=True, num=10)

        # plot coordinates
        ax.plot(sorted_utility, cumulative_likelihood, mk, label=sh)
        ax.plot(x, y, 'm:')

        # plot settings
        ax.set_title('{} {} truth model {}'.format(obs, truth_sh, err), fontsize=18)
        ax.grid(True); ax.set_ylim(-0.05,1.05); ax.tick_params(labelsize=15)
        if j == 1:
            ax.legend(loc='center left', fontsize=18)

# only one legend and x label
ax.set_xlabel('Utility', fontweight='bold', fontsize=18) 

plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()


# %%
# ===================== ORIGINAL CODE (DON'T CHANGE) ====================================
# =======================================================================================
# set stakeholder truth model and determine if analysis includes error
truth_sh = 'town'
error_flag = True

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# Load files and define sh cullmodels depending on truth model and error
env_cullmodels = np.loadtxt("env_{}downsamp-{}_cullmodels.csv".format(neg,truth_sh),
                            delimiter=",",
                            dtype=float)
town_cullmodels = np.loadtxt("town_{}downsamp-{}_cullmodels.csv".format(neg,truth_sh),
                            delimiter=",",
                            dtype=float)
ag_cullmodels = np.loadtxt("ag_{}downsamp-{}_cullmodels.csv".format(neg,truth_sh),
                            delimiter=",",
                            dtype=float)

# pack stakeholder names/marker options into tuple
tuple_sh   = ('env', 'town', 'ag')
tuple_mk   = ( 'ro',   'gv', 'bd')

# pack cullmodels into tuple
tuple_cull = (env_cullmodels, town_cullmodels, ag_cullmodels)
dict_cullmodels = dict(zip(tuple_sh, tuple_cull))

rmse_col = 64; util_col = 68; like_col = 18

# %%
# ============================== GENERATE GRAPHS ==============================
# Identify the columns bearing the relevant information and set variables
# rmse_col = 16; util_col = 20

plt.figure(figsize=(12,8), facecolor='whitesmoke')

for i in range(3):
    sh = tuple_sh[i]
    mk = tuple_mk[i]

    # pull values from dictionaries
    rmse    = dict_cullmodels[sh][:, rmse_col]
    utility = dict_cullmodels[sh][:, util_col]
    lklhd   = dict_cullmodels[sh][:, like_col]

    # plot coordinates
    plt.plot(rmse, utility, mk, label=sh)
    # plt.plot(lklhd, utility, mk, label=sh)
    plt.xlabel('RMSE'); plt.ylabel('Utility')
    plt.title('Pareto Front for all stakeholders\n{} truth model {}'.format(truth_sh,err))
    plt.invert_xaxis()
    plt.grid(True); plt.legend()#; plt.gca().invert_yaxis()

plt.show()


# %%
'''env_MOCs = env_cullmodels[:,6]
ag_MOCs  = ag_cullmodels[:,6]
town_MOCs = town_cullmodels[:,6]
# %%
town_MOCs
# %%
townCount = 0
agCount   = 0
envCount  = 0
for i in range(len(env_MOCs)):
    
    if (env_MOCs[i] == ag_MOCs[i]) & (env_MOCs[i] != town_MOCs[i]):
        townCount +=1
    if (env_MOCs[i] == town_MOCs[i]) & (env_MOCs[i] != ag_MOCs[i]):
        agCount +=1
    if (town_MOCs[i] == ag_MOCs[i]) & (town_MOCs[i] != env_MOCs[i]):
        envCount +=1

print(townCount, 'unique Town MOCs')
print(agCount,   'unique Ag MOCs')
print(envCount,  'unique Env MOCs')'''

# %%
# ======================== LOGISTITC REGRESSION MODEL ===================================
# =======================================================================================
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Collect data
X = env_cullmodels[:,3:9]
y = env_cullmodels[:,69]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logreg.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Coefficients and intercept of the logistic regression model
coefficients = logreg.coef_
intercept = logreg.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# %%
# ========================== PLOT OBSERVATIONS PULLED FROM LOG-REG MODEL ================
# =======================================================================================
# p = 1 / (1 + exp(-(intercept + coef1 * x1 + coef2 * x2 + ...)))
# looking at MOCs
# filter models based on moc flag
moc_mask = ag_cullmodels[:, 69] == 1
moc_ag   = ag_cullmodels[moc_mask]
moc_ag_obs1 = moc_ag[:, 4]
moc_ag_obs2 = moc_ag[:, 5]

plt.hist(moc_ag_obs1, color='b', bins=200, label='ag1')
plt.hist(moc_ag_obs2, color='lightblue', bins=200, label='ag2')

moc_mask = env_cullmodels[:, 69] == 1
moc_env   = env_cullmodels[moc_mask]
moc_env_obs1 = moc_env[:, 4]
moc_env_obs2 = moc_env[:, 5]

plt.hist(moc_env_obs1, color='r', bins=200, label='env1')
plt.hist(moc_env_obs2, color='salmon', bins=200, label='env2')

moc_mask = town_cullmodels[:, 69] == 1
moc_town = town_cullmodels[moc_mask]
moc_town_obs1 = moc_town[:, 4]
moc_town_obs2 = moc_town[:, 5]

plt.hist(moc_town_obs1, color='g', bins=200, label='town1')
plt.hist(moc_town_obs2, color='lightgreen', bins=200, label='town2')

plt.legend()
plt.show()

# %%
# ============================== EXAMPLE UTILITY GRAPH ==================================
u1 = [1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0]
u2 = u1[::-1]
u3 = np.ones(10) * 0.5
x1 = np.linspace(0, 5, endpoint=True, num=10)

plt.figure(facecolor='whitesmoke')
plt.title('Utility of Predicted Outcomes')
plt.plot(x1, u1, 'go-', label='greater_than')
plt.plot(x1, u2, 'bv-', label='less_than')
plt.plot(x1, u3, 'r:', label='MOC_threshold')
plt.legend(loc='center right'); plt.grid(True); 
plt.ylabel('Utility', fontweight='bold'); plt.xlabel('Metric', fontweight='bold')
plt.plot()

# %%
# ============================= STAKEHOLDER UTILITY GRAPHS ==============================
# =======================================================================================
moc_basis_list = [2,4,3]
moc_limit_list = [75,1,70]
moc_compa_list = [1,0,1]
sh_list = ['env', 'town', 'ag']
mk_list = ['ro-',  'gv-','bd-']
metric_list = ['streamflow (m3/day)', 'drawdown (m)', 'hydraulic head (m)']

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.set_facecolor('whitesmoke')
# plt.suptitle('Stakeholder utility functions', fontsize=16)
fig.supylabel('Utility',fontweight='bold', fontsize=14)

# run through all truth models
for i, ax in enumerate(axs):
    moc_basis = moc_basis_list[i]
    moc_limit = moc_limit_list[i]
    moc_compa = moc_compa_list[i]
    metric    = metric_list[i]

    sh = sh_list[i]; mk = mk_list[i]

    if moc_basis == 2:
        u_factor = 0.8               # set utility variance
    elif moc_basis == 3:
        u_factor = 0.1
    elif moc_basis == 4:
        u_factor = 0.9
    else:
        u_factor = 0.8             # throwaway value because I hope I don't use 0 or 1 bases
    u_var    = u_factor * moc_limit     # calculate variance of utility threshold
    u_range  = 2 * u_var                # calculate range of utility thresholds
    u_LL     = moc_limit - u_var        # lower threshold of utility


    if moc_basis == 3:
        x = np.linspace(0.8, 1.2, num=20) * moc_limit
    else:
        x = np.linspace(0,2, num = 20) * moc_limit

    u2 = np.ones(20) * 0.5

    # first utility
    u_norm = (x - u_LL)/u_range

    if moc_compa == 0:
        utility = 1 - u_norm
    if moc_compa == 1:
        utility = u_norm

    utility[utility < 0] = 0
    utility[utility > 1] = 1

    ax.plot(x, utility, mk, label=sh)
    ax.plot(x, u2, 'm:')
    ax.set_xlabel(metric)
    ax.set_title(sh)
    ax.grid(True)
plt.tight_layout()
plt.show()
# %%
# ======================= HISTOGRAMS OF PREDICTION OF INTEREST ==========================
# =======================================================================================
# determine if analysis includes error
error_flag = True
downsampled_flag = False

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# Select between full/downsampled error analyses
if downsampled_flag == False:
    rmse_col = 64; util_col = 68; like_col = 66; errorfile = 'Error30'; obs = 'full'
else:
    rmse_col = 16; util_col = 20; like_col = 18; errorfile = 'Error2'; obs = 'downsampled'

# pack stakeholder names/marker options into tuple
tuple_sh = ('env', 'town', 'ag')    # sh name
tuple_mk = ( 'r',   'g', 'b')    # sh marker options
tuple_ob = ('streamflow (m3/day)',  'drawdown (m)','hydraulic head (m)')    # pareto marker options

metric_col = 67
# create figure and select settings
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
fig.set_facecolor('whitesmoke')
plt.suptitle('Frequency distribution of model RMSE', fontsize=18)
fig.supylabel('Count',fontweight='bold', fontsize=14)

# run through all stakeholder perspectives for each truth model
for i, ax in enumerate(axs):
    sh = tuple_sh[i]

    # Load files and define sh cullmodels depending on truth model and error
    env_cullmodels = np.loadtxt("env_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                delimiter=",",
                                dtype=float)
    town_cullmodels = np.loadtxt("town_{}{}-{}_cullmodels.csv".format(neg, errorfile, truth_sh),
                                delimiter=",",
                                dtype=float)
    ag_cullmodels = np.loadtxt("ag_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                delimiter=",",
                                dtype=float)
    
    # pack cullmodels into tuple
    tuple_cull = (env_cullmodels, town_cullmodels, ag_cullmodels)
    # create dictionary of stakeholder cullmodels with their names as keys
    dict_cullmodels = dict(zip(tuple_sh, tuple_cull))

    # select marker setting for data and pareto front
    mk = tuple_mk[i]; ob = tuple_ob[i]

    # pull values from dictionaries
    metric = dict_cullmodels[sh][:, util_col]

    # add MOC threshold limit
    # x = np.ones(10) * 0.5
    # y = np.linspace(0, 1, endpoint=True, num=10)

    # plot coordinates
    ax.hist(metric, color = mk, bins = 100, label=ob)

    # plot settings
    ax.set_title('{}\'s prediction of interest'.format(sh))
    ax.grid(True); ax.tick_params(labelsize=15);# ax.legend()

    # prediction stats
    print(sh, ' avg. value:', np.round(np.mean(metric), 2))
    print(sh, ' median value:', np.round(np.median(metric), 2))
    print(sh, ' standard deviation:', np.round(np.std(metric), 2))

# only one legend and x label
ax.set_xlabel('Utility', fontweight='bold', fontsize=14) 

plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
# %%
# ========================== COUNT MODEL BEHAVIORS ======================================
# =======================================================================================
# determine if analysis includes error
error_flag = True
downsampled_flag = False

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# Select between full/downsampled error analyses
if downsampled_flag == False:
    rmse_col = 64; util_col = 68; like_col = 66; moc_col = 70
    errorfile = 'Error30'; obs = 'full'
else:
    rmse_col = 16; util_col = 20; like_col = 18; moc_col = 19
    errorfile = 'Error2'; obs = 'downsampled'

# pack stakeholder names/marker options into tuple
tuple_sh = ('env', 'town', 'ag')    # sh name


# run through all stakeholder perspectives for each truth model
for i, ax in enumerate(axs):
    sh = tuple_sh[i]

    # Load files and define sh cullmodels depending on truth model and error
    env_cullmodels = np.loadtxt("env_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                delimiter=",",
                                dtype=float)
    town_cullmodels = np.loadtxt("town_{}{}-{}_cullmodels.csv".format(neg, errorfile, truth_sh),
                                delimiter=",",
                                dtype=float)
    ag_cullmodels = np.loadtxt("ag_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                delimiter=",",
                                dtype=float)
    
    # pack cullmodels into tuple
    tuple_cull = (env_cullmodels, town_cullmodels, ag_cullmodels)
    # create dictionary of stakeholder cullmodels with their names as keys
    dict_cullmodels = dict(zip(tuple_sh, tuple_cull))

    # select marker setting for data and pareto front
    mk = tuple_mk[i]; ob = tuple_ob[i]

    # pull values from dictionaries
    moc = dict_cullmodels[sh][:, moc_col]

    # sum metrics
    moc_count = int(np.sum(moc))
    behavioralModels    = len(moc)
    nonBehavioralModels = 500 - behavioralModels
    
    print('====== {}\'s perspective ====='.format(sh))
    print('Models of Concern: {}'.format(moc_count))
print('*****************************')
print('Behavioral models: {}'.format(behavioralModels))
print('Nonbehavioral models: {}'.format(nonBehavioralModels))
# %%
# ======================= HISTOGRAMS OF RMSE ==========================
# =======================================================================================
# determine if analysis includes error
error_flag = False
downsampled_flag = True

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# Select between full/downsampled error analyses
if downsampled_flag == False:
    rmse_col = 64; util_col = 68; like_col = 66; errorfile = 'Error30'; obs = 'full'
else:
    rmse_col = 16; util_col = 20; like_col = 18; errorfile = 'Error2'; obs = 'downsampled'

# pack stakeholder names/marker options into tuple
tuple_sh = ('env', 'town', 'ag')    # sh name
tuple_mk = ( 'r',   'g', 'b')    # sh marker options
tuple_ob = ('streamflow (m3/day)',  'drawdown (m)','hydraulic head (m)')    # pareto marker options

metric_col = 67
# create figure and select settings
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
fig.set_facecolor('whitesmoke')
# plt.suptitle('Frequency distribution of model RMSE', fontsize=18)
fig.supylabel('Count',fontweight='bold', fontsize=18)

# run through all stakeholder perspectives for each truth model
for i, ax in enumerate(axs):
    truth_sh = tuple_sh[i]

    # Load files and define sh cullmodels depending on truth model and error
    env_cullmodels = np.loadtxt("env_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                delimiter=",",
                                dtype=float)
    town_cullmodels = np.loadtxt("town_{}{}-{}_cullmodels.csv".format(neg, errorfile, truth_sh),
                                delimiter=",",
                                dtype=float)
    ag_cullmodels = np.loadtxt("ag_{}{}-{}_cullmodels.csv".format(neg,errorfile,truth_sh),
                                delimiter=",",
                                dtype=float)
    
    # pack cullmodels into tuple
    tuple_cull = (env_cullmodels, town_cullmodels, ag_cullmodels)
    # create dictionary of stakeholder cullmodels with their names as keys
    dict_cullmodels = dict(zip(tuple_sh, tuple_cull))
    # pull values from dictionaries
    rmse = dict_cullmodels[sh][:, rmse_col]
    # plot coordinates
    ax.hist(rmse, color = 'grey', bins = 100, edgecolor='k')
    # plot settings
    ax.set_title('{} {} truth model {}'.format(obs, truth_sh, err), fontsize=18)
    ax.grid(True); ax.tick_params(labelsize=16);# ax.legend()

# only one legend and x label
ax.set_xlabel('RMSE', fontweight='bold', fontsize=18) 

plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
# %%
