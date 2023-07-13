# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
og_filepath = 'c:\\Users\\moral\\OneDrive\\UofA\\2022-2023\\Research\\thesisCode_morales23\\Breanna'
likelihood_filepath = '\\6.22_likelihood'

os.chdir(og_filepath+likelihood_filepath)

# determine if analysis includes error
error_flag = False

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# pack stakeholder names/marker options into tuple
tuple_sh   = ('env', 'town', 'ag')
tuple_mk   = ( 'ro',   'gv', 'bd')
rmse_col = 16; util_col = 20; like_col = 18

# create figure and select settings
fig, axs = plt.subplots(3, 1, figsize=(16, 20))
fig.set_facecolor('whitesmoke')
plt.suptitle('Pareto front for all stakeholder across truth models', fontsize=20)
fig.supylabel('Utility',fontweight='bold', fontsize=14)

# run through all truth models
for j, ax in enumerate(axs):
    truth_sh = tuple_sh[j]    

# run through all stakeholder perspectives for each truth model
    for i, sh in enumerate(tuple_sh):
        # Load files and define sh cullmodels depending on truth model and error
        env_cullmodels = np.loadtxt("env_{}Error2-{}_cullmodels.csv".format(neg,truth_sh),
                                    delimiter=",",
                                    dtype=float)
        town_cullmodels = np.loadtxt("town_{}Error2-{}_cullmodels.csv".format(neg,truth_sh),
                                    delimiter=",",
                                    dtype=float)
        ag_cullmodels = np.loadtxt("ag_{}Error2-{}_cullmodels.csv".format(neg,truth_sh),
                                    delimiter=",",
                                    dtype=float)
        # pack cullmodels into tuple
        tuple_cull = (env_cullmodels, town_cullmodels, ag_cullmodels)
        # create dictionary of stakeholder cullmodels with their names as keys
        dict_cullmodels = dict(zip(tuple_sh, tuple_cull))

        # select marker setting
        mk = tuple_mk[i]

        # pull values from dictionaries
        rmse    = dict_cullmodels[sh][:, rmse_col]
        utility = dict_cullmodels[sh][:, util_col]
        lklhd   = dict_cullmodels[sh][:, like_col]

        # plot coordinates
        ax.plot(rmse, utility, mk, label=sh)
        # plt.plot(lklhd, utility, mk, label=sh)
        # ax.set_ylabel('Utility')
        ax.set_title('{} truth model {}'.format(truth_sh, err))
        ax.grid(True)

# only one legend and x label
ax.legend(); ax.set_xlabel('RMSE', fontweight='bold', fontsize=14) 

plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
        

# %%
# ===================== ORIGINAL CODE (DON'T CHANGE) ====================================
# set stakeholder truth model and determine if analysis includes error
truth_sh = 'env'
error_flag = False

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# Load files and define sh cullmodels depending on truth model and error
env_cullmodels = np.loadtxt("env_{}Error2-{}_cullmodels.csv".format(neg,truth_sh),
                            delimiter=",",
                            dtype=float)
town_cullmodels = np.loadtxt("town_{}Error2-{}_cullmodels.csv".format(neg,truth_sh),
                            delimiter=",",
                            dtype=float)
ag_cullmodels = np.loadtxt("ag_{}Error2-{}_cullmodels.csv".format(neg,truth_sh),
                            delimiter=",",
                            dtype=float)

# pack cullmodels into tuple
tuple_cull = (env_cullmodels, town_cullmodels, ag_cullmodels)

# pack stakeholder names/marker options into tuple
tuple_sh   = ('env', 'town', 'ag')
tuple_mk   = ( 'ro',   'gv', 'bd')
rmse_col = 16; util_col = 20; like_col = 18

# %%
# ============================== GENERATE GRAPHS ==============================
# Identify the columns bearing the relevant information and set variables
rmse_col = 16; util_col = 20

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
    plt.grid(True); plt.legend()

plt.show()

# %%
# ========================= GRAPHS WITH SORTED UTILITY AND RMSE ===============
plt.figure(figsize=(12,8), facecolor='whitesmoke')

for i in range(3):
    sh = tuple_sh[i]
    mk = tuple_mk[i]

    # pull values from dictionaries
    rmse    = dict_cullmodels[sh][:, rmse_col] #  rmse    = np.sort(rmse)
    utility = dict_cullmodels[sh][:, util_col] # ; utility = np.sort(utility)

    # plot coordinates
    plt.plot(rmse, utility, mk, label=sh)
    plt.xlabel('RMSE'); plt.ylabel('Utility')
    plt.title('Pareto Front for all stakeholders\n{} truth model {}'.format(truth_sh,err))
    plt.grid(True); plt.legend()

plt.show()
# %%
env_MOCs = env_cullmodels[:,6]
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
print(envCount,  'unique Env MOCs')
# %%
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 2*np.pi, 100)
datasets = [
    [np.sin(x), np.sin(2*x), np.sin(3*x)],
    [np.cos(x), np.cos(2*x), np.cos(3*x)],
    [np.tan(x), np.tan(2*x), np.tan(3*x)]
]

# Create the figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# Iterate over each subplot and plot the datasets
for i, ax in enumerate(axs):
    for dataset in datasets[i]:
        ax.plot(x, dataset)

    # Set labels and title for each subplot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Plot {}'.format(i+1))

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()

# %%
