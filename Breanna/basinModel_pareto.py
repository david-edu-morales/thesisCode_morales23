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
# determine if analysis includes error
error_flag = True

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# pack stakeholder names/marker options into tuple
tuple_sh = ('env', 'town', 'ag')
tuple_mk = ( 'ro',   'gv', 'bd')
tuple_pa = ('ro-',  'gv-','bd-')

# rmse_col = 16; util_col = 20; like_col = 18; errorfile = 'Error2'
rmse_col = 64; util_col = 68; like_col = 18; errorfile = 'Error30'

# create figure and select settings
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
fig.set_facecolor('whitesmoke')
plt.suptitle('Pareto front for all stakeholder across truth models', fontsize=18)
fig.supylabel('Utility',fontweight='bold', fontsize=14)

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

        # select marker setting
        mk = tuple_mk[i]; pa = tuple_pa[i]

        # pull values from dictionaries
        rmse    = dict_cullmodels[sh][:, rmse_col]
        utility = dict_cullmodels[sh][:, util_col]
        lklhd   = dict_cullmodels[sh][:, like_col]

        # Pareto front
        pareto_rmse, pareto_util = paretoFront(rmse, utility)

        # plot coordinates
        ax.plot(rmse, utility, mk, label=sh)
        ax.plot(pareto_rmse, pareto_util, pa)
        # plt.plot(lklhd, utility, mk, label=sh)
        # ax.set_ylabel('Utility')
        ax.set_title('{} truth model {}'.format(truth_sh, err))
        ax.grid(True)
        if j == 1:
            ax.legend(loc='center left', fontsize=14)
        ax.invert_xaxis()

# only one legend and x label
ax.set_xlabel('RMSE', fontweight='bold', fontsize=14) 

plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
        

# %%
# ===================== ORIGINAL CODE (DON'T CHANGE) ====================================
# =======================================================================================
# set stakeholder truth model and determine if analysis includes error
truth_sh = 'env'
error_flag = True

# adds error or removes it depending on above flag
if error_flag == True:
    neg = ''; err = 'w/ error'
else:
    neg = 'no'; err = ''

# Load files and define sh cullmodels depending on truth model and error
env_cullmodels = np.loadtxt("env_{}Error30-{}_cullmodels.csv".format(neg,truth_sh),
                            delimiter=",",
                            dtype=float)
town_cullmodels = np.loadtxt("town_{}Error30-{}_cullmodels.csv".format(neg,truth_sh),
                            delimiter=",",
                            dtype=float)
ag_cullmodels = np.loadtxt("ag_{}Error30-{}_cullmodels.csv".format(neg,truth_sh),
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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Sample data
X = env_cullmodels[:,3:9]
y = env_cullmodels[:,69]

# Convert data to NumPy arrays
# X = np.array(observations)  # Observations as the independent variables
# y = np.array(utility)       # Utility as the dependent variable

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
# Logistic regression prediction use:
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
rmse = env_cullmodels[:, 64]
utility = env_cullmodels[:, 68]

sorted_data = sorted(zip(rmse,utility))
sorted_rmse, sorted_utility = zip(*sorted_data)

pareto_util = [sorted_utility[0]]
pareto_rmse = [sorted_rmse[0]]

for i in range(len(sorted_utility)):
    if sorted_utility[i] <= pareto_util[-1]:
        pareto_util.append(sorted_utility[i])
        pareto_rmse.append(sorted_rmse[i])
plt.figure(figsize=(12,5))
plt.plot(pareto_rmse, pareto_util, 'ro-')

# %%



