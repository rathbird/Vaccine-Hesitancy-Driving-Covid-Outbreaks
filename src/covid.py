import numpy as np
import pandas as pd
import scipy.stats as scs
from numpy import mean, std
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss, make_scorer, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict


covid = pd.read_csv('./data/full_feature_covid.csv')
covid = covid.set_index('GEOID')

#clean up nans
covid = covid.fillna(0)

#break out categorical data
covid = pd.get_dummies(covid, columns=['DominantReligion','Region'])

#save georegion for later
geocols = ['Geographical_Point','Target_PctChg']
geo = covid[geocols]

#Drop categorical columns that won't help with our model
cols = ['County', 'State','Geographical_Point']
covid = covid.drop(cols, axis = 1)

#normalize data
covid['Target_PctChg'] = covid['Target_PctChg']/10
covid['Unempl_Rate'] = covid['Unempl_Rate']/100
covid['Median_HHI'] = covid['Median_HHI']/covid['Median_HHI'].sum()
covid['DensPerSqMile'] = covid['DensPerSqMile']/covid['DensPerSqMile'].sum()
covid['Avg_HH_Size'] = covid['Avg_HH_Size']/10
covid['Avg_Fam_Size'] = covid['Avg_Fam_Size']/10
covid.mean(axis=0)

covid.head()

X = covid.copy()
y = X.pop('Target_PctChg')

X_train, X_test, y_train, y_test = train_test_split(X.astype(float), y, test_size=0.2, random_state=1) 

#mean is the baseline for all models
mean_arr = np.zeros(y_test.shape)
mean_arr[mean_arr==0] = y.mean()
y_hat_mean = mean_arr

#linear regression OLS
def summary_model(X, y, label='scatter'):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    summary = model.summary()
    return summary

def plot_model(X, y, label='Residual Plot'):
    model = sm.OLS(y, X).fit()
    student_resids = model.outlier_test()['student_resid']
    y_hats = model.predict(X)

    plt.scatter(y_hats, student_resids, alpha = .35, label=label)
    plt.legend()
    plt.show()

def learning_curves(X_train, y_train, X_test, y_test):
    
    fig = plt.figure(figsize=(10,8))
    # Generate 40 evenly spaced numbers (rounded to nearest integer) over a specified interval 1 to 354
    datapoints = np.rint(np.linspace(1, len(X_train), 40)).astype(int)
    #initialise array of shape (40,)
    train_err = np.zeros(len(datapoints))
    test_err = np.zeros(len(datapoints))
 
    # Create 6 different models based on max_depth
    for k, depth in enumerate(range(1,7)):
        for i, s in enumerate(datapoints):
            reg = DecisionTreeRegressor(max_depth = depth) #increasing depth
            # Iteratively increase training set size
            reg.fit(X_train[:s], y_train[:s])
            # MSE for training and test sets of increasing size
            train_err[i] = mean_squared_error(y_train[:s], reg.predict(X_train[:s]))
            test_err[i] = mean_squared_error(y_test, reg.predict(X_test))

        print('Train_error: {}'.format(train_err[-1]))
        print('Test_error: {}'.format(test_err[-1]))
 
        # Subplot learning curves
        sub = fig.add_subplot(3, 3, k+1)
        sub.plot(datapoints, test_err, lw = 1, label = 'Testing Error')
        sub.plot(datapoints, train_err, lw = 1, label = 'Training Error')
        sub.legend()
        sub.set_title('DT Max Depth = %s'%(depth))
        sub.set_xlabel('No. of Data Points in Training Set')
        sub.set_ylabel('Total Error')
        sub.set_xlim([0, len(X_train)])
        
        fig.suptitle('Decision Tree Regressor Learning Curves', fontsize=18, y=1.03)
        fig.tight_layout()
        fig.show()

def bootstrap_confidence_interval(data, function, alpha=0.05, n_bootstraps=1000):
    '''return a the confidence interval for a function of data using bootstrapping'''
    medians = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, len(data))
        medians.append(function(bootstrap_sample))
    return (np.percentile(medians, 100*(alpha/2.)),
            np.percentile(medians, 100*(1-alpha/2.))), medians

def gridsearch(x, y):
        # Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]
	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
		'max_features': max_features,
		'max_depth': max_depth,
		'min_samples_split': min_samples_split,
		'min_samples_leaf': min_samples_leaf,
		'bootstrap': bootstrap}

	#print(random_grid)

	# Use the random grid to search for best hyperparameters
	# First create the base model to tune
	rf = RandomForestRegressor()
	# Random search of parameters, using 3 fold cross validation, 
	# search across 100 different combinations, and use all available cores
	rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
	# Fit the random search model
	rf_random.fit(x, y)

	return rf_random.best_params_


    
#linear regression
summ = summary_model(X_train,y_train)

#linear regression
ols = linear_model.LinearRegression()
model = ols.fit(X_train, y_train)
y_hat_ols = model.predict(X_test)
mse_ols = mean_squared_error(y_hat_ols, y_test)

#decision tree
dt = DecisionTreeRegressor(max_depth=6)
dt.fit(X_train, y_train)
y_hat_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_hat_dt, y_test)

#random Forest - base model
model = RandomForestRegressor(n_estimators=100, bootstrap=True, max_features='sqrt')
model.fit(X_train, y_train)
# Actual class predictions
rf_predictions = model.predict(X_test)
mse_rf_base = mean_squared_error(rf_predictions, y_test)

#random forest - best model based on 
best_model = RandomForestRegressor(n_estimators=800, min_samples_split=10, min_samples_leaf=2, max_features='sqrt', max_depth=20, bootstrap=False)
best_model.fit(X_train, y_train)

# Actual class predictions
rf_predictions = best_model.predict(X_test)
mse_rf_best = mean_squared_error(rf_predictions, y_test)

#Predict the entire data set
all_predictions = best_model.predict(X)
predicted_rate_of_change = np.array(all_predictions*10)

#Plot geographic info
geo.info()
geo['NewPrediction'] = predicted_rate_of_change
geo.describe()

#Feature importance based on random forest
# manual shuffle 
rf = RandomForestRegressor()
scores = defaultdict(list)

names = X.columns
 
rf = RandomForestRegressor()
scores = defaultdict(list)
 
# crossvalidate the scores on a number of 
# different random splits of the data
splitter = ShuffleSplit(100, test_size=.3)

for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y.values[train_idx], y.values[test_idx]
    rf.fit(X_train, y_train)
    acc = r2_score(y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)

score_series = pd.DataFrame(scores).mean()
scores = pd.DataFrame({'Mean Decrease Accuracy' : score_series})
scores.sort_values(by='Mean Decrease Accuracy').plot(kind='barh', figsize = (14,10))

#based on feature importance, compute correlations of significant predictors
for col in cols:
    print('{} {}'.format(col,compute_correlation(covid[col],y)))
    
for col in cols:
    print('{} {}'.format(col,spearmanr(covid[col],y)))
    

