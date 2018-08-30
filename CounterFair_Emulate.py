"""
Counterfactual Fairness (Kusner et al. 2017) Replication in Python 3 by Philip Ball

NB: Stan files courtesy of Matt Kusner

Options
-do_l2: Performs the replication of the L2 (Fair K) model, which can take a while depending on computing power
-save_l2: Saves the resultant models (or not) for the L2 (Fair K) model, which produces large-ish files (100s MBs)

Dependencies (shouldn't really matter as long as it's up-to-date Python >3.5):
Python 3.5.5
NumPy 1.14.3
Pandas 0.23.0
Scikit-learn 0.19.1
PyStan 2.17.1.0
StatsModels 0.9.0
"""

import pystan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()
parser.add_argument('-do_l2', type=str2bool, nargs='?', const=True, default=True, help="Perform L2 Train/Test (Warning: TAKES TIME)")
parser.add_argument('-save_l2', type=str2bool, nargs='?', const=True, default=False, help="Save L2 Train/Test Models (Warning: LARGE FILES)")
args = parser.parse_args()

# wrapper class for statsmodels linear regression (more stable than SKLearn)
class SM_LinearRegression():
    def __init__(self):
        pass
        
    def fit(self, X, y):
        N = X.shape[0]
        self.LRFit = sm.OLS(y, np.hstack([X,np.ones(N).reshape(-1,1)]),hasconst=True).fit()
        
    def predict(self,X):
        N = X.shape[0]
        return self.LRFit.predict(np.hstack([X,np.ones(N).reshape(-1,1)]))

# function to convert to a dictionary for use with STAN train-time model
def get_pystan_train_dic(pandas_df, sense_cols):
    dic_out = {}
    dic_out['N'] = len(pandas_df)
    dic_out['K'] = len(sense_cols)
    dic_out['a'] = np.array(pandas_df[sense_cols])
    dic_out['ugpa'] = list(pandas_df['UGPA'])
    dic_out['lsat'] = list(pandas_df['LSAT'].astype(int))
    dic_out['zfya'] = list(pandas_df['ZFYA'])
    return dic_out

# function to convert to a dictionary for use with STAN test-time model
def get_pystan_test_dic(fit_extract, test_dic):
    dic_out = {}
    for key in fit_extract.keys():
        if key not in ['sigma_g_Sq', 'u', 'eta_a_zfya', 'eta_u_zfya', 'lp__']:
            dic_out[key] = np.mean(fit_extract[key], axis = 0)
    
    need_list = ['N', 'K', 'a', 'ugpa', 'lsat']
    for data in need_list:
        dic_out[data] = test_dic[data]

    return dic_out

# Preprocess data for all subsequent experiments
def get_data_preprocess():
    law_data = pd.read_csv('./law_data.csv', index_col=0)
    law_data = pd.get_dummies(law_data,columns=['race'],prefix='',prefix_sep='')

    law_data['male'] = law_data['sex'].map(lambda z: 1 if z == 2 else 0)
    law_data['female'] = law_data['sex'].map(lambda z: 1 if z == 1 else 0)
    
    law_data['LSAT'] = law_data['LSAT'].apply(lambda x: int(np.round(x)))

    law_data = law_data.drop(axis=1, columns=['sex'])

    sense_cols = ['Amerindian','Asian','Black','Hispanic','Mexican','Other','Puertorican','White','male','female']

    law_train,law_test = train_test_split(law_data, random_state = 1234, test_size = 0.2);

    law_train_dic = get_pystan_train_dic(law_train, sense_cols)
    law_test_dic = get_pystan_train_dic(law_test, sense_cols)

    return law_train_dic, law_test_dic

# Get the Unfair Model predictions
def Unfair_Model_Replication(law_train_dic, law_test_dic):
    lr_unfair = SM_LinearRegression()
    lr_unfair.fit(np.hstack((law_train_dic['a'],np.array(law_train_dic['ugpa']).reshape(-1,1),np.array(law_train_dic['lsat']).reshape(-1,1))),law_train_dic['zfya'])
    
    preds = lr_unfair.predict(np.hstack((law_test_dic['a'],np.array(law_test_dic['ugpa']).reshape(-1,1),np.array(law_test_dic['lsat']).reshape(-1,1))))
    
    # Return Results:
    return preds

# Get the FTU Model predictions
def FTU_Model_Replication(law_train_dic, law_test_dic):
    lr_unaware = SM_LinearRegression()
    lr_unaware.fit(np.hstack((np.array(law_train_dic['ugpa']).reshape(-1,1),np.array(law_train_dic['lsat']).reshape(-1,1))),law_train_dic['zfya']); 

    preds = lr_unaware.predict(np.hstack((np.array(law_test_dic['ugpa']).reshape(-1,1),np.array(law_test_dic['lsat']).reshape(-1,1))))
    
    # Return Results:
    return preds

# Get the Fair K/L2 Model predictions
def L2_Model_Replication(law_train_dic, law_test_dic, save_models = False):

    check_fit = Path("./model_fit.pkl")

    if check_fit.is_file():
        print('File Found: Loading Fitted Training Model Samples...')
        if save_models:
            print('No models will be trained or saved')
        with open("model_fit.pkl", "rb") as f:
            post_samps = pickle.load(f)
    else:
        print('File Not Found: Fitting Training Model...\n')
        # Compile Model
        model = pystan.StanModel(file = './law_school_train.stan')
        print('Finished compiling model!')
        # Commence the training of the model to infer weights (500 warmup, 500 actual)
        fit = model.sampling(data = law_train_dic, iter=100, chains = 1)
        post_samps = fit.extract()
        # Save parameter posterior samples if specified
        if save_models:
            with open("model_fit.pkl", "wb") as f:
                pickle.dump(post_samps, f, protocol=-1)
            print('Saved fitted model!')

    # Retreive posterior weight samples and take means
    law_train_dic_final = get_pystan_test_dic(post_samps, law_train_dic)
    law_test_dic_final = get_pystan_test_dic(post_samps, law_test_dic)

    check_train = Path("./model_fit_train.pkl")
    
    if check_train.is_file():
        # load posterior training samples from file
        print('File Found: Loading Test Model with Train Data...')
        if save_models:
            print('No models will be trained or saved')
        with open("model_fit_train.pkl", "rb") as f:
            fit_train_samps = pickle.load(f)
    else:
        # Obtain posterior training samples from scratch
        print('File Not Found: Fitting Test Model with Train Data...\n')
        model_train = pystan.StanModel(file = './law_school_only_u.stan')
        fit_train = model_train.sampling(data = law_train_dic_final, iter=2000, chains = 1)
        fit_train_samps = fit_train.extract()
        if save_models:
            with open("model_fit_train.pkl", "wb") as f:
                pickle.dump(fit_train_samps, f, protocol=-1)
            print('Saved train samples!')
    
    train_K = np.mean(fit_train_samps['u'],axis=0).reshape(-1,1)

    check_test = Path("./model_fit_test.pkl")

    if check_test.is_file():
        # load posterior test samples from file
        print('File Found: Loading Test Model with Test Data...')
        if save_models:
            print('No models will be trained or saved')
        with open("model_fit_test.pkl", "rb") as f:
            fit_test_samps = pickle.load(f)
    else:
        # Obtain posterior test samples from scratch
        print('File Not Found: Fitting Test Model with Test Data...\n')
        model_test = pystan.StanModel(file = './law_school_only_u.stan')
        fit_test = model_test.sampling(data = law_test_dic_final, iter=2000, chains = 1)
        fit_test_samps = fit_test.extract()
        if save_models:
            with open("model_fit_test.pkl", "wb") as f:
                pickle.dump(fit_test_samps, f, protocol=-1)
            print('Saved test samples!')
    
    test_K = np.mean(fit_test_samps['u'],axis=0).reshape(-1,1)

    # Train L2 Regression
    smlr_L2 = SM_LinearRegression()
    smlr_L2.fit(train_K,law_train_dic['zfya'])

    # Predict on test
    preds = smlr_L2.predict(test_K)

    # Return Results:
    return preds

# Get the Fair All/L3 Model Predictions
def L3_Model_Replication(law_train_dic, law_test_dic):

    # abduct the epsilon_G values
    linear_eps_g = SM_LinearRegression()
    linear_eps_g.fit(np.vstack((law_train_dic['a'],law_test_dic['a'])),law_train_dic['ugpa']+law_test_dic['ugpa'])
    eps_g_train = law_train_dic['ugpa'] - linear_eps_g.predict(law_train_dic['a'])
    eps_g_test = law_test_dic['ugpa'] - linear_eps_g.predict(law_test_dic['a'])
    
    # abduct the epsilon_L values
    linear_eps_l = SM_LinearRegression()
    linear_eps_l.fit(np.vstack((law_train_dic['a'],law_test_dic['a'])),law_train_dic['lsat']+law_test_dic['lsat'])
    eps_l_train = law_train_dic['lsat'] - linear_eps_l.predict(law_train_dic['a'])
    eps_l_test = law_test_dic['lsat'] - linear_eps_l.predict(law_test_dic['a'])

    # predict on target using abducted latents
    smlr_L3 = SM_LinearRegression()
    smlr_L3.fit(np.hstack((eps_g_train.reshape(-1,1),eps_l_train.reshape(-1,1))),law_train_dic['zfya'])

    # predict on test epsilons
    preds = smlr_L3.predict(np.hstack((eps_g_test.reshape(-1,1),eps_l_test.reshape(-1,1))))

    # Return Results:
    return preds

def main():

    # Get the data, split train/test
    law_train_dic, law_test_dic = get_data_preprocess()

    # Get the predictions
    unfair_preds = Unfair_Model_Replication(law_train_dic, law_test_dic)
    ftu_preds = FTU_Model_Replication(law_train_dic, law_test_dic)
    if args.do_l2:
        l2_preds = L2_Model_Replication(law_train_dic, law_test_dic, args.save_l2)
    l3_preds = L3_Model_Replication(law_train_dic, law_test_dic)

    # Print the predictions
    print('Unfair RMSE: \t\t\t%.3f' % np.sqrt(mean_squared_error(unfair_preds,law_test_dic['zfya'])))
    print('FTU RMSE: \t\t\t%.3f' % np.sqrt(mean_squared_error(ftu_preds,law_test_dic['zfya'])))
    if args.do_l2:
        print('Level 2 (Fair K) RMSE: \t\t%.3f' % np.sqrt(mean_squared_error(l2_preds,law_test_dic['zfya'])))
    print('Level 3 (Fair Add) RMSE: \t%.3f' % np.sqrt(mean_squared_error(l3_preds,law_test_dic['zfya'])))

if __name__ == '__main__':
    main()