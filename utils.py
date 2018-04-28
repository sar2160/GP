import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AR

def preprocess(df, start_date, training_end_date, testing_end_date = None, train_periods = None, time_only = False):

    from numpy import vstack
    df = df.set_index('DATETIME')
    index = df.index.unique()

    if isinstance(training_end_date,int):
        start_idx = np.argwhere((index == start_date))[0][0]
        training_end_date = str(index[start_idx + training_end_date])
        print('Training ends on: {}'.format(training_end_date))
    train = df.loc[start_date:training_end_date].copy()

    if isinstance(testing_end_date,int):
        train_idx = np.argwhere((index == training_end_date))[0][0]
        testing_end_date = str(index[train_idx + testing_end_date])
        test = df.loc[training_end_date:testing_end_date].copy()
        print('testing ends on: {}'.format(testing_end_date))

    elif testing_end_date:
        test = df.loc[training_end_date:testing_end_date].copy()
    else:
        test = df.loc[training_end_date:].copy()
    
    test.drop(training_end_date, inplace = True) # drop the first week still in training period
    
    reindex = train.DATE_IND.min()
    train.DATE_IND -= reindex
    test.DATE_IND  -=  reindex
    y_train = train.COUNT.values.reshape((len(train),1))

    y_test = test.COUNT.values.reshape((len(test),1))
    
    if time_only:
            X_train = np.vstack([train.DATE_IND.values.ravel()]).T
            X_test = np.vstack([test.DATE_IND.values.ravel()]).T
    else:
            X_train = np.vstack([train.DATE_IND.values.ravel(), train.x_point.ravel(), train.y_point.ravel()]).T
            X_test = np.vstack([test.DATE_IND.values.ravel(), test.x_point.ravel(), test.y_point.ravel()]).T


    out = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test': y_test, 'train':train, 'test': test}
    return out


def pred_GP(m, data_dict):
    pred = m.predict_y(data_dict['X_test'])
    data_dict['test']['gp_pred'] = np.round(pred[0],0)
    data_dict['test']['gp_error'] = np.round(pred[0],0) - data_dict['y_test']
    data_dict['test']['gp_sq_error'] = np.square(data_dict['test']['gp_error'])
    print('added gp pred and error to test')

def fit_AR(series, t_pred =12 ):
    ar = AR(endog= series)
    ar_fit = ar.fit(maxlag=1)
    pred = ar.predict(params= ar_fit.params, end=t_pred)
    return np.round(pred,0)

def get_MSE(pred,y_test):
    MSE = ((np.square(pred - y_test)).sum()) / len(pred)
    return MSE

def AR_pipe(series, y_test):
    t_pred = len(y_test)
    pred = np.round(fit_AR(series.values, t_pred=t_pred),0)
    return pred , pred - y_test.values

def run_AR(data_dict, group_by = ['DATETIME','GRID_SQUARE']):
    if group_by == None:
        ar_df = data_dict['train']['COUNT']
        ar_df_test = data_dict['test']['COUNT']
        return AR_pipe(ar_df, ar_df_test)
        
    ar_df = data_dict['train'].groupby(group_by)['COUNT'].sum().unstack()
    ar_df_test = data_dict['test'].groupby(group_by)['COUNT'].sum().unstack()

    grid_error = pd.DataFrame()
    for c in ar_df:
        c = int(c)
        grid_error[c] =  np.array(AR_pipe(ar_df[c],ar_df_test[c])[1])
    grid_error.index = data_dict['test'].index.unique()
    return grid_error.apply(np.square, axis = 1).sum(axis = 1) / len(grid_error.columns)
