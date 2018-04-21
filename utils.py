import numpy as np
from statsmodels.tsa.ar_model import AR

def preprocess(df, start_date, training_end_date, testing_end_date = None, train_periods = None, ):

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

    X_train = np.vstack([train.DATE_IND.values.ravel(), train.x_point.ravel(), train.y_point.ravel()]).T
    y_train = train.COUNT.values.reshape((len(train),1))

    X_test = np.vstack([test.DATE_IND.values.ravel(), test.x_point.ravel(), test.y_point.ravel()]).T
    y_test = test.COUNT.values.reshape((len(test),1))

    out = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test': y_test, 'train':train, 'test': test}
    return out


def get_GPstats(m, data_dict):
    pred = m.predict_y(data_dict['X_test'])
    data_dict['test']['error'] = pred[0] -data_dict['y_test']
    MSE = data_dict['test'].groupby("DATETIME")['error'].mean()
    return MSE

def fit_AR(series, t_pred =12 ):
    ar = AR(endog= series)
    ar_fit = ar.fit(maxlag=1)
    pred = ar.predict(params= ar_fit.params, end=t_pred)
    return pred

def get_MSE(pred,y_test):
    MSE = ((np.square(pred - y_test)).sum()) / len(pred)
    return MSE

def AR_pipe(series, y_test):
    t_pred = len(y_test)
    pred = fit_AR(series.values, t_pred=t_pred)
    return pred - y_test.values

def run_AR(data_dict):
    ar_df = data_dict['train'].groupby(['DATETIME','GRID_SQUARE'])['COUNT'].sum().unstack()
    ar_df_test = data_dict['test'].groupby(['DATETIME','GRID_SQUARE'])['COUNT'].sum().unstack()

    grid_error = pd.DataFrame()
    for c in ar_df:
        c = int(c)
        grid_error[c] =  np.array(AR_pipe(ar_df[c],ar_df_test[c]))

    return grid_error.apply(np.square, axis = 1).sum(axis = 1) / len(grid_error.columns)
