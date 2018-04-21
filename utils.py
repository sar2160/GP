import numpy as np

def preprocess(df, start_date, training_end_date, testing_end_date = None, train_periods = None, ):

    from numpy import vstack
    df = df.set_index('DATETIME')
    index = df.index.unique()

    if isinstance(training_end_date,int):
        start_idx = np.argwhere((index == start_date))[0][0]
        training_end_date = str(index[start_idx + training_end_date])
        print(f'Training ends on: {training_end_date}')
    train = df.loc[start_date:training_end_date].copy()

    if isinstance(testing_end_date,int):
        train_idx = np.argwhere((index == training_end_date))[0][0]
        testing_end_date = str(index[train_idx + testing_end_date])
        test = df.loc[training_end_date:testing_end_date].copy()
        print(f'testing ends on: {testing_end_date}')

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
