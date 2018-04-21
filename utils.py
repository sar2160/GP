import numpy as np

def preprocess(df, start_date , training_end_date, testing_end_date = None):
    from numpy import vstack
    df = df.set_index('DATETIME')
    train = df.loc[start_date:training_end_date].copy()
    if testing_end_date:
        test = df.loc[training_end_date:testing_end_date].copy()
    else:
        test = df.loc[training_end_date:].copy()

    X_train = np.vstack([train.DATE_IND.values.ravel(), train.x_point.ravel(), train.y_point.ravel()]).T
    y_train = train.COUNT.values.reshape((len(train),1))

    X_test = np.vstack([test.DATE_IND.values.ravel(), test.x_point.ravel(), test.y_point.ravel()]).T
    y_test = test.COUNT.values.reshape((len(test),1))

    return X_train, y_train, X_test, y_test, train, test
