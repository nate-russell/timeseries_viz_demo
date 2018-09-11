'''
This page has everything you need to make your own Time Series Forecasting Model that conforms to the API!

TSF_MODEL -> The inherited class and description

test_TSF_Subclass -> Handy function to test if you TSF_Model sublcass is working. Check out __main__ to see it in action.

fit_method and forecast_method -> meant to be used as a decorator. They make data validation automatic so you know you are recieving and returning the right stuff !

MyProphet_Forecaster -> A usable example of a TSF_Model subclass implemnted correctly
WrongProphet_Forecaster -> Example of a bad class. See if you can fix it!

'''
from functools import wraps
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
from datetime import date, timedelta



class TSF_Model(object):
    '''
    Time Series Model prototype. All models should inherit from this model and follow it's api
    '''

    def fit(self, df, t='ds', y='y', x=['y']):
        '''
        Fit a model to the given data
        :param df: pandas.DataFrame containing all of your data
        :param y: str - column name that contains the y value you wish to predict. default name = 'y'
        :param x: list of strs- the list of strs that you wish to use as features in your predictio.
        default list contains only 'y' since most of the time, we only will use y itself to predict future y
        :return:
        '''
        # fit the model and persist inside class as self.m or something equivalent
        pass

    def forecast(self, df, t='ds', y='y', x=['y'], n=1):
        '''
        Predict 'y' from df, using column(s) in x up to n timesteps in the future
        Passed values should match those used in fit call
        :param df: pandas.DataFrame containing all of your data
        :param y: str - column name that contains the y value you wish to predict. default name = 'y'
        :param x: list of strs- the list of strs that you wish to use as features in your predictio.
        default list contains only 'y' since most of the time, we only will use y itself to predict future y
        :param n: int - how many time steps into the future will the algorithm predict. default is 1 time step
        :return: panda.DataFrame with @ least the following columns ['ds','yhat','yhat_upper','yhat_lower']. It should be the length of the dataframe put into the model + n
        '''
        # Check that

        # ----- Do your prediction here -----
        yhat = []
        yhat_upper = []
        yhat_lower = []
        # ----- End of modification region

        df = pd.DataFrame({'yhat': yhat,
                           'yhat_upper': yhat_upper,
                           'yhat_lower': yhat_lower})
        return df

    def diagnostics(self, out_dir):
        ''' Save diagnostic plots and log files to the outdir inside of a time stamped dir'''
        pass


def test_TSF_Subclass(c, n=50):
    '''
     Conducts a series of test to help you identify potential flaws in your TSF Subclass
     assumes methods are wrapped with decorator that tests inputs
    '''
    # Has the needed methods
    ci = c()

    if not hasattr(ci, 'fit'):
        raise AttributeError('fit method not found')
    if not hasattr(ci, 'forecast'):
        raise AttributeError('forecast method not found')
    if not hasattr(ci, 'diagnostics'):
        raise AttributeError('diagnostics method not found')
    # Make test data
    startdate = date(2000, 1, 1)
    ds = []
    for i in range(n):
        startdate += timedelta(days=1)
        ds.append(startdate)
    y = np.linspace(0, 12, num=n)
    y = np.power(y,1.2)
    y = y + np.cos(y)
    y = y + np.random.rand(n)
    df = pd.DataFrame({'y': y,
                      'ds': ds})

    # Test that fit and forecast methods are decorated
    # todo

    # The algorithm runs to completion on test data
    ci.fit(df=df,t='ds',y='y',x='y')
    prediction = ci.forecast(df=df,t='ds',y='y',x='y',n=5)
    # The methods give what is expected
    if not isinstance(prediction,pd.DataFrame):
        raise TypeError('%s returned from %s.forecast(...) instead of Pandas DataFrame'%(type(prediction),ci.__name__))
    not_found = []
    for col in ['yhat','yhat_upper','yhat_lower','ds']:
        if col not in prediction.columns:
            not_found.append(col)
    if not_found:
        raise ValueError('%s not found in %s.forecast(...) df columns'%(not_found,ci.__name__))

    # Give warnings on cl ass performance
    # todo
    =

    return True


class fit_method(object):
    ''' fit_method is a decorator that automates checking inputs to a fit method '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        pass

    def __call__(self, f):
        @wraps(f)
        def wrapped_f(*args,**kwargs):
            not_found = []
            for arg in ['df','x','y','t']:
                if arg not in kwargs:
                    not_found.append(arg)
                if not_found:
                    raise ValueError('%s not provided as arguments'%not_found)
            df = kwargs['df']
            if not isinstance(df,pd.DataFrame):
                raise TypeError('argument df is type: %s and not pandas.DataFrame as it should be'%type(df))
            x = kwargs['x']
            y = kwargs['y']
            t = kwargs['t']

            not_found = []
            for col in [x,y,t]:
                if col not in df.columns:
                    not_found.append(col)
                if not_found:
                    raise ValueError('%s not columns in df' % not_found)

            freturn = f(*args,**kwargs)
            return freturn

        return wrapped_f

    def __str__(self):
        return self.__doc__


class forecast_method(object):
    ''' fit_method is a decorator that automates checking inputs to a fit method '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        pass

    def __call__(self, f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            not_found = []
            for arg in ['df', 'x', 'y', 't','n']:
                if arg not in kwargs:
                    not_found.append(arg)
                if not_found:
                    raise ValueError('%s not provided as arguments' % not_found)
            df = kwargs['df']
            if not isinstance(df, pd.DataFrame):
                raise TypeError('argument df is type: %s and not pandas.DataFrame as it should be' % type(df))
            x = kwargs['x']
            y = kwargs['y']
            t = kwargs['t']

            not_found = []
            for col in [x, y, t]:
                if col not in df.columns:
                    not_found.append(col)
                if not_found:
                    raise ValueError('%s not columns in df' % not_found)

            freturn = f(*args, **kwargs)

            if not isinstance(freturn, pd.DataFrame):
                raise TypeError('return of .forecast(...) is type: %s and not pandas.DataFrame as it should be' % type(df))

            not_found = []
            for col in ['ds', 'yhat', 'yhat_upper','yhat_lower']:
                if col not in freturn.columns:
                    not_found.append(col)
                if not_found:
                    raise ValueError('%s are expected column(s) in .forecast(...)\'s returned DataFrame' % not_found)

            return freturn

        return wrapped_f

    def __str__(self):
        return self.__doc__


class MyProphet_Forecaster:
    ''' A well defined class that wraps the Facebook Prophet Module'''
    def __init__(self):
        self.m = Prophet()
        pass

    @fit_method() # This is a decorator that automates the checking of input arguments and returned objects.
    def fit(self, df, t='ds', y='y', x='y'):
        '''   '''
        prophet_df = pd.DataFrame({'ds': df[t].values,  # prophet model doesn't use x
                                   'y': df[y].values})
        self.m.fit(prophet_df)
        pass

    @forecast_method() # This is a decorator that automates the checking of input arguments and returned objects.
    def forecast(self, df, t='ds', y='y', x='y', n=1):
        ''' '''
        future = self.m.make_future_dataframe(periods=(n))
        self.prediction = self.m.predict(future)
        return self.prediction

    def diagnostics(self, out_dir):
        ''' '''
        self.m.plot(self.prediction)
        plt.savefig('%s/forecast.png' % (out_dir))
        plt.close()
        self.m.plot_components(self.prediction)
        plt.savefig('%s/trends.png' % (out_dir))
        plt.close()
        pass

class WrongProphet_Forecaster:
    ''' Purposely wrong class  missing the x argument in fit and only gives back yhat in forecast return dataframe'''
    def __init__(self):
        self.m = Prophet()
        pass

    @fit_method()
    def fit(self, df,y='y', t='ds'):
        '''   '''
        prophet_df = pd.DataFrame({'ds': df[t].values,  # prophet model doesn't use x
                                   'y': df[y].values})
        self.m.fit(prophet_df)
        pass

    @forecast_method()
    def forecast(self, df, t='ds', y='y', x='y', n=1):
        ''' '''
        future = self.m.make_future_dataframe(periods=(n))
        self.prediction = self.m.predict(future)['yhat']
        return self.prediction

    def diagnostics(self, out_dir):
        ''' '''
        self.m.plot(self.prediction)
        plt.savefig('%s/forecast.png' % (out_dir))
        plt.close()
        self.m.plot_components(self.prediction)
        plt.savefig('%s/trends.png' % (out_dir))
        plt.close()
        pass



if __name__ == '__main__':
    print('----------------- Testing %s -----------------\n\n'%MyProphet_Forecaster.__name__)
    if test_TSF_Subclass(MyProphet_Forecaster):
        print('\n\n---------------- The Class Looks Good! -----------------')

    print('----------------- Testing %s -----------------\n\n' % WrongProphet_Forecaster.__name__)
    if test_TSF_Subclass(WrongProphet_Forecaster):
        print('\n\n---------------- The Class Looks Good! -----------------')

