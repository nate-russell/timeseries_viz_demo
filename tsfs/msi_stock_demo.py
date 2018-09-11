import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
import json
import datetime
import os

if __name__ == '__main__':
    # Get Stock data into format desired by Prophet
    raw_df = pd.read_csv('MSI.csv', nrows=1000)  # restrict data size for faster runs
    ds = raw_df['Date'].values
    y = raw_df['Close'].values
    df_dict = {
        'ds': ds,
        'y': y
    }
    df = pd.DataFrame(data=df_dict)
    print(df.head())

    # Fit and Predict
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=(365))
    future.tail()
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Make Plots
    m.plot(forecast)
    plt.savefig('../static_site/images/forecast.png')
    plt.close()
    m.plot_components(forecast)
    plt.savefig('../static_site/images/trends.png')
    plt.close()

    # Manipulate Data for getting data
    df['ds'] = pd.to_datetime(df['ds'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    df['epoch'] = df['ds'].astype('int64')
    epoch = datetime.datetime.utcfromtimestamp(0)
    forecast['epoch'] = (forecast['ds'] - epoch)
    forecast['epoch'] = [d.total_seconds() for d in forecast['epoch']]

    # rickshaw needs series data to be same length
    forecast = pd.concat([forecast, df['y']], axis=1, join_axes=[forecast.index])
    # make some extra series to include in interactive graph
    forecast['gap'] = forecast['yhat_upper'] - forecast['yhat_lower']
    forecast.y.fillna(forecast.yhat, inplace=True)
    forecast['error'] = np.abs(forecast['yhat'] - forecast['y'])

    # using index as x is not correct but for some reason mutch faster
    f = [{"x": index, "y": row['yhat']} for index, row in forecast.iterrows()]
    fub = [{"x": index, "y": row['yhat_upper']} for index, row in forecast.iterrows()]
    flb = [{"x": index, "y": row['yhat_lower']} for index, row in forecast.iterrows()]
    gap = [{"x": index, "y": row['gap']} for index, row in forecast.iterrows()]
    err = [{"x": index, "y": row['error']} for index, row in forecast.iterrows()]
    ts = [{"x": index, "y": row['y']} for index, row in forecast.iterrows()]

    # using row['epoch'] gives the correct dates for the data but is much slower for some reason i don't yet know
    '''
    f = [{"x": row['epoch'], "y": row['yhat']} for index, row in forecast.iterrows()]
    fub = [{"x": row['epoch'], "y": row['yhat_upper']} for index, row in forecast.iterrows()]
    flb = [{"x": row['epoch'], "y": row['yhat_lower']} for index, row in forecast.iterrows()]
    gap = [{"x": row['epoch'], "y": row['gap']} for index, row in forecast.iterrows()]
    err = [{"x": row['epoch'], "y": row['error']} for index, row in forecast.iterrows()]
    ts = [{"x": row['epoch'], "y": row['y']} for index, row in forecast.iterrows()]
    '''

    # Compose the dict and dump it as json so jquery can pass it to rickshaw
    data_dict = {
        "forecast": f,
        "flb": flb,
        "fub": fub,
        "truth": ts,
        "gap": gap,
        "error": err
    }
    with open('../static_site/data/data.json', 'w', ) as outfile:
        json.dump(data_dict, outfile, indent=3)

    # Now and Alternative to doing all of that ! I could just have used the forecating model implemendted in TSF_Model_prototype.py
    print('\n'*5)   # import my model
    from tsfs.TSF_Model_Prototype import MyProphet_Forecaster

    # Initialize an instance of my model
    instance_of_my_model = MyProphet_Forecaster()

    # Train my model
    instance_of_my_model.fit(df=df,  # the same df I make on line 17
                             t='ds',  # the time columns of that df
                             y='y',  # the y column of the df (what we are predicting)
                             x='y'   # doesn't matter for prophet models since its always the same as x. but we include it for consistency ()
                             )

    # Now Predict future forecasts
    prediction_df = instance_of_my_model.forecast(
        df=df,  # the same df i use earlier on line 18
        t='ds',  # the time columns of that df
        y='y',  # the y column of the df,
        x='y',   # the thing we are using to predict y
        n = 365, # same as periods on line 23
    )

    # see what that gets us
    print(prediction_df)
    instance_of_my_model.diagnostics(os.getcwd()) # this model will dump results right into its directory wheras the other on lines 29 - 34 goes into static site


    # Now compare the two and see they are the same

