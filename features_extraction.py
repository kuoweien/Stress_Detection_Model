"""
Created on Thu Jul 2 10:00:00 2024

@author: weien
"""

import pandas as pd
import Library.TimeDomainSelection as TimeDomainSelection
import Library.FrequencyDomainSelection as FrequencyDomainSelection
from datetime import date
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




if __name__ == '__main__':

    # --------------------

    output_url = 'Data/Features/'
    input_N_start = 1
    input_N_end = 1

    # --------------------

    situation_time = 300  # sec
    epoch_time = 30  # sec
    fs = 250

    df_timedomian_features = pd.DataFrame()
    df_frequencydomian_features = pd.DataFrame()

    # -------Extract features--------
    for n in range(input_N_start, input_N_end + 1):

        if n == 7:  # No data of N7
            continue

        print('!Start extracting time domain features!')
        df_oneN_timedomain = TimeDomainSelection.get_timedomian_features(n)
        df_timedomian_features = pd.concat([df_timedomian_features, df_oneN_timedomain], axis=0)


        print('!Start extracting frequency domain features!')
        df_oneN_fqdomian = FrequencyDomainSelection.get_frequencydomian_features(n)
        df_frequencydomian_features = pd.concat([df_frequencydomian_features, df_oneN_fqdomian], axis=0)

    print()
    print(df_timedomian_features.head())
    print(df_frequencydomian_features.head())

    today = date.today()
    today_date = today.strftime("%y%m%d")

    df_features = df_timedomian_features.merge(df_frequencydomian_features, how='inner', on=['N', 'Epoch', 'Situation'])
    outputfile_url = output_url+'{}_Features.csv'.format(today_date)
    if not os.path.exists(outputfile_url):
        df_features.to_csv('Data/Features/{}_Features.csv'.format(today_date))
    else:
        print(f"The file '{outputfile_url}' already exists.")
