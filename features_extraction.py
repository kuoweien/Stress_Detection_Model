
import pandas as pd
import Library.TimeDomainSelection as TimeDomainSelection
import Library.FrequencyDomainSelection as FrequencyDomainSelection
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




if __name__ == '__main__':

    input_N_start = 1
    input_N_end = 3
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













