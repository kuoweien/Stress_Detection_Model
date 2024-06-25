
import pandas as pd
import getFeatures_TimeDomain
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':

    input_N_start = 1
    input_N_end = 5
    situation_time = 300  # sec
    epoch_time = 30  # sec
    fs = 250

    situations = ['Baseline', 'Stroop', 'Arithmetic', 'Speech']
    df_timedomian_features = pd.DataFrame()

    for n in range(input_N_start, input_N_end + 1):

        if n == 7:  # No data of N7
            continue

        for situation in situations:
            ecg_url = 'Data/ClipSituation_CSVfile/N{}/{}.csv'.format(n, situation) # 讀取之ECG csv檔
            df = pd.read_csv(ecg_url)
            ecg_situation = df['ECG']

            for i in range(0, int(situation_time / epoch_time)):
                print('Epoch {}'.format(i))
                ecg_epoch = ecg_situation[i*fs*epoch_time:(i+1)*epoch_time*fs]
                if len(ecg_epoch) < (epoch_time*fs):
                    break
                df_features_epoch = getFeatures_TimeDomain.get_timedomian_features(ecg_epoch)
                df_features_epoch['N'] = n
                df_features_epoch['Epoch'] = i+1
                df_features_epoch['Situation'] = situation

                df_timedomian_features = pd.concat([df_timedomian_features, df_features_epoch], axis=0)

    print()
