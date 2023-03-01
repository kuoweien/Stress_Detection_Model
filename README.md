# Stress_Detection_python
The proposed of this project was to reduce the number of sensors and had high accuracy for psychological stress detection.

We used a single wearable to record electrocardiography (ECG) from participants. The proposed algorithm based on signal quality indices (SQI) for reducing the noisy can divide EMG from a wearable ECG device. We want to reduce the usage of electrodes and had higher accuracy than multi-device.

There were three algorithm were develpoed:
* Signal quality indices calculation
* R peak detection
* Separation of EMG from ECG algorithm


# Signal prepocessing
<img width="240" alt="image" src="https://user-images.githubusercontent.com/25921591/222178747-9e10052f-998a-466b-9828-b80f6e1e419a.png">

# Feature extraction
## ECG (HRV)
14 numbers of heart rate variability (HRV) features extracted from ECG
7 from time domain, and 7 from frequency domain

| N  | Symbol | Feature description |
|  ----  | ----  | ---- |
| 1 | $RR_Mean$ | Mean of R-R intervals|
| 2 | SD | Standard deviation of R-R intervals |
| 3 | RMSSD | Root mean square of successive differences of R-R intervals |
| 4 | N$N_50$ | Number of interval differences of successive R-R intervals greater than 50ms |
| 5 | pNN50 | Corresponding percentage of NN50 |
| 6 | RRSkew | Skewness of R-R intervals |
| 7 | RRKurt | Kurtosis of R-R intervals |
| 8 | TP | Total power (0-0.4 Hz) |
| 9 | HF | High frequency power (0.15-0.4 Hz) |
| 10 | LF | Low frequency power (0.04–0.15 Hz) |
| 11 | VLF | Very low frequency power (0.003–0.04 Hz) |
| 12 | LF/HF | A ratio of low Frequency to high frequency |
| 13 | HF% | Normalized HF [HF / (TP - VLF) *100] |
| 14 | LF% | Normalized LF [LF / (TP - VLF) *100] |

## EMG
7 numbers of features extracted from EMG
5 from time domain, and 2 from frequency domain

| N  | Symbol | Feature description |
|  ----  | ----  | ---- |
| 1 | $RMS_EMG$ | Root mean square of electromyogram|
| 2 | MAVEMG | Mean absolute value |
| 3 | VAREMG | Variance |
| 4 | EnergyEMG | Energy |
| 5 | ZCEMG | Zero crossing |
| 6 | MNF | Mean Frequency |
| 7 | MDF | Median Frequency |
