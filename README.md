# Stress_Detection_python
The proposed of this project was to reduce the number of sensors and had high accuracy for psychological stress detection.

We used a single wearable to record electrocardiography (ECG) from participants. The proposed algorithm based on signal quality indices (SQI) for reducing the noisy can divide EMG from a wearable ECG device. We want to reduce the usage of electrodes and had higher accuracy than multi-device.

There were three algorithm were develpoed:
* Signal quality indices calculation
* R peak detection
* Separation of EMG from ECG algorithm

# Languages
Python 3.8.8

# Installation
Use the package manager pip to install numpy.
```bash
pip install numpy
```
Use the package manager pip to install pandas.
```bash
pip install pandas
```
Use the package manager pip to install matplotlib.
```bash
pip install matplotlib
```

# Signal preprocessing
<img width="240" alt="image" src="https://user-images.githubusercontent.com/25921591/222178747-9e10052f-998a-466b-9828-b80f6e1e419a.png">


# Meaning of this project
## Main files
* **readandclip_Rawdata.py**

  Read and decode the raw file, and clip the signal according to the time of the experiment. Then restore the data as a CSV file.

* **getFeatures_TimeDomain.py**

  Extract the features from the time domain of ECG and EMG signals. The epoch of signals was 30 seconds. The description of the features is shown below (title: Feature extraction)

* **getFeatures_FrequencyDomain.py**

  Extract the features from the frequency domain of ECG and EMG signals. The epoch of signals was 150 seconds and overlapped 80%.

* **validation_RpeakAlgo_fromMIT-BIH.py**

  Using MIT-BIH Database to validation the R peak detection we developed.

* **machine_learning.py**

  There were six models: Support Vector Machine、Random forest, Regression, Decision tree, Random forest regression, and Xgboost. The method of validation has a confusion matrix, AUC, Accuracy, Kappa, Sensitivity, Precision, and F1-score.

* **signalNoise_SQIthreshold.py**

  Find the threshold of signal quality indices, and measure the indices of input signals.

* **statistic_analyze.py**

  Including checking whether the data is normal or unnormal, T-test, Mann Whitney U test, Paired T-Test, Wilcoxon Signed-Rank Test, Pearson correlation, and Spearman correlation.

## Functional files
* **def_dataDecode.py**

  The code for decoding raw file

* **def_getRpeak_main.py**

  Including the Pantompkin Algorithm (Pan, J. and W.J. Tompkins, 1985) and the Shannon envelope algorithm (Manikandan et al., 2012). 

* **def_measureSQI.py**

  Measure the SQI (epoch was 2 seconds).

* **def_readandget_Rawdata.py**
  
  Reading the raw file.

## Files for drawing images
* **forpicture_devideECGandEMG.py**

  The result of the algorithm: separation of EMG from ECG algorithm.
<img src="https://user-images.githubusercontent.com/25921591/222200852-1a1bce8b-3b1e-4447-abbe-4c87726efbdf.png" width="300"> 

* **forpicture_shannon.py**

  The process of the algorithm: R peak detection (Adjust from the present study. (Manikandan et al., 2012). )
<img src="https://user-images.githubusercontent.com/25921591/222200887-cd8df1d3-04b8-45b7-b2f8-1f71920c8fff.png" width="400"> 

# Feature extraction
## ECG (HRV)
14 numbers heart rate variability (HRV) features extracted from ECG.

7 from the time domain, and 7 from the frequency domain.

| N  | Symbol | Feature description |
|  ----  | ----  | ---- |
| 1 | $RR_{Mean}$ | Mean of R-R intervals|
| 2 | $SD$ | Standard deviation of R-R intervals |
| 3 | $RMSSD$ | Root mean square of successive differences of R-R intervals |
| 4 | $NN_{50}$ | Number of interval differences of successive R-R intervals greater than 50ms |
| 5 | $pNN_{50}$ | Corresponding percentage of NN50 |
| 6 | $RR_{Skew}$ | Skewness of R-R intervals |
| 7 | $RR_{Kurt}$ | Kurtosis of R-R intervals |
| 8 | $TP$ | Total power (0-0.4 Hz) |
| 9 | $HF$ | High frequency power (0.15-0.4 Hz) |
| 10 | $LF$ | Low frequency power (0.04–0.15 Hz) |
| 11 | $VLF$ | Very low frequency power (0.003–0.04 Hz) |
| 12 | $LF/HF$ | A ratio of low Frequency to high frequency |
| 13 | $HF%$ | Normalized HF [HF / (TP - VLF) *100] |
| 14 | $LF%$ | Normalized LF [LF / (TP - VLF) *100] |

## EMG
7 numbers features extracted from EMG.

5 from the time domain, and 2 from the frequency domain.

| N  | Symbol | Feature description |
|  ----  | ----  | ---- |
| 1 | $RMS_{EMG}$ | Root mean square of electromyogram|
| 2 | $MAV_{EMG}$ | Mean absolute value |
| 3 | $VAR_{EMG}$ | Variance |
| 4 | $Energy_{EMG}$ | Energy |
| 5 | $ZC_{EMG}$ | Zero crossing |
| 6 | $MNF$ | Mean Frequency |
| 7 | $MDF$ | Median Frequency |

# Conclusion

This project had more focus on signal preprocessing before training the machine learning model. We can reduce the number of sensors and have high accuracy for psychological stress detection.
