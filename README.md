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
14 numbers of heart rate variability (HRV) parameters calculated from ECG
7 from time domain, and 7 from frequency domain

| N  | Symbol | Feature description |
|  ----  | ----  | ---- |
| 1 | $RR_Mean$ | Mean of R-R intervals|
| 2 | SD | Standard deviation of R-R intervals |
| 3 | RMSSD | Root mean square of successive differences of R-R intervals |
| 4 | NN50 | Number of interval differences of successive R-R intervals greater than 50ms |
| 5 | SD | Standard deviation of R-R intervals |
| 6 | SD | Standard deviation of R-R intervals |
| 7 | SD | Standard deviation of R-R intervals |
| 8 | SD | Standard deviation of R-R intervals |
| 9 | SD | Standard deviation of R-R intervals |
| 10 | SD | Standard deviation of R-R intervals |
| 11 | SD | Standard deviation of R-R intervals |
| 12 | SD | Standard deviation of R-R intervals |
| 13 | SD | Standard deviation of R-R intervals |
| 14 | SD | Standard deviation of R-R intervals |

