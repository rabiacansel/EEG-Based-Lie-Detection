# Level-5 Wavelet Decomposition and Deep Learning Architectures for EEG-Based Lie Detection: A Comparative Study on the LieWaves Dataset

## Dataset and Preprocessing

In this study, the LieWaves dataset was used, and ATAR-processed signals were taken as input data. The preprocessing pipeline is described below.

### 1. Overlapping Sliding Window (OSW)
The time-series signals were segmented using an overlapping sliding window approach.

- Window size: **384**
- Stride (step size): **32**

This step was applied to increase the number of training samples and capture temporal patterns more effectively.

---

### 2. Discrete Wavelet Transform (DWT)
After segmentation, feature extraction was performed using Discrete Wavelet Transform.

- Wavelet type: **db4**
- Decomposition level: **5 (Lv5)**

This step was used to extract both time and frequency domain information from the signals.

---

### 3. Feature Extraction
From each segment, the following statistical features were extracted:

- Minimum Value  
- Maximum Value  
- Median  
- Mean  
- Energy  
- Standard Deviation  
- Variance  
- Skewness  
- Entropy  
- Relative Energy  

---

### 4. Normalization
A block-based min-max normalization was applied.

Each subject block was normalized separately using min-max scaling to the range **[-1, 1]**:

$$
x' = 2 \cdot \frac{x - x_{min}}{x_{max} - x_{min}} - 1
$$

If the maximum and minimum values were equal, the normalized value was set to 0.

---

### 5. Validation Strategy
Model evaluation was performed using **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** to ensure subject-independent testing.

---

### 6. Models Used
The extracted features were evaluated using both machine learning and deep learning models:

- KNN  
- SVM  
- Random Forest (RF)  
- LSTM  
- BiLSTM  
- GRU  
- CNN  
- EEGNet  
