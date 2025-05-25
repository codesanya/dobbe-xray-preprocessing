# dobbe-xray-preprocessing
Adaptive image preprocessing for IOPA X-rays - Technical Assignment for Dobbe AI


# Adaptive Preprocessing Pipeline for DICOM IOPA X-Ray Images

This repository contains a preprocessing pipeline for dental X-ray images in DICOM format, designed to improve image quality for downstream machine learning and analysis tasks. The pipeline includes static and adaptive preprocessing methods based on key image quality metrics such as brightness, contrast, sharpness, and noise.


## Features
- Load and visualize DICOM images.
- Compute image quality metrics:
  - Brightness (mean intensity)
  - Contrast (standard deviation, Michelson contrast)
  - Sharpness (Laplacian variance, Tenengrad, gradient magnitude)
  - Noise estimation (flat region standard deviation, wavelet noise estimate, local variance)

- Static preprocessing pipeline:
  - Histogram equalization
  - Sharpening filter
  - Gaussian denoising

- Adaptive preprocessing pipeline that adjusts enhancement and denoising based on image metrics.
- Visual comparison of original and processed images.
- Quantitative comparison of image metrics before and after preprocessing.



## Directory Structure

```

dobbe-xray-preprocessing/  


├── notebooks/
│   └── main\_code.ipynb      
├── Reference\_Output\_Quality.jpg  
└── README.md

````

## Installation
1. Clone the repository:

```
git clone https://github.com/codesanya/dobbe-xray-preprocessing.git
cd dobbe-xray-preprocessing
```

2. Install required packages:

```
pip install pydicom opencv-python-headless scikit-image matplotlib seaborn pywt
```

## Usage
## Load and Visualize DICOM Images

This section explains how to read and visualize intraoral periapical (IOPA) X-ray images stored in DICOM format.

### Set the Folder Path

Specify the path where your DICOM files are stored:
```python
data_path = '/content/drive/MyDrive/Images_Data_science_intern'
```
---

### Load and Display DICOM Images

```python
import os
import pydicom
import matplotlib.pyplot as plt

def load_dicom(file_path):
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array
    return image

folder_path = data_path
dcm_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
print(f"Found {len(dcm_files)} DICOM files.")

plt.figure(figsize=(20, 5))
for i, filename in enumerate(dcm_files):
    filepath = os.path.join(folder_path, filename)
    image = load_dicom(filepath)
    plt.subplot(1, len(dcm_files), i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Image {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## Image Quality Metrics

To guide preprocessing decisions and enable image-specific enhancement, several **quantitative image quality metrics** are computed for each input IOPA DICOM image. These metrics form the basis of the adaptive preprocessing pipeline and are also useful for analysis and reporting.

### 1. Brightness

- **Function**: `compute_brightness(img)`
- **Description**: Computes the average intensity of all pixels in the image.
- **Interpretation**: Higher values indicate brighter images.
```python
def compute_brightness(img):
    return np.mean(img)
```
---

### 2. Contrast

#### a. Standard Deviation-Based Contrast

* **Function**: `compute_contrast_std(img)`
* **Description**: Uses standard deviation of pixel intensities to estimate contrast.
* **Interpretation**: Higher values indicate more contrast.

```python
def compute_contrast_std(img):
    return np.std(img)
```

#### b. Michelson Contrast

* **Function**: `compute_contrast_michelson(img)`
* **Description**: Based on the difference between the maximum and minimum intensity.
* **Formula**: $(I_{max} - I_{min}) / (I_{max} + I_{min})$

```python
def compute_contrast_michelson(img):
    img = img.astype(np.float32)
    I_max = np.max(img)
    I_min = np.min(img)
    if I_max + I_min == 0:
        return 0
    return (I_max - I_min) / (I_max + I_min)
```

---

### 3. Sharpness

#### a. Laplacian Variance

* **Function**: `compute_sharpness_laplacian(img)`
* **Description**: Variance of Laplacian filter output — measures edge strength.
* **Interpretation**: Higher values indicate sharper images.

```python
def compute_sharpness_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()
```

#### b. Tenengrad Method

* **Function**: `compute_sharpness_tenengrad(img)`
* **Description**: Uses gradients (Sobel filter) to calculate sharpness.
* **Interpretation**: Measures energy in edges; higher values mean higher detail.

```python
def compute_sharpness_tenengrad(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx ** 2 + gy ** 2)
```

---

### 4. Noise Level

* **Function**: `estimate_noise_std(img)`
* **Description**: Uses wavelet decomposition to estimate noise standard deviation.
* **Technique**: Based on median absolute deviation of high-frequency wavelet coefficients.

```python
import pywt

def estimate_noise_std(img):
    coeffs = pywt.wavedec2(img, 'db1', level=1)
    _, (cH, cV, cD) = coeffs
    return np.median(np.abs(cD)) / 0.6745
```

---

These metrics are used as **input heuristics** for the **adaptive preprocessing pipeline**, enabling dynamic adjustments to brightness, contrast, sharpness, and noise suppression based on the condition of each image.



## Static Preprocessing Baseline

A static preprocessing pipeline was implemented to apply a fixed set of enhancements to all input IOPA DICOM images. This pipeline improves overall image visibility and clarity without relying on image-specific adaptation.

### Steps in the Static Pipeline:

1. **Histogram Equalization**  
   Enhances contrast by spreading out pixel intensity values, making details in low-contrast areas more visible.
   ```python
   def histogram_equalization(img):
       img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
       return cv2.equalizeHist(img_uint8)```

2. **Sharpening**
   A sharpening kernel is applied to emphasize edges and enhance fine details in the image.

   ```python
   def sharpen_image(img):
       kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
       return cv2.filter2D(img, -1, kernel)
   ```

3. **Denoising (Gaussian Blur)**
   A light Gaussian blur is applied to suppress high-frequency noise without heavily degrading image structure.

   ```python
   def denoise_image(img):
       return cv2.GaussianBlur(img, (5, 5), 0)
   ```

4. **Combined Static Preprocessing Pipeline**
   The above steps are combined into a single function:

   ```python
   def static_preprocess(img):
       img = histogram_equalization(img)
       img = sharpen_image(img)
       img = denoise_image(img)
       return img
   ```

---

While effective in many cases, this approach does not adapt dynamically to the varying quality levels of different X-ray images, which motivates the need for an adaptive preprocessing strategy.


## Adaptive Preprocessing Pipeline

The adaptive preprocessing pipeline dynamically adjusts image enhancement techniques based on the quality metrics of each input DICOM X-ray image. This approach ensures image-specific correction for varying brightness, contrast, sharpness, and noise levels — overcoming the limitations of fixed/static preprocessing.

### Key Quality Metrics & Heuristics

For each image, the following metrics are computed:

- **Brightness** – Mean pixel intensity.
- **Contrast** – Standard deviation of pixel intensities.
- **Sharpness** – Laplacian variance for edge strength.
- **Noise** – Estimated using local standard deviation or similar statistical methods.

### Adaptive Enhancements Based on Metrics

1. **Brightness Adjustment**
   Adjusts exposure dynamically:
   - If brightness is **too low** (`< 100`), the image is brightened using `alpha=1.2` and `beta=20`.
   - If brightness is **too high** (`> 150`), the image is darkened with `alpha=0.8` and `beta=-20`.

   ```python
   img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)  # brighten
   img = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)  # darken


2. **Contrast Enhancement (CLAHE)**
   Adaptive histogram equalization (CLAHE) is applied based on contrast:

   * Contrast > 70 → `clipLimit = 1.5`
   * Contrast > 50 → `clipLimit = 2.5`
   * Contrast > 30 → `clipLimit = 4.5`
   * Else → `clipLimit = 4.0`

   ```python
   clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
   img = clahe.apply(img)
   ```

3. **Sharpness Enhancement (Unsharp Masking)**
   If image sharpness is below a threshold (`< 150`), unsharp masking is applied:

   * A Gaussian-blurred version of the image is subtracted from the original using a weighted blend (`1.5` and `-0.5`) to enhance edges.

   ```python
   blurred = cv2.GaussianBlur(img, (9, 9), 3.0)
   img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
   ```

4. **Noise Reduction**
   Applied based on estimated noise level:

   * If noise > 2 → Apply **bilateral filtering**
   * If noise > 1.2 → Apply **non-local means denoising**

   ```python
   img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
   img = cv2.fastNlMeansDenoising(img, h=12)
   ```

5. **Normalization**
   Final image is normalized to range `[0, 255]` to ensure consistent brightness and contrast output.

   ```python
   img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
   ```

### Visualization

Each DICOM image is visualized alongside its adaptively enhanced version using the `show_adaptive()` function. This helps in qualitative comparison and validation of the adaptive processing effectiveness.

```python
show_adaptive(dicom_data)
```

---

This adaptive approach tailors preprocessing to each image’s unique characteristics, leading to better visibility of diagnostic features in intraoral periapical (IOPA) radiographs. It serves as a strong foundation for downstream tasks such as anomaly detection or diagnostic assistance.


## Quantitative Evaluation Metrics

To objectively assess the quality of image enhancement techniques, two standard image similarity metrics are computed: **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index Measure)**. These metrics compare the original and preprocessed images to quantify enhancement performance.

**Function**: `compute_metrics(original, processed)`

This function calculates:

1. **PSNR (Peak Signal-to-Noise Ratio)**  
   - Measures the ratio between the maximum possible pixel value and the power of corrupting noise.
   - Higher PSNR generally indicates better image quality.
   - Expressed in decibels (dB).

2. **SSIM (Structural Similarity Index Measure)**  
   - Evaluates perceived quality based on structural information, luminance, and contrast.
   - Ranges from `-1` to `1`, where `1` means perfect similarity.

### Data Handling

- Both `original` and `processed` images are converted to `uint8` with intensity range [0, 255] before computation to ensure compatibility with PSNR and SSIM functions.

### Code

```python
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def compute_metrics(original, processed):
    if original.dtype != np.uint8:
        original_uint8 = (original * 255).astype(np.uint8)
    else:
        original_uint8 = original

    if processed.dtype != np.uint8:
        processed_uint8 = (processed * 255).astype(np.uint8)
    else:
        processed_uint8 = processed

    psnr = compare_psnr(original_uint8, processed_uint8, data_range=255)
    ssim = compare_ssim(original_uint8, processed_uint8, data_range=255)

    return psnr, ssim
```


## Metrics Computed

- **Brightness**: Average pixel intensity; higher values indicate brighter images.

- **Contrast**:
  - **Standard Deviation (Std)**: Measures the spread of pixel intensity values.
  - **Michelson Contrast**: Computed as (I_max - I_min) / (I_max + I_min); useful for normalized contrast evaluation.

- **Sharpness**:
  - **Laplacian Variance**: Measures the presence of edges using the Laplacian operator.
  - **Tenengrad**: Uses the gradient magnitude (via Sobel operator) to assess sharpness.
  - **Gradient Magnitude**: Evaluates how rapidly intensity changes, contributing to perceived detail.

- **Noise**:
  - **Flat Region STD**: Standard deviation in smooth regions, indicating background noise.
  - **Wavelet Estimate**: Uses wavelet decomposition to assess noise.
  - **Local Variance**: Measures noise based on small patches' variability.

- **Evaluation Metrics**:
  - **PSNR (Peak Signal-to-Noise Ratio)**: Quantifies reconstruction quality compared to the original image. Higher values indicate better fidelity.
  - **SSIM (Structural Similarity Index)**: Measures structural similarity between original and processed images. Closer to 1 indicates better quality.



## Installation

1. Clone the repository:

```bash
git clone https://github.com/codesanya/dobbe-xray-preprocessing.git
cd dobbe-xray-preprocessing
````


## Results

- Bar plots comparing brightness, contrast, sharpness, and noise metrics.
![image](https://github.com/user-attachments/assets/f2f5a635-7f83-4e1e-aa3c-71601680424a)

- Visual comparison plots showing original and processed images.
![image](https://github.com/user-attachments/assets/46022658-2806-4adf-a315-41813d63996e)

- Summary tables with metric values before and after preprocessing.
![image](https://github.com/user-attachments/assets/41ca8721-651f-4319-802e-b57501a08df7)
```
| Image   | Brightness Before | Contrast STD Before | Contrast Mich Before | Sharpness Lap Before | Sharpness Ten Before | Noise Est Before | Brightness After | Contrast STD After | Contrast Mich After | Sharpness Lap After | Sharpness Ten After | Noise Est After |
|---------|-------------------|---------------------|---------------------|----------------------|---------------------|------------------|------------------|--------------------|---------------------|---------------------|--------------------|-----------------|
| Image 1 | 116.35            | 83.57               | 1.0                 | 89.44                | 1171.73             | 1.48             | 117.96           | 72.50              | 1.0                 | 58.18               | 1975.92            | 0.74            |
| Image 2 | 130.84            | 76.73               | 1.0                 | 116.70               | 1383.18             | 1.48             | 128.15           | 67.88              | 1.0                 | 125.03              | 2673.12            | 0.74            |
| Image 3 | 174.68            | 49.16               | 1.0                 | 82.74                | 730.63              | 0.74             | 135.45           | 50.06              | 1.0                 | 737.25              | 6765.51            | 2.97            |
| Image 4 | 175.43            | 50.17               | 1.0                 | 175.70               | 1658.82             | 1.48             | 150.17           | 49.86              | 1.0                 | 56.90               | 1908.19            | 0.74            |
| Image 5 | 127.78            | 72.30               | 1.0                 | 101.51               | 1124.74             | 1.48             | 129.61           | 64.44              | 1.0                 | 87.44               | 1824.39            | 0.74            |
| Image 6 | 142.20            | 43.95               | 1.0                 | 373.52               | 4381.72             | 2.97             | 133.99           | 55.59              | 1.0                 | 367.05              | 8885.65            | 1.48            |
| Image 7 | 166.90            | 34.43               | 1.0                 | 236.92               | 3065.78             | 1.48             | 127.12           | 52.85              | 1.0                 | 736.01              | 15543.77           | 2.22            |

```

## Machine Learning / Deep Learning Approach

To scale and automate the preprocessing of IOPA X-ray images, a Machine Learning (ML) or Deep Learning (DL) based approach can be employed. Such a model can learn to adaptively enhance images based on their quality profile, removing the need for manually tuned pipelines.

### Proposed Directions

1. **Image Quality Classification-Based Pipeline**:
   - Train a lightweight classification model (e.g., Logistic Regression, Random Forest, or CNN) to categorize images based on quality metrics like brightness, contrast, and sharpness.
   - Based on the predicted class (e.g., "Low Brightness", "High Noise"), apply a corresponding static preprocessing method tailored to improve the image.

2. **Autoencoder-Based Denoising**:
   - Use an autoencoder or U-Net architecture to learn image-to-image mapping for enhancement or denoising.
   - The model would take in a noisy or low-quality image and output a cleaner version, learning from a set of synthetic degraded–clean image pairs.

### Proof-of-Concept Implementation (Planned / Basic Setup)

- A basic **Convolutional Autoencoder** was set up using synthetic noise injection (e.g., Gaussian noise) to create degraded–clean image pairs for supervised learning.
- The model structure includes:
  - Encoder: 2–3 Conv + ReLU + MaxPooling layers
  - Decoder: 2–3 ConvTranspose + ReLU layers
- Trained on a small subset to observe qualitative improvement.

> _Note: Due to time and data constraints, this is a limited proof-of-concept. Further training and testing on a larger dataset is required for robust results._

### Challenges and Data Requirements

- **Paired Dataset Need**: DL models for image enhancement typically require high-quality ground truth images paired with degraded versions, which are not easily available in real-world clinical scenarios.
- **Domain Specificity**: Medical images vary in acquisition conditions. A model trained on one type of noise/artifact might not generalize well.
- **Annotation for Classification Models**: Labeling images based on quality requires expert input or robust automated heuristics.
- **Computational Resources**: DL models benefit from GPU acceleration and large-scale training data.



## References
- PyDicom
- OpenCV documentation
