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
1. Set your DICOM images folder path:

```
data_path = '/content/drive/MyDrive/Images_Data_science_intern'
```
2. Load and visualize images:

```
# Load and display DICOM images
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
![image](https://github.com/user-attachments/assets/08bbd8a8-995c-4e57-9a5c-c110dfb87dd2)

Explanation:
-  Uses pydicom to read DICOM file.
- Plots the pixel arrays with grayscale colormap.


3. Compute image quality metrics:

```
# Use functions like compute_brightness, compute_contrast_std, etc.
def compute_brightness(img):
    return np.mean(img)

def compute_contrast_std(img):
    return np.std(img)

def compute_contrast_michelson(img):
    img = img.astype(np.float32)
    I_max = np.max(img)
    I_min = np.min(img)
    if I_max + I_min == 0:
        return 0
    return (I_max - I_min) / (I_max + I_min)

def compute_sharpness_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def compute_sharpness_tenengrad(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx ** 2 + gy ** 2)

def estimate_noise_std(img):
    coeffs = pywt.wavedec2(img, 'db1', level=1)
    _, (cH, cV, cD) = coeffs
    return np.median(np.abs(cD)) / 0.6745

```

4. Apply static preprocessing pipeline:

```
def histogram_equalization(img):
    img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.equalizeHist(img_uint8)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def denoise_image(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def static_preprocess(img):
    img = histogram_equalization(img)
    img = sharpen_image(img)
    img = denoise_image(img)
    return img

```
5. Apply adaptive preprocessing pipeline:

```
# Adaptive preprocessing function
def adaptive_preprocess(img):
    img = img.astype(np.uint8)

    brightness = compute_brightness(img)
    contrast = compute_contrast_std(img)
    sharpness = compute_sharpness_laplacian(img)
    noise = estimate_noise_std(img)

    # Adjust brightness
    if brightness < 100:
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)  # brighten
    elif brightness > 150:
        img = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)  # darken

    # Contrast enhancement using CLAHE
    if contrast > 70:
        clahe_clip = 1.5
    elif contrast > 50:
        clahe_clip = 2.5
    elif contrast > 30:
        clahe_clip = 4.5
    else:
        clahe_clip = 4.0
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Sharpen if sharpness low
    if sharpness < 100:
        img = sharpen_image(img)

    # Denoise if noise high
    if noise > 20:
        img = denoise_image(img)

    return img
```
6. Compare image quality metrics before and after processing:

```
# Use compute_all_metrics and display results
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
