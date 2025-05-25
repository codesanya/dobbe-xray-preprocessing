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

fig, axes = plt.subplots(1, len(dcm_files), figsize=(5 * len(dcm_files), 5))
if len(dcm_files) == 1:
    axes = [axes]

for ax, file in zip(axes, dcm_files):
    file_path = os.path.join(data_path, file)
    dicom_data = pydicom.dcmread(file_path)
    ax.imshow(dicom_data.pixel_array, cmap="gray")
    ax.set_title(file, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()
![image](https://github.com/user-attachments/assets/08bbd8a8-995c-4e57-9a5c-c110dfb87dd2)

```
3. Compute image quality metrics:

```
# Use functions like compute_brightness, compute_contrast_std, etc.
```

4. Apply static preprocessing pipeline:

```
processed_img = static_preprocess(image)
```
5. Apply adaptive preprocessing pipeline:

```
processed_img = adaptive_preprocess(image)
```
6. Compare image quality metrics before and after processing:

```
# Use compute_all_metrics and display results
```

## Metrics Computed
- Brightness: Average pixel intensity

- Contrast (Std, Michelson): Measures variation in pixel intensities

- Sharpness (Laplacian variance, Tenengrad, Gradient Magnitude): Measures edges and detail clarity

- Noise (Flat region std, Wavelet estimate, Local variance): Measures noise level in images



## Installation

1. Clone the repository:

```bash
git clone https://github.com/codesanya/dobbe-xray-preprocessing.git
cd dobbe-xray-preprocessing
````


## Results
- Visual comparison plots showing original and processed images.

- Bar plots comparing brightness, contrast, sharpness, and noise metrics.

- Summary tables with metric values before and after preprocessing.



## Future Work

* Incorporate ML/DL models for adaptive enhancement
* Add support for RVG format parsing
* Deploy as a web app with streamlit or Gradio

## References
- PyDicom
- OpenCV documentation
