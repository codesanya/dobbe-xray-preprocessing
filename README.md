# dobbe-xray-preprocessing
Adaptive image preprocessing for IOPA X-rays - Technical Assignment for Dobbe AI


# Adaptive Image Preprocessing Pipeline for IOPA X-rays

This repository contains an image preprocessing pipeline designed to enhance IOPA (Intraoral Periapical) X-ray images. The system supports both static and adaptive preprocessing approaches for improving image quality based on brightness, contrast, sharpness, and noise metrics. It also provides support for DICOM and RVG image formats.

## Directory Structure

```

dobbe-xray-preprocessing/  


├── notebooks/
│   └── main\_code.ipynb      
├── Reference\_Output\_Quality.jpg  
└── README.md

````

##  Features

- DICOM file support using `pydicom` and OpenCV
- Quality analysis:
  - Brightness (mean pixel intensity)
  - Contrast (standard deviation and Michelson contrast)
  - Sharpness (Laplacian, Tenengrad, Gradient Magnitude)
  - Noise Estimation (Wavelet-based, Local Variance, Flat Region STD)
- Static Preprocessing:
  - Histogram Equalization
  - CLAHE
  - Gaussian/Median Denoising
- Adaptive Preprocessing:
  - Automatically selects preprocessing methods based on image quality metrics
- Visualization:
  - Histograms and comparison plots

## Installation

1. Clone the repository:

```bash
git clone https://github.com/codesanya/dobbe-xray-preprocessing.git
cd dobbe-xray-preprocessing
````


## Sample Results

| Metric           | Description                          |
| ---------------- | ------------------------------------ |
| Brightness\_Mean | Overall intensity level              |
| Contrast\_Std    | Pixel intensity variability          |
| Sharp\_Laplacian | Edge strength via Laplacian operator |
| Noise\_Wavelet   | High-frequency noise estimate        |

## Future Work

* Incorporate ML/DL models for adaptive enhancement
* Add support for RVG format parsing
* Deploy as a web app with streamlit or Gradio

