# SeaFront Dataset Preview Tool

A visualization tool for exploring the SeaFront synthetic dataset for visual container inspection.

## About the Dataset

SeaFront is a synthetic dataset for shipping container inspection that includes:
- Damage detection and segmentation
- IMDG (International Maritime Dangerous Goods) label detection
- Door/no door classification
- OCR (Optical Character Recognition) tasks

The dataset contains nearly 10,000 images for training and validation, and 2,480 for testing.

## Requirements
(see requirements.txt)

- Python 3.6+
- Required packages:
  - numpy
  - matplotlib
  - Pillow (PIL)
  - requests (for downloading)
  - tqdm (for download progress)

## Installation

### Install requirements

```
pip install requirements.txt
```

## Usage

Run the preview tool by providing the path to the extracted dataset:

```
python seafront_preview.py /Users/xxx/Documents/dataset
```

### Interactive Controls

- **Task Selector**: Choose between different tasks (damage, IMDG, door classification, OCR)
- **Navigation**:
  - Use the "Previous" and "Next" buttons to navigate through images
  - Use the slider to quickly jump to a specific image
- **Visualizations**:
  - **Damage task**: Shows the original image with segmentation overlay and bounding boxes
  - **IMDG task**: Shows bounding boxes for dangerous goods labels
  - **Door Classification**: Shows image with door/no door classification
  - **OCR task**: Shows text recognition results

## Dataset Structure

The SeaFront dataset is organized as follows:

- **ann_dir/**: Annotations of damages in image format (PNG)
- **bbannotation/**: Annotations of damages in plain text format
- **ds/classification**: Images for door/no door classification
- **ds/detection**: Images and YOLO format annotations for IMDG task
- **images/**: Original rendered images
- **labels/**: Annotations for IMDG task in YOLOv4 format
- **ocr/**: (Test dataset only) Images and annotations for OCR task
