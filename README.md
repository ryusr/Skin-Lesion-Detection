# Skin Lesion Detection

A deep learning-based application for detecting and classifying various types of skin lesions and diseases. This project uses convolutional neural networks to analyze skin images and provide predictions for different skin conditions.

## Features

- Interactive GUI interface for easy image upload and analysis
- Support for multiple image formats (JPG, JPEG, PNG)
- Real-time prediction with confidence scores
- Multiple model implementations for comparison and evaluation
- Classification of 7 different skin conditions:
  - Actinic keratoses and intraepithelial carcinoma
  - Basal cell carcinoma
  - Benign keratosis-like lesions
  - Dermatofibroma
  - Melanoma
  - Melanocytic nevi
  - Vascular lesions

## Requirements

- Python 3.10.0
- Required Python packages:
  ```
  tensorflow
  keras
  numpy
  Pillow
  tkinter
  ```

## Installation

 Install the required packages:
   ```bash
   pip install tensorflow keras numpy Pillow
   ```

## Project Structure

```
Skin-Lesion-Detection/
├── models/
│   ├── model_01.h5         # First model implementation weights
│   └── model_02.h5         # Enhanced model implementation weights
├── srcs/
│   ├── train_main_model_01.py    # Training script for first model
│   └── train_main_model_02.py    # Training script for enhanced model
├── tests/
│   ├── test_model_01.py    # Testing script for first model
│   └── test_model_02.py    # Testing script for enhanced model
├── data/                   # Training data directory
├── Test_Data1/            # First test dataset
├── Test_Data2/            # Second test dataset
├── Test_Data3/            # Third test dataset
├── images/                # Additional project images
├── class.png              # Class diagram of the project
└── README.md
```

## Model Training

The project includes two model implementations:
1. Base Model (model_01): Initial implementation with basic architecture
2. Enhanced Model (model_02): Improved architecture with better performance

To train the models:
```bash
# Train the base model
python srcs/train_main_model_01.py

# Train the enhanced model
python srcs/train_main_model_02.py
```

## Usage

1. Navigate to the project directory:
   ```bash
   cd Skin-Lesion-Detection
   ```

2. Run the application (you can choose either model):
   ```bash
   # For base model
   python tests/test_model_01.py
   
   # For enhanced model
   python tests/test_model_02.py
   ```

3. Using the application:
   - Click the "Select Image" button to choose an image file
   - The selected image will be displayed in the application window
   - The prediction result and confidence score will appear below the image

## Model Information

The application uses deep learning models trained on a dataset of skin lesion images. Both models process images of size 128x128 pixels and provide predictions for 7 different categories of skin conditions. The enhanced model (model_02) typically provides better accuracy and more reliable predictions.

## Testing

The project includes three separate test datasets (Test_Data1, Test_Data2, Test_Data3) for comprehensive model evaluation. These datasets contain different types of skin lesion images to ensure robust testing of the models.

## Project Architecture

The project's architecture is documented in the `class.png` file, which shows the relationships between different components of the system. This diagram helps in understanding the overall structure and flow of the application.

## Notes

- The application supports image files in JPG, JPEG, and PNG formats
- Images are automatically resized to fit the display while maintaining aspect ratio
- Confidence scores are provided as percentages
- The GUI window is fixed at 800x800 pixels for optimal display
- Two model implementations are available for comparison
- Comprehensive test datasets are included for evaluation

## Data Management

Due to the large size of the datasets and image files, they are not included directly in the GitHub repository. You can access the data in one of the following ways:

### Download from External Storage

The test datasets and images are available from the following locations:
- Test datasets (Test_Data1, Test_Data2, Test_Data3): [Google Drive Link]
- Sample images: [Google Drive Link]

To use these files:
1. Download the zip files from the provided links
2. Extract them to the project root directory
3. Ensure the folder names match exactly: `Test_Data1`, `Test_Data2`, `Test_Data3`, and `images`
