# Prescription Handwriting Recognition Using Residual U-Net, BiLSTM, and Lexicon-Based Post-Processing

A deep learning framework for recognizing doctor handwritten prescriptions using a Residual U-Net architecture, Bidirectional LSTM sequence modeling, CTC loss optimization, and a medical lexicon-based error correction layer. This project demonstrates an end-to-end pipeline for medical handwriting recognition while addressing one of the most critical challenges in healthcare AI: reducing transcription errors through domain-specific post-processing.

Overview

Medical prescriptions are notoriously difficult to read due to varying handwriting styles, abbreviations, and domain-specific terminology. This project proposes a hybrid approach that combines:

Residual U-Net for visual feature extraction
Bidirectional LSTM (BiLSTM) for sequence learning
Connectionist Temporal Classification (CTC) for transcription without character-level alignment
Lexicon-Based Post-Processing using Levenshtein distance to correct low-confidence predictions

The implementation is fully self-contained and requires no external edit-distance libraries.

# Key Features

✅ End-to-End Handwritten Text Recognition (HTR)

✅ Residual U-Net CNN Encoder

✅ Bidirectional LSTM Sequence Modeling

✅ CTC Loss Training Strategy

✅ Self-Contained Levenshtein Distance Implementation

✅ Medical Lexicon-Based Error Correction

✅ Simulated Prescription Dataset Pipeline

✅ CPU and GPU Support

✅ Fully Runnable Without External Data Dependencies

# System Architecture
Input Prescription Image
           │
           ▼
   Image Preprocessing
   (Resize + Normalize)
           │
           ▼
    Residual U-Net Encoder
           │
           ▼
      Feature Extraction
           │
           ▼
       Sequence Layer
           │
           ▼
     Bidirectional LSTM
           │
           ▼
       CTC Decoding
           │
           ▼
 Raw Prescription Text
           │
           ▼
 Lexicon-Based Correction
           │
           ▼
 Final Medical Transcript

# Model Components
1. Image Preprocessing

The preprocessing pipeline performs:

Grayscale conversion
Aspect-ratio-preserving resizing
Image padding/cropping
Pixel normalization
Color inversion
Tensor conversion for PyTorch

Target dimensions:

Height = 64
Width = 512

2. Residual U-Net Encoder

The encoder extracts hierarchical visual features from handwritten prescriptions using Residual Blocks.

Encoder Structure
Input
 ↓
Conv + ReLU
 ↓
Residual Block + MaxPool
 ↓
Residual Block + MaxPool
 ↓
Residual Block + MaxPool
 ↓
Bridge Layer


Benefits:

Better gradient flow
Reduced vanishing gradients
Efficient feature learning
Improved handwriting feature extraction
3. BiLSTM Sequence Modeling

After spatial features are extracted, the model converts feature maps into sequential representations.

nn.LSTM(
    input_size=num_classes,
    hidden_size=256,
    num_layers=2,
    bidirectional=True
)


The BiLSTM captures:

Left-to-right dependencies
Right-to-left dependencies
Contextual word understanding
4. CTC-Based Learning

The model uses:

nn.CTCLoss()


Advantages:

No need for character-level segmentation
No alignment requirements
Suitable for handwritten sequence transcription
5. Medical Lexicon Post-Processor

A domain-specific correction module improves transcription quality.

Features:

Confidence-aware correction
Levenshtein distance matching
Nearest-neighbor lexicon search

Example medical lexicon:

{
    "PARACETAMOL",
    "AMOXICILLIN",
    "DIGOXIN",
    "IBUPROFEN",
    "500MG",
    "10ML",
    "TID",
    "BID",
    "PRN",
    "CAPSULE",
    "TABLET"
}

🔍 Example Error Corrections
Example 1

Input:

Diqoxin 5omg BID


Output:

DIGOXIN 500MG BID

Example 2

Input:

AMOXICILN 250mg


Output:

AMOXICILLIN 250mg

Example 3

Input:

Zyxyx 10ml


Output:

Zyxyx 10ml


(No suitable lexicon match found.)

# Project Structure
project/
│
├── main.py
├── images/
├── annotations.txt
├── README.md
│
└── outputs/

# Installation
Clone Repository
git clone https://github.com/yourusername/prescription-htr.git
cd prescription-htr

Create Environment
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

Install Dependencies
pip install torch torchvision numpy opencv-python

# Running the Project

Execute:

python main.py


Example output:

Using device: cuda

--- 6.1 Data Simulation ---

--- 6.2 Residual U-Net Setup ---
Model instantiated with 3,000,000+ parameters

--- 6.3 Mock Training Step ---
Simulated Training Loss (CTC): 2.8314

--- 6.4 Lexicon Post-Processing Demo ---
Raw 1: 'Diqoxin 5omg BID'
Corrected 1: 'DIGOXIN 500MG BID'

Raw 2: 'AMOXICILN 250mg'
Corrected 2: 'AMOXICILLIN 250mg'

Raw 3: 'Zyxyx 10ml'
Corrected 3: 'Zyxyx 10ml'

# Research Contributions

This implementation demonstrates:

Deep neural architectures for medical handwriting recognition
Integration of Residual U-Net and BiLSTM networks
CTC-based sequence learning
Clinical error reduction through lexicon-guided correction
A fully self-contained reproducible research prototype
# Future Improvements
Beam Search Decoding
Transformer-based Sequence Modeling
Attention Mechanisms
Confidence Calibration
Real Kaggle Dataset Integration
Model Quantization for Edge Deployment
Clinical Lexicon Expansion
Multi-Language Prescription Recognition
# Technologies Used
Technology	PurposePyTorch	Deep Learning Framework
OpenCV	Image Processing
NumPy	Numerical Computing
BiLSTM	Sequence Learning
Residual U-Net	Feature Extraction
CTC Loss	Sequence Alignment
Levenshtein Distance	Error Correction
# License

This project is provided for academic, educational, and research purposes. Please review and update the license according to your intended use before public deployment.

Developed as a research-oriented implementation of a medical handwritten prescription recognition system combining deep learning and clinical post-processing techniques.

# If you found this project useful, consider starring the repository and contributing improvements to advance AI-assisted healthcare transcription systems.
