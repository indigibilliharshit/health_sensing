# Health Sensing Assignment - Breathing Irregularity Detection During Sleep

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project for automated detection of breathing irregularities during sleep using physiological signals. This project implements deep learning models (1D CNN, Conv-LSTM, and Transformer) for both breathing irregularity detection and sleep stage classification.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project addresses **Scenario 2: Breathing Irregularity Detection During Sleep** as part of a health sensing assignment. The system analyzes overnight sleep data from 5 participants to detect breathing irregularities including:

- **Hypopnea**: Reduced airflow during sleep
- **Obstructive Apnea**: Complete blockage of airflow
- **Normal Breathing**: Regular breathing patterns

### Key Features

- ✅ Multi-rate physiological signal processing (32Hz nasal flow, thoracic movement; 4Hz SpO₂)
- ✅ Advanced digital filtering with Butterworth bandpass filters
- ✅ Leave-One-Participant-Out Cross-Validation for clinical relevance
- ✅ Class imbalance handling with weighted training
- ✅ Comprehensive visualization system
- ✅ Bonus: Sleep stage classification (Wake, N1, N2, N3, REM)

## 📊 Dataset

### Breathing Irregularity Dataset
- **Total Windows**: 8,800 (30-second windows with 50% overlap)
- **Classes**: 
  - Normal: 8,046 (91.4%)
  - Hypopnea: 593 (6.7%)
  - Obstructive Apnea: 161 (1.8%)

### Sleep Stage Dataset (Bonus)
- **Total Windows**: 8,751
- **Classes**:
  - Wake: 3,273 (37.4%)
  - N2: 2,442 (27.9%)
  - N1: 1,320 (15.1%)
  - N3: 1,066 (12.2%)
  - REM: 650 (7.4%)

### Participants
- **AP01-AP05**: 5 participants with overnight sleep recordings
- **Signals**: Nasal Airflow, Thoracic Movement, SpO₂
- **Annotations**: Breathing events and sleep stages

## 📁 Project Structure

```
Health-Sensing-Assignment/
├── After_Filtering_Visualizations/          # Filtered signal visualizations
│   ├── After_Cleaning_Visualizations/
│   │   ├── AP01_After_Cleaning_visual.pdf
│   │   ├── AP02_After_Cleaning_visual.pdf
│   │   ├── AP03_After_Cleaning_visual.pdf
│   │   ├── AP04_After_Cleaning_visual.pdf
│   │   └── AP05_After_Cleaning_visual.pdf
│   └── Scripts_for_Filtered_Visualization/
│       └── filter_clean.py                  # Signal filtering script
├── Bonus_sleep_stage/                       # Sleep stage classification (Bonus Task)
│   ├── Dataset/
│   │   └── sleep_stage_dataset.csv         # Sleep stage dataset
│   └── Scripts/
│       ├── create_sleep_stage_dataset.py   # Sleep stage dataset creation
│       └── train_model.py                  # Sleep stage model training
├── Scripts/                                # Main breathing irregularity scripts
│   ├── create_dataset.py                   # Breathing dataset creation
│   ├── train_model.py                      # Breathing irregularity training
│   └── vis.py                              # Visualization script
├── Data/                                   # Raw participant data
│   ├── AP01/
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
├── Dataset/
│   └── breathing_dataset.csv               # Main breathing irregularity dataset
├── Models/
│   ├── cnn_model.py                        # 1D CNN implementation
│   └── conv_lstm_model.py                  # Conv-LSTM implementation
├── Visualizations/                         # Generated visualizations
│   ├── AP01_visualization.pdf
│   ├── AP02_visualization.pdf
│   ├── AP03_visualization.pdf
│   ├── AP04_visualization.pdf
│   ├── AP05_visualization.pdf
│   └── attributes                          # Visualization attributes
├── health_sensing_report.pdf               # Comprehensive project report
├── requirements.txt                        # Python dependencies
├── LICENSE                                 # MIT License
└── README.md                              # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (recommended for training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/health-sensing-assignment.git
cd health-sensing-assignment
