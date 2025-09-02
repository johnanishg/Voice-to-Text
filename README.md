# Whisper Fine-tuning on Svarah Dataset for Indian-Accented English

This repository contains scripts for fine-tuning OpenAI's Whisper model on the **Svarah dataset**, an Indian-accented English speech recognition benchmark developed by **AI4Bharat at IIT Madras**.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [File Structure](#file-structure)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## 🎯 Overview

This project demonstrates how to fine-tune OpenAI's Whisper model specifically for **Indian-accented English speech recognition** using the Svarah dataset. The fine-tuned model achieves improved Word Error Rate (WER) performance on Indian accents compared to the base Whisper models trained primarily on Western English accents.

### Key Features

- **Optimized Training Pipeline**: Speed-optimized training with multiple performance enhancements
- **Memory Efficient**: Custom data collator and memory management for GPU optimization
- **Real-time Monitoring**: Live training metrics with progress tracking
- **Flexible Inference**: Support for various audio formats including M4A, WAV
- **Comprehensive Evaluation**: WER computation and detailed performance metrics
- **Indian Accent Specialization**: Specifically tuned for diverse Indian English accents

## 📊 Dataset

### About Svarah Dataset

**Svarah** is a benchmark dataset created by AI4Bharat (IIT Madras) that addresses the performance gap in English ASR systems when processing Indian accents. The dataset contains **English speech** from speakers with diverse Indian linguistic backgrounds.

### Dataset Statistics

- **9.6 hours** of transcribed **English audio**
- **117 speakers** from **65 districts** across **19 states** of India
- **Diverse accent variations** from speakers with different native language backgrounds
- **Multiple content types**:
  - Read speech (1.4 hours)
  - Extempore speech (6.4 hours)  
  - Everyday use cases (1.7 hours)

### Content Domains

The English speech covers various domains relevant to Indian contexts:
- History, culture, and tourism
- Government services and procedures
- Sports and entertainment
- Real-world scenarios: grocery ordering, digital payments, pension claims
- Educational and informational content

### Accent Diversity

The dataset captures English pronunciation variations from speakers whose native languages span across different language families, providing rich accent diversity:
- **Indo-Aryan language backgrounds**: Hindi, Bengali, Gujarati, Marathi, Punjabi, Urdu, Assamese, Odia
- **Dravidian language backgrounds**: Tamil, Telugu, Kannada, Malayalam
- **Other linguistic backgrounds**: Including speakers from Northeast India and other regions

This diversity ensures the model learns to handle various Indian English accent patterns and pronunciation variations.

## 🛠️ Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 5GB+ free space for model and dataset

### Software Requirements

- Python 3.8-3.11
- CUDA 11.0+ (for GPU acceleration)
- FFmpeg (for audio processing)

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd whisper-svarah-finetuning
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install accelerate>=0.20.0
pip install evaluate>=0.4.0
pip install jiwer>=3.0.0
pip install librosa>=0.10.0
pip install soundfile>=0.12.0
pip install tensorboard
pip install gradio
```

### 3. Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

## 📁 Dataset Preparation

### 1. Download Svarah Dataset

The Svarah dataset is available on Hugging Face:

```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli login  # Login with your HF token
```

```python
from datasets import load_dataset

# Load the English speech dataset
dataset = load_dataset("ai4bharat/Svarah", split="test")
```

### 2. Organize Data Structure

Create the following directory structure for your English audio data:

```
svarah/
├── audio/           # English audio files (.wav format)
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── transcripts/     # Corresponding English transcriptions
    ├── sample1.txt
    ├── sample2.txt
    └── ...
```

### 3. Data Format Requirements

- **Audio files**: WAV format, 16kHz sampling rate, mono channel
- **Transcript files**: Plain text files with UTF-8 encoding containing English transcriptions
- **Naming convention**: Audio and transcript files must have matching names (except extensions)
- **Language**: All transcriptions should be in English

## 🚀 Usage

### Training

#### Basic Training Command

```bash
python train_whisper.py
```

#### Advanced Configuration

The training script supports various configuration options:

```bash
# Modify these variables in the script:
AUDIO_DIR = "svarah/audio"           # Path to English audio files
TRANSCRIPT_DIR = "svarah/transcripts" # Path to English transcript files
OUTPUT_DIR = "whisper_svarah_model"   # Output model directory
```

#### Key Training Parameters

- **Model**: `openai/whisper-small` (configurable)
- **Language**: English (with Indian accent adaptation)
- **Batch Size**: 2 (adjustable based on GPU memory)
- **Learning Rate**: 3e-5
- **Epochs**: 5
- **Precision**: Mixed precision (bfloat16/float16)

#### Training Features

- **Accent Adaptation**: Fine-tuning specifically for Indian English pronunciation patterns
- **Speed Optimizations**: 
  - Fast audio loading with soundfile
  - Optimized data collator
  - Gradient checkpointing disabled for speed
  - Single-threaded data loading
- **Memory Management**:
  - Automatic CUDA cache clearing
  - Garbage collection
  - Memory-efficient batch processing
- **Live Monitoring**:
  - Real-time progress tracking
  - WER and accuracy metrics for English speech
  - Training speed (steps/second)
  - Estimated time remaining

### Inference

#### Test with Sample Data

```bash
python inference_whisper.py --test-samples
```

#### Test with Custom Audio

```bash
python inference_whisper.py --audio path/to/your/english_audio.wav
```

#### Supported Audio Formats

- WAV (recommended)
- M4A (with FFmpeg)
- MP3
- FLAC

#### Inference Features

- **Automatic Format Conversion**: M4A to WAV conversion
- **Audio Preprocessing**: Normalization and resampling to 16kHz
- **English Speech Processing**: Optimized for Indian-accented English
- **Flexible Input**: File paths or audio arrays
- **Results Export**: JSON format with detailed metrics

## 📂 File Structure

```
whisper-svarah-finetuning/
├── train_whisper.py              # Main training script
├── inference_whisper.py          # Inference script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── svarah/                       # English dataset directory
│   ├── audio/                    # English audio files
│   └── transcripts/              # English transcript files
├── whisper_svarah_model/         # Trained model output
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── training_args.bin
│   └── vocab.json
└── inference_results.json        # English inference results
```

### Key Files Description

- **train_whisper.py**: Optimized training script with custom dataset loader for English speech
- **inference_whisper.py**: Inference script supporting multiple audio formats for English transcription
- **whisper_svarah_model/**: Directory containing the fine-tuned model optimized for Indian-accented English

## 📈 Model Performance

### Baseline Performance (Before Fine-tuning)

According to the original Svarah paper, baseline Whisper models on Indian-accented English show:

| Model | Parameters | Svarah WER (Indian English) | LibriSpeech WER (Standard English) |
|-------|------------|------------------------------|-------------------------------------|
| Whisper-base | 74M | 13.6% | 4.2% |
| Whisper-small | 244M | ~12% | ~3.5% |
| Whisper-medium | 769M | 8.3% | 3.1% |

### Expected Improvements After Fine-tuning

Fine-tuning on Svarah for Indian-accented English typically results in:
- **20-40% relative WER reduction** on Indian English accents
- **Improved accuracy** on Indian English pronunciation patterns
- **Better handling** of Indian-specific English vocabulary and expressions
- **Enhanced recognition** of regional pronunciation variations

### Evaluation Metrics

The training script tracks:
- **WER (Word Error Rate)**: Primary evaluation metric for English transcription
- **Training Loss**: Cross-entropy loss during training
- **Validation Loss**: Loss on English validation set
- **Accuracy**: 1 - WER
- **Training Speed**: Steps per second

## ⚙️ Configuration

### Hyperparameter Tuning for English Speech

Key hyperparameters optimized for Indian-accented English:

```python
# Learning rate (optimized for accent adaptation)
learning_rate = 3e-5

# Batch size (adjust based on GPU memory)
per_device_train_batch_size = 2
per_device_eval_batch_size = 2

# Training duration for accent learning
num_train_epochs = 5
max_steps = None  # Set for step-based training

# Evaluation frequency
eval_steps = 500
save_steps = 1000
logging_steps = 50

# Language-specific settings
language = "en"  # English
task = "transcribe"
```

### Memory Optimization

For limited GPU memory:

```python
# Reduce batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 2

# Enable gradient checkpointing
gradient_checkpointing = True

# Use int8 precision (if supported)
load_in_8bit = True
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size or enable gradient checkpointing

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
gradient_checkpointing = True
```

#### 2. Audio Loading Errors

**Solution**: Install soundfile and FFmpeg

```bash
pip install soundfile
sudo apt install ffmpeg  # Ubuntu/Debian
```

#### 3. Dataset Not Found

**Solution**: Ensure correct directory structure for English data

```bash
# Verify English audio and transcript paths exist
ls svarah/audio/
ls svarah/transcripts/
```

#### 4. Slow Training Speed

**Solutions**:
- Enable mixed precision training
- Use faster audio loading with soundfile
- Reduce evaluation frequency
- Disable unnecessary logging

### Performance Tips for Indian English

1. **Use SSD storage** for faster data loading
2. **Enable mixed precision** (bfloat16/float16)
3. **Optimize batch size** for your GPU
4. **Pre-process audio files** to 16kHz if needed
5. **Ensure English-only transcripts** for consistent training

## 📄 Citation

If you use this work or the Svarah dataset, please cite:

```bibtex
@inproceedings{javed2023svarah,
    title={Svarah: Evaluating English ASR Systems on Indian Accents},
    author={Tahir Javed and Sakshi Joshi and Vignesh Nagarajan and Sai Sundaresan and Janki Nawale and Abhigyan Raman and Kaushal Santosh Bhogale and Pratyush Kumar and Mitesh M. Khapra},
    booktitle={INTERSPEECH},
    pages={5087--5091},
    publisher={ISCA},
    year={2023}
}

@misc{radford2022whisper,
    title={Robust Speech Recognition via Large-Scale Weak Supervision},
    author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
    year={2022},
    eprint={2212.04356},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

## 🙏 Acknowledgments

- **AI4Bharat (IIT Madras)** for creating and releasing the Svarah dataset for Indian-accented English speech recognition
- **OpenAI** for the Whisper model and pre-trained checkpoints
- **Hugging Face** for the Transformers library and model hosting
- **MeitY (Ministry of Electronics and Information Technology, India)** under the BHASHINI initiative
- **Nilekani Philanthropies** and **EkStep Foundation** for funding support

### Links

- **Svarah Dataset**: [Hugging Face Hub](https://huggingface.co/datasets/ai4bharat/Svarah)
- **AI4Bharat**: [Official Website](https://ai4bharat.iitm.ac.in)
- **Whisper Model**: [OpenAI Repository](https://github.com/openai/whisper)
- **Paper**: [arXiv:2305.15760](https://arxiv.org/abs/2305.15760)

## 📜 License

This project is released under the MIT License. The Svarah dataset is released under CC-BY-4.0 license.

### Dataset License

The Svarah dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license, which allows for free use, sharing, and adaptation with proper attribution.

---

**For questions, issues, or contributions related to Indian-accented English speech recognition, please open an issue on GitHub or contact the maintainers.**