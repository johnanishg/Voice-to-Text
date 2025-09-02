#!/usr/bin/env python3
"""
Inference script for the fine-tuned Whisper model on Svarah dataset
"""

import os
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import json
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def load_model(model_path="whisper_svarah_model"):
    """Load the fine-tuned Whisper model and processor"""
    print(f"Loading model from {model_path}...")
    
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return processor, model, device

def load_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file"""
    print(f"Loading audio: {audio_path}")
    
    # For m4a files, try using ffmpeg directly
    if audio_path.lower().endswith('.m4a'):
        try:
            import subprocess
            import tempfile
            
            # Use ffmpeg to convert m4a to wav
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_wav = tmp_file.name
            
            # Convert m4a to wav using ffmpeg
            cmd = ['ffmpeg', '-i', audio_path, '-ar', str(target_sr), '-ac', '1', '-y', temp_wav]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load the converted wav file
            audio, sr = librosa.load(temp_wav, sr=target_sr)
            
            # Clean up temporary file
            os.unlink(temp_wav)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("FFmpeg not available, trying librosa...")
            audio, sr = librosa.load(audio_path, sr=target_sr, res_type='kaiser_best')
    else:
        # For other formats, use librosa directly
        audio, sr = librosa.load(audio_path, sr=target_sr, res_type='kaiser_best')
    
    # Normalize audio
    audio = librosa.util.normalize(audio)
    
    return audio, sr

def transcribe_audio(processor, model, audio, device, language="en"):
    """Transcribe audio using the fine-tuned model"""
    print("Transcribing audio...")
    
    # Prepare input
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    # Generate token ids
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language=language,
            task="transcribe",
            do_sample=False,
            max_length=448,
            return_timestamps=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            attention_mask=None  # Explicitly set to None to avoid warning
        )
    
    # Decode the generated ids to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

def test_with_sample_audio():
    """Test the model with sample audio files from Svarah dataset"""
    processor, model, device = load_model()
    
    # Test with a few sample audio files
    audio_dir = Path("Svarah/audio")
    transcript_dir = Path("Svarah/transcripts")
    
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return
    
    # Get first few audio files
    audio_files = list(audio_dir.glob("*.wav"))[:5]
    
    results = []
    
    for audio_file in audio_files:
        print(f"\n{'='*50}")
        print(f"Processing: {audio_file.name}")
        
        try:
            # Load audio
            audio, sr = load_audio(str(audio_file))
            
            # Transcribe
            transcription = transcribe_audio(processor, model, audio, device)
            
            # Load ground truth if available
            transcript_file = transcript_dir / f"{audio_file.stem}.txt"
            ground_truth = ""
            if transcript_file.exists():
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    ground_truth = f.read().strip()
            
            result = {
                "audio_file": audio_file.name,
                "predicted": transcription,
                "ground_truth": ground_truth,
                "audio_length": len(audio) / sr
            }
            
            results.append(result)
            
            print(f"Audio length: {result['audio_length']:.2f}s")
            print(f"Predicted: {transcription}")
            if ground_truth:
                print(f"Ground truth: {ground_truth}")
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
    
    # Save results
    with open("inference_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Results saved to inference_results.json")
    print(f"Processed {len(results)} audio files")

def test_with_custom_audio(audio_path):
    """Test the model with a custom audio file"""
    processor, model, device = load_model()
    
    print(f"\n{'='*50}")
    print(f"Testing with custom audio: {audio_path}")
    
    try:
        # Load audio
        audio, sr = load_audio(audio_path)
        
        # Transcribe
        transcription = transcribe_audio(processor, model, audio, device)
        
        print(f"Audio length: {len(audio) / sr:.2f}s")
        print(f"Transcription: {transcription}")
        
        return transcription
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference script for fine-tuned Whisper model")
    parser.add_argument("--audio", type=str, help="Path to custom audio file to transcribe")
    parser.add_argument("--test-samples", action="store_true", help="Test with sample audio files from Svarah dataset")
    
    args = parser.parse_args()
    
    if args.audio:
        test_with_custom_audio(args.audio)
    elif args.test_samples:
        test_with_sample_audio()
    else:
        # Default: test with sample audio files
        test_with_sample_audio() 