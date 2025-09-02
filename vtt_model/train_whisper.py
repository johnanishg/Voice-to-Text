import os
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import TrainerCallback
import librosa
import numpy as np
import logging
import random
import gc
import time

# Try to import soundfile for faster loading
try:
    import soundfile as sf
    USE_SOUNDFILE = True
    print("✅ Using soundfile for fast audio loading")
except ImportError:
    USE_SOUNDFILE = False
    print("⚠️  soundfile not found - using slower librosa. Install with: pip install soundfile")

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastSvarahDataset(Dataset):
    def __init__(self, audio_dir: str, transcript_dir: str, processor: WhisperProcessor, file_list=None):
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        self.processor = processor
        
        if file_list is not None:
            self.sample_files = file_list
        else:
            self.sample_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        
        # Validate files exist
        valid_files = []
        for audio_file in self.sample_files:
            audio_path = os.path.join(audio_dir, audio_file)
            transcript_path = os.path.join(transcript_dir, audio_file.replace('.wav', '.txt'))
            if os.path.exists(audio_path) and os.path.exists(transcript_path):
                valid_files.append(audio_file)
        
        self.sample_files = valid_files
        logger.info(f"Found {len(self.sample_files)} valid audio-transcript pairs")
        
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        try:
            audio_file = os.path.join(self.audio_dir, self.sample_files[idx])
            transcript_file = os.path.join(self.transcript_dir, self.sample_files[idx].replace('.wav', '.txt'))
            
            # FASTEST audio loading
            if USE_SOUNDFILE:
                speech, sr = sf.read(audio_file)
                if sr != 16000:
                    speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            else:
                speech, sr = librosa.load(audio_file, sr=16000)
            
            if len(speech) == 0:
                speech = np.zeros(16000)
            
            # Load transcript
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                text = "<|nospeech|>"
            
            # Process inputs
            inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
            labels = self.processor.tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=448)
            
            return {
                "input_features": inputs["input_features"].squeeze(0),
                "labels": labels["input_ids"].squeeze(0)
            }
            
        except Exception as e:
            logger.error(f"Error processing {self.sample_files[idx]}: {e}")
            dummy_audio = np.zeros(16000)
            inputs = self.processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
            labels = self.processor.tokenizer("<|nospeech|>", return_tensors="pt")
            return {
                "input_features": inputs["input_features"].squeeze(0),
                "labels": labels["input_ids"].squeeze(0)
            }


class FastDataCollator:
    def __init__(self, processor, model_dtype=None):
        self.processor = processor
        self.model_dtype = model_dtype
    
    def __call__(self, features):
        input_features = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]
        
        batch = self.processor.feature_extractor.pad({"input_features": input_features}, return_tensors="pt")
        
        if self.model_dtype:
            batch["input_features"] = batch["input_features"].to(dtype=self.model_dtype)
        
        max_label_length = max(len(label) for label in labels)
        padded_labels = []
        
        for label in labels:
            padded_label = torch.full((max_label_length,), self.processor.tokenizer.pad_token_id, dtype=label.dtype)
            padded_label[:len(label)] = label
            padded_labels.append(padded_label)
        
        batch["labels"] = torch.stack(padded_labels)
        batch["labels"] = batch["labels"].masked_fill(batch["labels"] == self.processor.tokenizer.pad_token_id, -100)
        
        return batch


class SpeedCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.step_times = []
        self.last_step_time = None
        self.last_train_loss = None
        self.last_eval_loss = None
        self.last_wer = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_step_time = time.time()
        print(f"\n🚀 SPEED-OPTIMIZED Training | Steps: {state.max_steps}")
        print("Step | Progress | Train Loss | Val Loss | WER | Accuracy | Speed | ETA")
        print("-" * 75)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.start_time:
            # Update stored values
            if 'loss' in logs:
                self.last_train_loss = logs['loss']
            if 'eval_loss' in logs:
                self.last_eval_loss = logs['eval_loss']
            if 'eval_wer' in logs:
                self.last_wer = logs['eval_wer']
            
            if len(self.step_times) >= 5:
                avg_time = sum(self.step_times[-5:]) / 5
                speed = 1.0 / avg_time if avg_time > 0 else 0
            else:
                speed = 0
            
            progress = (state.global_step / state.max_steps) * 100
            remaining = state.max_steps - state.global_step
            eta_min = (remaining / speed / 60) if speed > 0 else 999
            
            # Format values
            train_loss_str = f"{self.last_train_loss:.3f}" if self.last_train_loss else "---"
            val_loss_str = f"{self.last_eval_loss:.3f}" if self.last_eval_loss else "---"
            wer_str = f"{self.last_wer:.3f}" if self.last_wer else "---"
            acc_str = f"{(1-self.last_wer)*100:.1f}%" if self.last_wer else "---"
            
            print(f"{state.global_step:4d} | {progress:5.1f}% | {train_loss_str:9s} | {val_loss_str:8s} | {wer_str:6s} | {acc_str:7s} | {speed:.1f}/s | {eta_min:.0f}m")

    def on_step_end(self, args, state, control, **kwargs):
        if self.last_step_time:
            current_time = time.time()
            step_time = current_time - self.last_step_time
            self.step_times.append(step_time)
            if len(self.step_times) > 10:
                self.step_times.pop(0)
            self.last_step_time = current_time
        else:
            self.last_step_time = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time:
            total_time = time.time() - self.start_time
            print("-" * 75)
            print(f"🎉 Training completed in {total_time/3600:.1f} hours!")


class OptimizedTrainer(Seq2SeqTrainer):
    """Custom trainer with memory optimizations"""
    
    def training_step(self, model, inputs):
        result = super().training_step(model, inputs)
        
        # Clear cache every 25 steps for stability
        if self.state.global_step % 25 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
        return result


def compute_metrics(pred, processor, wer_metric):
    """Compute WER metrics"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token id for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer_scores = []
    for pred_text, label_text in zip(pred_str, label_str):
        if label_text.strip():
            wer_score = wer_metric.compute(predictions=[pred_text], references=[label_text])
            wer_scores.append(wer_score)
    
    avg_wer = np.mean(wer_scores) if wer_scores else 1.0
    
    return {"wer": avg_wer}


if __name__ == "__main__":
    # Setup
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Install and setup WER metric
    try:
        import jiwer
        print("Using jiwer for WER computation")
        
        class WERMetric:
            def compute(self, predictions, references):
                if isinstance(predictions, list) and len(predictions) == 1:
                    predictions = predictions[0]
                if isinstance(references, list) and len(references) == 1:
                    references = references[0]
                return jiwer.wer(references, predictions)
        
        wer_metric = WERMetric()
        
    except ImportError:
        print("Installing jiwer...")
        os.system("pip install jiwer")
        import jiwer
        
        class WERMetric:
            def compute(self, predictions, references):
                if isinstance(predictions, list) and len(predictions) == 1:
                    predictions = predictions[0]
                if isinstance(references, list) and len(references) == 1:
                    references = references[0]
                return jiwer.wer(references, predictions)
        
        wer_metric = WERMetric()
    
    # Install soundfile if not available
    if not USE_SOUNDFILE:
        print("Installing soundfile for faster audio loading...")
        os.system("pip install soundfile")
        try:
            import soundfile as sf
            USE_SOUNDFILE = True
            print("✅ soundfile installed successfully")
        except ImportError:
            print("❌ Failed to install soundfile, using librosa")
    
    # Load model
    model_name = "openai/whisper-small"
    print(f"Loading model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.use_cache = False
    
    # Optimize model precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if torch.cuda.is_available():
        if use_bf16:
            model = model.to(device, dtype=torch.bfloat16)
            model_dtype = torch.bfloat16
            print("Using bfloat16 precision")
        else:
            model = model.to(device)
            model_dtype = torch.float16
            print("Using float16 precision")
    else:
        model = model.to(device)
        model_dtype = torch.float32
        print("Using CPU")
    
    # Set paths
    AUDIO_DIR = "svarah/audio"
    TRANSCRIPT_DIR = "svarah/transcripts"
    OUTPUT_DIR = "whisper_svarah_model"
    
    # Validate directories
    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")
    if not os.path.exists(TRANSCRIPT_DIR):
        raise FileNotFoundError(f"Transcript directory not found: {TRANSCRIPT_DIR}")
    
    # Prepare data
    all_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    if not all_files:
        raise ValueError(f"No .wav files found in {AUDIO_DIR}")
    
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    logger.info(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")
    
    # Create datasets
    train_dataset = FastSvarahDataset(AUDIO_DIR, TRANSCRIPT_DIR, processor, train_files)
    val_dataset = FastSvarahDataset(AUDIO_DIR, TRANSCRIPT_DIR, processor, val_files)
    
    # MAXIMUM SPEED settings
    batch_size = 2  # Increased for speed
    
    # Optimized training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,  # No accumulation for speed
        learning_rate=3e-5,  # Slightly higher for faster convergence
        warmup_steps=200,  # Minimal warmup
        num_train_epochs=5,  # Full training
        gradient_checkpointing=False,  # Disabled for speed
        fp16=not use_bf16,
        bf16=use_bf16,
        
        # Evaluation settings
        evaluation_strategy="steps",
        eval_steps=500,  # Show validation metrics
        save_steps=1000,
        logging_steps=50,  # More frequent logging
        
        # Generation settings
        predict_with_generate=True,
        generation_max_length=448,
        generation_num_beams=1,  # Greedy decoding for speed
        
        # Performance optimizations
        dataloader_num_workers=0,  # No multiprocessing overhead
        dataloader_pin_memory=False,  # Disabled for speed
        
        # Training optimizations
        max_grad_norm=1.0,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        
        # Model saving
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        
        # Logging
        disable_tqdm=False,
        report_to=[],
    )
    
    # Initialize components
    data_collator = FastDataCollator(processor, model_dtype)
    
    def compute_metrics_fn(pred):
        return compute_metrics(pred, processor, wer_metric)
    
    # Initialize optimized trainer
    trainer = OptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
        callbacks=[SpeedCallback()],
    )
    
    # Print configuration
    print(f"\n{'='*75}")
    print("🏎️ MAXIMUM SPEED TRAINING CONFIGURATION")
    print(f"{'='*75}")
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {training_args.learning_rate}")
    print(f"Workers: {training_args.dataloader_num_workers} (single-threaded)")
    print(f"Soundfile: {'✅ Yes' if USE_SOUNDFILE else '❌ No'}")
    print(f"Precision: {'bfloat16' if use_bf16 else 'float16'}")
    print(f"Gradient Checkpointing: {training_args.gradient_checkpointing}")
    print(f"Target Speed: 3-5 steps/sec")
    print(f"Training Steps: {len(train_dataset) // batch_size * training_args.num_train_epochs}")
    print(f"{'='*75}\n")
    
    # Start training
    logger.info("Starting MAXIMUM SPEED training...")
    trainer.train()
    
    # Save model
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
    gc.collect()
    
    print("\n🎉 Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
