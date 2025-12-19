# Model Files

This directory contains the trained XLM-RoBERTa model for toxic content classification.

## ⚠️ Important Notice

**Model weights are NOT included in this repository** due to GitHub's 100MB file size limit. The model file (`model.safetensors`) is approximately 1GB.

## What's Included

The following configuration files are included in the repository:

- ✅ `config.json` - Model configuration
- ✅ `tokenizer_config.json` - Tokenizer configuration
- ✅ `tokenizer.json` - Tokenizer vocabulary
- ✅ `special_tokens_map.json` - Special tokens mapping
- ✅ `classification_report.json` - Model evaluation metrics
- ✅ `test_metrics.json` - Test set metrics
- ✅ `validation_metrics.json` - Validation set metrics
- ✅ `training_metrics.json` - Training metrics

## What You Need to Add

You need to download and place the following file manually:

- ❌ `model.safetensors` (~1GB) - Model weights

## How to Get the Model

### Option 1: Download from Your Source

If you have the model file, simply place it in this directory:

```
models/xlm-roberta-toxic-classifier/model.safetensors
```

### Option 2: Use Hugging Face

If the model is available on Hugging Face:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "your-model-name/xlm-roberta-toxic-classifier"
model_path = "./models/xlm-roberta-toxic-classifier"

# Download model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
```

### Option 3: Train Your Own

If you need to train the model, refer to the training scripts (if available) or use the Hugging Face Transformers library to fine-tune XLM-RoBERTa on your toxic content dataset.

## Verification

After adding `model.safetensors`, verify the model loads correctly:

```python
from transformers import AutoModel, AutoTokenizer

model_path = "./models/xlm-roberta-toxic-classifier"
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("✅ Model loaded successfully!")
```

## Directory Structure

After adding the model file, your directory should look like:

```
models/xlm-roberta-toxic-classifier/
├── config.json
├── model.safetensors          # ← You need to add this
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
├── classification_report.json
├── test_metrics.json
├── validation_metrics.json
└── training_metrics.json
```

## File Size Information

- `model.safetensors`: ~1GB (not in repo)
- All other files: <10MB total (included in repo)

## Troubleshooting

**Error: Model file not found**

- Ensure `model.safetensors` is in the correct directory
- Check file permissions
- Verify the file is not corrupted

**Error: Model loading fails**

- Ensure all config files are present
- Check that the model version matches the config files
- Verify PyTorch/Transformers versions are compatible
