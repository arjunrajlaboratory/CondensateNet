# CondensateNet

Deep learning for biomolecular condensate segmentation in microscopy images.

## Installation

```bash
# From GitHub (recommended for now)
pip install git+https://github.com/arjunrajlaboratory/condensatenet.git

# From PyPI (coming soon)
# pip install condensatenet
```

## Quick Start

```python
from condensatenet import CondensateNetPipeline

# Load the model (downloads from HuggingFace on first use)
pipeline = CondensateNetPipeline.from_pretrained()

# Segment an image
instances = pipeline.segment(image)
print(f"Found {instances.max()} condensates")
```

## Usage

### Basic Segmentation

```python
from condensatenet import CondensateNetPipeline
import numpy as np

# Load pipeline
pipeline = CondensateNetPipeline.from_pretrained()

# Your microscopy image (numpy array)
image = ...  # shape: (H, W)

# Segment condensates
instances = pipeline.segment(image)

# instances is a numpy array of shape (H, W)
# Background = 0, condensates labeled 1, 2, 3, ...
num_condensates = instances.max()
```

### Full Output

```python
# Get probability maps and flow vectors too
result = pipeline.segment(image, full_output=True)

print(result['num_instances'])      # Number of condensates
print(result['instances'].shape)    # (H, W) instance labels
print(result['mask_probs'].shape)   # (H, W) probability map
print(result['flows'].shape)        # (2, H, W) flow vectors
```

### Custom Parameters

```python
# Override default parameters
instances = pipeline.segment(
    image,
    prob_threshold=0.2,   # Higher = fewer detections
    min_size=20,          # Minimum condensate size (pixels)
    max_size=500,         # Maximum condensate size (pixels)
)

# Or set defaults at pipeline creation
pipeline = CondensateNetPipeline.from_pretrained(
    prob_threshold=0.2,
    min_size=20
)
```

### Step-by-Step Processing

```python
# For more control, run steps separately
normalized = pipeline.preprocess(image)
mask_probs, flows = pipeline.predict(normalized, skip_preprocessing=True)
instances = pipeline.postprocess(mask_probs, flows, prob_threshold=0.25)
```

### Device Selection

```python
# Auto-detect (default) - uses CUDA if available
pipeline = CondensateNetPipeline.from_pretrained()

# Force specific device
pipeline = CondensateNetPipeline.from_pretrained(device="cuda")
pipeline = CondensateNetPipeline.from_pretrained(device="cpu")
pipeline = CondensateNetPipeline.from_pretrained(device="mps")  # Apple Silicon
```

## Docker Deployment

For Docker images, download the model at build time:

```dockerfile
# In your Dockerfile
RUN pip install git+https://github.com/arjunrajlaboratory/condensatenet.git
RUN python -m condensatenet download --output /models/condensatenet
```

Then load from local path at runtime:

```python
from condensatenet import CondensateNetPipeline

pipeline = CondensateNetPipeline.from_local("/models/condensatenet")
instances = pipeline.segment(image)
```

## Command Line Interface

```bash
# Download model to specific location
python -m condensatenet download --output /models/condensatenet

# Download to default cache
python -m condensatenet download

# Show model information
python -m condensatenet info
```

## Standalone Functions

```python
from condensatenet import normalize_image, flow_to_instances, download_model

# Normalize an image (percentile-based)
normalized = normalize_image(image, low_percentile=0.5, high_percentile=99.5)

# Convert model outputs to instances
instances = flow_to_instances(mask_probs, flows, prob_threshold=0.15)

# Download model to specific directory
download_model(output_dir="/models/condensatenet")
```

## Model Details

CondensateNet uses:
- **Encoder**: EfficientNetV2-S backbone (single channel input)
- **Decoder**: Style-modulated Feature Pyramid Network
- **Output**: Binary segmentation mask + 2D flow vectors for instance separation

The model is hosted on HuggingFace: [rajlab/condensatenet](https://huggingface.co/rajlab/condensatenet)

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- transformers >= 4.30
- timm >= 0.9
- scikit-image >= 0.19
- numpy >= 1.21

## License

MIT License

## Citation

If you use CondensateNet in your research, please cite:

```bibtex
@software{condensatenet,
  author = {Raj Lab},
  title = {CondensateNet: Deep Learning for Biomolecular Condensate Segmentation},
  url = {https://github.com/arjunrajlaboratory/condensatenet},
  year = {2024}
}
```
