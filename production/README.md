# Production Code - Depth-Aware Portrait Effects

This folder contains the **production-ready implementations** of depth-aware portrait effects using MiDaS and Depth-Anything models.

## 📁 Folder Structure

```
production/
├── midas/
│   ├── midas_portrait.py    # MiDaS portrait processor
│   └── __init__.py
├── depth_anything/
│   ├── depth_anything_portrait.py  # Depth-Anything portrait processor
│   └── __init__.py
└── README.md                 # This file
```

## 🚀 Quick Start

### Using the Production Processors Directly

#### MiDaS Processor

```python
from production.midas.midas_portrait import MiDaSPortraitProcessor

# Initialize processor
processor = MiDaSPortraitProcessor(device='cpu')  # or 'cuda' for GPU

# Process a single image
results = processor.process_image(
    image_path='data/portrait1.jpg',
    output_dir='results/midas',
    strategies=['conservative', 'face_aware', 'adaptive']
)

# Access results
for strategy, result_data in results.items():
    print(f"{strategy}: {result_data['processing_time']:.2f}s")
    # result_data contains: 'result', 'depth_map', 'mask', 'processing_time'
```

#### Depth-Anything Processor

```python
from production.depth_anything.depth_anything_portrait import DepthAnythingPortraitProcessor

# Initialize processor
processor = DepthAnythingPortraitProcessor(device='cpu', encoder='vits')

# Process a single image
results = processor.process_image(
    image_path='data/portrait1.jpg',
    output_dir='results/depth_anything',
    strategies=['conservative', 'face_aware', 'adaptive']
)
```

### Running Individual Modules

#### Run MiDaS on a Single Image

```bash
# From project root
python -m production.midas.midas_portrait data/portrait1.jpg
```

#### Run Depth-Anything on a Single Image

```bash
# From project root
python -m production.depth_anything.depth_anything_portrait data/portrait1.jpg
```

### Using the Main Comparison Script

The easiest way to use production code is through the main comparison script:

```bash
# Process single image with both approaches
python portrait_comparison.py --single data/portrait1.jpg

# Process all images
python portrait_comparison.py --all

# Run comprehensive comparison
python portrait_comparison.py --compare
```

## 📖 API Reference

### MiDaSPortraitProcessor

**Class**: `production.midas.midas_portrait.MiDaSPortraitProcessor`

**Initialization:**
```python
processor = MiDaSPortraitProcessor(device='cpu')
```

**Methods:**

- `estimate_depth(image_bgr)`: Estimate depth map from BGR image
- `create_portrait_effect(image_bgr, depth_map, face_mask=None, strategy='conservative', threshold=None)`: Create portrait effect
- `process_image(image_path, output_dir, strategies=['conservative', 'face_aware', 'adaptive'])`: Process complete image pipeline

### DepthAnythingPortraitProcessor

**Class**: `production.depth_anything.depth_anything_portrait.DepthAnythingPortraitProcessor`

**Initialization:**
```python
processor = DepthAnythingPortraitProcessor(device='cpu', encoder='vits')
```

**Parameters:**
- `device`: 'cpu' or 'cuda'
- `encoder`: 'vits' (small), 'vitb' (base), or 'vitl' (large)

**Methods:**

- `estimate_depth(image_bgr)`: Estimate depth map from BGR image
- `create_portrait_effect(image_bgr, depth_map, face_mask=None, strategy='conservative', threshold=None)`: Create portrait effect
- `process_image(image_path, output_dir, strategies=['conservative', 'face_aware', 'adaptive'])`: Process complete image pipeline

## 🎯 Processing Strategies

Both processors support three blurring strategies:

1. **Conservative** (`threshold=0.8`): Gentle background blur, keeps more foreground sharp
2. **Face-Aware** (`threshold=0.2`): Aggressive blur with face protection
3. **Adaptive** (`threshold=85th percentile`): Dynamic threshold based on depth distribution

## 📊 Output Files

For each processed image, the following files are generated:

- `{image_name}_{approach}_original.jpg` - Original image
- `{image_name}_{approach}_conservative.jpg` - Conservative strategy result
- `{image_name}_{approach}_conservative_depth.jpg` - Depth map visualization
- `{image_name}_{approach}_conservative_mask.jpg` - Blur mask visualization
- `{image_name}_{approach}_face_aware.jpg` - Face-aware strategy result
- `{image_name}_{approach}_face_aware_depth.jpg` - Face-aware depth map
- `{image_name}_{approach}_face_aware_mask.jpg` - Face-aware blur mask
- `{image_name}_{approach}_adaptive.jpg` - Adaptive strategy result
- `{image_name}_{approach}_adaptive_depth.jpg` - Adaptive depth map
- `{image_name}_{approach}_adaptive_mask.jpg` - Adaptive blur mask
- `{image_name}_{approach}_face_mask.jpg` - Detected face regions (if faces found)

## ⚙️ Configuration

### Device Selection

```python
# CPU (default)
processor = MiDaSPortraitProcessor(device='cpu')

# GPU (if CUDA available)
processor = MiDaSPortraitProcessor(device='cuda')
```

### Custom Thresholds

```python
# Use custom threshold instead of strategy defaults
result, mask, depth = processor.create_portrait_effect(
    image, depth_map, face_mask,
    strategy='conservative',
    threshold=0.7  # Custom threshold
)
```

## 🔍 Differences from Tutorials (chapter19/)

| Aspect | Production Code | Tutorials (chapter19/) |
|--------|----------------|------------------------|
| **Purpose** | Production-ready, optimized | Educational, step-by-step |
| **Complexity** | Complete implementation | Simplified examples |
| **Comments** | Minimal, focused | Extensive explanations |
| **Error Handling** | Robust, production-grade | Basic, for learning |
| **Performance** | Optimized | Demonstrative |
| **Use Case** | Real applications | Learning Chapter 19 |

## 📚 Reference

- **Chapter 19, Section 19.2.1**: MiDaS Implementation
- **Chapter 19, Section 19.2.2**: Depth-Anything Implementation
- **Chapter 19, Section 19.4**: Portrait Effect Creation

For educational tutorials, see the [`chapter19/`](../chapter19/) directory.


