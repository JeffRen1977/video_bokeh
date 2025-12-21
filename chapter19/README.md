# Chapter 19: Depth-Aware Portrait Effects - Educational Implementation

This directory contains educational implementations and examples that help you understand Chapter 19 of "Mobile Computational Photography" book.

## 📚 Learning Path

Follow these tutorials in order to understand the concepts step by step:

### 1. **Basic Concepts** (`01_basic_concepts.py`)
   - Understanding depth maps
   - Basic depth visualization
   - Simple blurring techniques

### 2. **Face Detection** (`02_face_detection.py`)
   - OpenCV face detection
   - Creating face protection masks
   - Face-aware processing

### 3. **Depth Estimation - MiDaS** (`03_depth_estimation_midas.py`)
   - Loading and using MiDaS model
   - Understanding DPT architecture
   - Depth map interpretation

### 4. **Depth Estimation - Depth-Anything** (`04_depth_estimation_depth_anything.py`)
   - Loading and using Depth-Anything model
   - Understanding DINOv2 backbone
   - Comparison with MiDaS

### 5. **Portrait Effect Creation** (`05_portrait_effect.py`)
   - Combining depth estimation with blurring
   - Corrected depth logic
   - Multiple blurring strategies

### 6. **Complete Pipeline** (`06_complete_pipeline.py`)
   - End-to-end portrait effect creation
   - Performance optimization
   - Quality assessment

## 🎯 Key Concepts from Chapter 19

### Concept 1: Monocular Depth Estimation
- **What**: Estimating depth from a single image
- **Why**: Essential for creating realistic portrait effects
- **How**: Deep learning models (MiDaS, Depth-Anything)

### Concept 2: Depth Interpretation
- **Challenge**: Understanding what depth values mean
- **Solution**: Corrected logic (low = background, high = foreground)
- **Implementation**: Normalization and thresholding

### Concept 3: Face Protection
- **Why**: Faces must remain sharp in portraits
- **How**: Face detection + mask generation
- **Result**: Professional portrait effects

### Concept 4: Blurring Strategies
- **Conservative**: Gentle background blur
- **Face-Aware**: Aggressive blur with face protection
- **Adaptive**: Dynamic threshold based on depth distribution

## 📖 Reference to Book

This implementation corresponds to **Chapter 19** of "Mobile Computational Photography" book. The code demonstrates:

1. **Section 19.1**: Introduction to depth-aware portrait effects
2. **Section 19.2**: Monocular depth estimation techniques
3. **Section 19.3**: Face detection and protection
4. **Section 19.4**: Blurring strategies and implementation
5. **Section 19.5**: Performance optimization and mobile deployment

## 🚀 Quick Start

```bash
# Run basic concepts tutorial
python chapter19/01_basic_concepts.py

# Run face detection tutorial
python chapter19/02_face_detection.py

# Run complete pipeline
python chapter19/06_complete_pipeline.py
```

## 📝 Notes

- Each tutorial script is self-contained and can be run independently
- All scripts include detailed comments explaining Chapter 19 concepts
- Results are saved in `chapter19/results/` directory
- Compare your results with the book's examples

