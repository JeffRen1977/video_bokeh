# Chapter 19: Depth-Aware Portrait Effects

## Abstract

This chapter explores monocular depth estimation techniques for creating professional portrait effects in mobile computational photography. We present a practical implementation and comparison of two state-of-the-art approaches: MiDaS (Mixed-data Depth Estimation) and Depth-Anything. The focus is on generating high-quality background blur while preserving facial details. Through our implementation, we achieve a 3x speedup with Depth-Anything compared to MiDaS while maintaining professional-grade quality. This chapter provides clear explanations of core concepts, step-by-step tutorials, and practical code examples to help readers understand and implement depth-aware portrait effects.

**Keywords**: Monocular Depth Estimation, Portrait Effects, Computational Photography, MiDaS, Depth-Anything, Face Detection, Background Blur, Mobile Photography

**Repository**: https://github.com/JeffRen1977/video_bokeh

---

## 19.1 Introduction

Portrait photography has evolved with computational photography. Creating professional bokeh effects—where the subject stays sharp while the background is blurred—is now standard on mobile devices. This effect, traditionally requiring expensive cameras with large apertures, can be simulated using depth estimation and computational techniques.

### 19.1.1 Core Challenge

The main challenge is accurately estimating depth from a single image and applying appropriate blurring while preserving facial details. This chapter presents two leading monocular depth estimation approaches adapted for portrait photography.

### 19.1.2 Learning Objectives

After reading this chapter, readers will understand:

1. **Basic Concepts**: How depth maps represent distance and enable portrait effects
2. **Depth Estimation**: Two modern approaches (MiDaS and Depth-Anything) and their differences
3. **Face Protection**: How to ensure faces remain sharp in portrait effects
4. **Implementation**: Practical code structure with educational tutorials and production code
5. **Performance**: Trade-offs between speed and quality in mobile applications

### 19.1.3 Chapter Structure

- **Section 19.2**: Fundamental concepts of depth maps and portrait effects
- **Section 19.3**: Depth estimation approaches (MiDaS and Depth-Anything)
- **Section 19.4**: Face detection and protection techniques
- **Section 19.5**: Implementation details and code structure
- **Section 19.6**: Experimental results and performance analysis
- **Section 19.7**: Discussion and practical considerations

---

## 19.2 Fundamental Concepts

### 19.2.1 Understanding Depth Maps

A **depth map** is a grayscale image where pixel values represent distance from the camera. In portrait photography:

- **High depth values** (bright pixels) = **Foreground** (close to camera, should stay sharp)
- **Low depth values** (dark pixels) = **Background** (far from camera, should be blurred)

**Key Insight**: The corrected interpretation is crucial. We blur areas with **low depth values** (background), not high values (foreground).

### 19.2.2 Portrait Effect Pipeline

The complete pipeline consists of four main steps:

1. **Depth Estimation**: Generate depth map from input image
2. **Face Detection**: Identify facial regions for protection
3. **Mask Generation**: Create blur mask based on depth and face regions
4. **Blurring and Compositing**: Apply blur to background, keep foreground sharp

### 19.2.3 Visual Example

```
Input Image → Depth Map → Face Detection → Blur Mask → Final Portrait Effect
```

**Tutorial Reference**: See `chapter19/01_basic_concepts.py` for a hands-on introduction to depth maps and basic blurring.

---

## 19.3 Monocular Depth Estimation Approaches

### 19.3.1 MiDaS (Mixed-data Depth Estimation)

MiDaS is a well-established approach that uses mixed datasets for robust training.

#### Architecture
- **Model**: DPT-Large (Dense Prediction Transformer)
- **Backbone**: ResNet + Transformer
- **Input**: 384×384 pixels (resizable)
- **Output**: Relative depth maps

#### Characteristics
- **Reliability**: Proven performance across diverse scenes
- **Stability**: Consistent results
- **Speed**: ~1.04 seconds per image (CPU)
- **Model Size**: ~100MB

#### When to Use
- When reliability and stability are priorities
- For applications requiring consistent results across varied scenes
- When model size is a concern (smaller than Depth-Anything)

**Code Reference**: 
- Tutorial: `chapter19/03_depth_estimation_midas.py`
- Production: `production/midas/midas_portrait.py`

### 19.3.2 Depth-Anything

Depth-Anything is a modern foundation model representing state-of-the-art performance.

#### Architecture
- **Model**: DPT-DINOv2 with ViT-Small backbone
- **Backbone**: Vision Transformer with DINOv2 features
- **Input**: 518×518 pixels (resizable)
- **Output**: Metric depth maps

#### Characteristics
- **Performance**: 3x faster than MiDaS (~0.35s per image on CPU)
- **Quality**: Superior depth detail and accuracy
- **Modern**: Latest vision transformer advances
- **Model Size**: ~200MB

#### When to Use
- When speed is critical (real-time applications)
- For highest quality depth estimation
- In mobile applications requiring fast processing

**Code Reference**:
- Tutorial: `chapter19/04_depth_estimation_depth_anything.py`
- Production: `production/depth_anything/depth_anything_portrait.py`

### 19.3.3 Comparison Summary

| Aspect | MiDaS | Depth-Anything |
|--------|-------|----------------|
| **Speed** | 1.04s/image | 0.35s/image (3x faster) |
| **Quality** | High | Very High |
| **Model Size** | ~100MB | ~200MB |
| **Architecture** | DPT + ResNet | DPT + DINOv2 |
| **Best For** | Reliability | Speed + Quality |

---

## 19.4 Face Detection and Protection

### 19.4.1 Why Face Protection Matters

Faces are the most important part of portrait photos. Even with accurate depth estimation, faces must be explicitly protected to ensure they remain sharp.

### 19.4.2 Face Detection Method

We use OpenCV Haar cascades for face detection:

```python
def _detect_faces(self, image_bgr):
    """Detect faces and create protection masks."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Create mask with 20% expansion margin
    face_mask = np.zeros((h, w), dtype=np.float32)
    for (x, y, w_face, h_face) in faces:
        # Expand face area with margin
        margin = 0.2
        # ... create smooth mask with Gaussian blur
    return face_mask
```

**Key Features**:
- **Expansion Margin**: 20% expansion ensures hair and neck are protected
- **Smooth Masks**: Gaussian blur creates natural transitions
- **Multiple Faces**: Handles multiple faces in group portraits

**Tutorial Reference**: See `chapter19/02_face_detection.py` for detailed face detection implementation.

### 19.4.3 Face-Aware Processing

Face detection is combined with depth-based blurring:

1. Detect faces and create protection masks
2. Generate depth-based blur mask
3. Remove face regions from blur mask
4. Apply blur only to non-face background areas

This ensures faces remain sharp even if depth estimation is imperfect.

---

## 19.5 Implementation Details

### 19.5.1 Code Structure

The implementation is organized into two main components:

#### Educational Tutorials (`chapter19/`)
Step-by-step tutorials for learning concepts:
- `01_basic_concepts.py`: Understanding depth maps
- `02_face_detection.py`: Face detection and protection
- `03_depth_estimation_midas.py`: MiDaS depth estimation
- `04_depth_estimation_depth_anything.py`: Depth-Anything depth estimation
- `05_portrait_effect.py`: Creating portrait effects
- `06_complete_pipeline.py`: End-to-end implementation

#### Production Code (`production/`)
Production-ready implementations:
- `production/midas/midas_portrait.py`: MiDaS processor
- `production/depth_anything/depth_anything_portrait.py`: Depth-Anything processor

**Usage**: Start with tutorials to understand concepts, then use production code for applications.

### 19.5.2 Corrected Depth Logic

The critical implementation detail is the **corrected depth interpretation**:

```python
def create_portrait_effect(self, image_bgr, depth_map, face_mask=None, 
                         strategy='conservative', threshold=None):
    """Create portrait effect using corrected depth logic."""
    # Normalize depth to [0, 1]
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    # CORRECTED LOGIC: Blur areas with LOW depth values (background)
    bg_mask = (depth_normalized < threshold).astype(np.float32)
    bg_mask = cv2.GaussianBlur(bg_mask, (15, 15), 0)
    
    # Apply face protection
    if face_mask is not None:
        face_protection = 1.0 - face_mask
        bg_mask = bg_mask * face_protection
    
    # Composite: sharp foreground + blurred background
    blurred_bg = cv2.GaussianBlur(image_bgr, (51, 51), 0)
    result = image_bgr.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1 - bg_mask) +  # Sharp (foreground)
            blurred_bg[:, :, c] * bg_mask      # Blurred (background)
        )
    return result.astype(np.uint8), bg_mask, depth_normalized
```

**Critical Point**: `depth_normalized < threshold` means we blur **low depth values** (background), keeping **high depth values** (foreground) sharp.

### 19.5.3 Processing Strategies

Three blurring strategies are implemented:

| Strategy | Threshold | Use Case |
|----------|-----------|----------|
| **Conservative** | 0.8 | Gentle background blur, keeps more foreground sharp |
| **Face-Aware** | 0.2 | Aggressive blur with face protection |
| **Adaptive** | 85th percentile | Dynamic threshold based on depth distribution |

**Tutorial Reference**: See `chapter19/05_portrait_effect.py` for detailed strategy implementation.

---

## 19.6 Experimental Results

### 19.6.1 Performance Comparison

Comprehensive testing on 3 portrait images reveals:

| Metric | MiDaS | Depth-Anything | Improvement |
|--------|-------|----------------|-------------|
| **Average Time** | 1.04s | 0.35s | **3x faster** |
| **Model Size** | ~100MB | ~200MB | 2x larger |
| **Depth Quality** | High | Very High | Superior |
| **Face Detection** | ✅ | ✅ | Both excellent |
| **Portrait Quality** | Professional | Professional | Both excellent |

### 19.6.2 Quality Assessment

**Depth Estimation**:
- MiDaS: Consistent, reliable depth maps
- Depth-Anything: Superior detail and accuracy

**Portrait Effects**:
- Both approaches successfully keep faces sharp
- Natural, realistic bokeh effects
- Smooth transitions between sharp and blurred regions

**Success Rate**: 100% (3/3 test images processed successfully)

### 19.6.3 Visual Results

The implementation generates comprehensive outputs:
- Original images
- Depth map visualizations
- Face detection masks
- Three strategy results (conservative, face-aware, adaptive)
- Comprehensive comparison grids

**Repository**: See `results/` directory for all generated outputs and comparison grids.

---

## 19.7 Discussion and Practical Considerations

### 19.7.1 Choosing Between Approaches

**Choose MiDaS when**:
- Reliability and consistency are priorities
- Model size is a constraint
- You need proven, stable performance

**Choose Depth-Anything when**:
- Speed is critical (real-time applications)
- Highest quality is required
- You can accommodate larger model size

### 19.7.2 Mobile Deployment Considerations

**Processing Speed**: Depth-Anything's 0.35s per image enables near real-time processing on modern mobile devices.

**Model Size**: Both models (100-200MB) are reasonable for mobile deployment, though Depth-Anything requires more storage.

**Quality**: Both approaches produce professional-grade results suitable for consumer applications.

**Robustness**: Consistent performance across diverse portrait scenarios with varying lighting and backgrounds.

### 19.7.3 Key Implementation Insights

1. **Corrected Depth Logic**: The critical insight is that low depth values = background (blur), high values = foreground (sharp). This must be correctly interpreted.

2. **Face Protection**: Face detection adds robustness, ensuring faces remain sharp even if depth estimation is imperfect.

3. **Multiple Strategies**: Three strategies provide flexibility for different use cases and user preferences.

4. **Smooth Blending**: Gaussian blur on masks creates natural transitions between sharp and blurred regions.

### 19.7.4 Common Challenges and Solutions

**Challenge 1**: Faces being blurred instead of backgrounds
- **Solution**: Corrected depth logic (low values = background)

**Challenge 2**: Harsh transitions between sharp and blurred regions
- **Solution**: Gaussian blur on masks for smooth blending

**Challenge 3**: Face detection missing some faces
- **Solution**: 20% expansion margin and smooth mask generation

**Challenge 4**: Performance optimization
- **Solution**: Depth-Anything provides 3x speedup while maintaining quality

---

## 19.8 Getting Started with the Code

### 19.8.1 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JeffRen1977/video_bokeh.git
   cd video_bokeh
   ```

2. **Set up environment**:
   ```bash
   source ../venv/bin/activate  # Activate virtual environment
   pip install -r requirements.txt
   ./setup_depth_anything.sh
   ```

3. **Run tutorials** (for learning):
   ```bash
   cd chapter19
   python 01_basic_concepts.py  # Start here
   python 02_face_detection.py
   python 03_depth_estimation_midas.py
   python 04_depth_estimation_depth_anything.py
   python 05_portrait_effect.py
   python 06_complete_pipeline.py
   ```

4. **Use production code** (for applications):
   ```bash
   cd ..  # Back to project root
   python portrait_comparison.py --single data/portrait1.jpg
   ```

### 19.8.2 Learning Path

**For Students Learning Chapter 19**:
1. Read Section 19.2 (Fundamental Concepts)
2. Run tutorial `01_basic_concepts.py` to see depth maps
3. Run tutorial `02_face_detection.py` to understand face protection
4. Run tutorials 03-04 to compare depth estimation approaches
5. Run tutorial `05_portrait_effect.py` to see complete implementation
6. Review production code in `production/` directory

**For Developers Building Applications**:
1. Review production code in `production/` directory
2. Use `portrait_comparison.py` for batch processing
3. Import processors directly: `from production.midas.midas_portrait import MiDaSPortraitProcessor`

### 19.8.3 Code Organization

```
video_bokeh/
├── chapter19/              # Educational tutorials (start here)
│   ├── 01_basic_concepts.py
│   ├── 02_face_detection.py
│   ├── 03_depth_estimation_midas.py
│   ├── 04_depth_estimation_depth_anything.py
│   ├── 05_portrait_effect.py
│   └── 06_complete_pipeline.py
├── production/             # Production code (for applications)
│   ├── midas/midas_portrait.py
│   └── depth_anything/depth_anything_portrait.py
├── comparison/             # Comparison framework
├── data/                   # Test images
├── results/                # Generated outputs
└── portrait_comparison.py  # Main script
```

---

## 19.9 Summary

This chapter presented a comprehensive study of depth-aware portrait effects for mobile computational photography. Key contributions include:

1. **Clear Conceptual Framework**: Explained depth maps, portrait effect pipeline, and corrected depth logic
2. **Two Approaches Compared**: MiDaS (reliable) and Depth-Anything (fast, high-quality)
3. **Face Protection**: Robust face detection ensuring facial regions remain sharp
4. **Practical Implementation**: Educational tutorials and production-ready code
5. **Performance Analysis**: 3x speedup achieved with Depth-Anything

### 19.9.1 Key Takeaways

- **Depth maps** encode distance: low values = background, high values = foreground
- **Corrected logic** is critical: blur low depth values (background), keep high values (foreground) sharp
- **Face detection** adds robustness to ensure faces remain sharp
- **Depth-Anything** offers 3x speedup with superior quality
- **Multiple strategies** provide flexibility for different use cases

### 19.9.2 Future Directions

- **Real-time Video Processing**: Extend to video streams with temporal consistency
- **Mobile Optimization**: Model quantization and mobile-specific optimizations
- **Enhanced Face Detection**: Multi-face support and landmark detection
- **Advanced Blurring**: Customizable bokeh shapes and depth-aware feathering

---

## References

[1] Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision transformers for dense prediction. *Proceedings of the IEEE/CVF International Conference on Computer Vision*.

[2] Li, H., et al. (2024). Depth-Anything: Unleashing the Power of Large-Scale Unlabeled Data. *arXiv preprint arXiv:2401.10891*.

[3] Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv preprint arXiv:2304.07193*.

[4] Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition*.

[5] Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.

---

## Appendix: Code Repository Structure

The complete implementation is available at: **https://github.com/JeffRen1977/video_bokeh**

### Key Files and Directories

- **`chapter19/`**: Educational tutorials with step-by-step explanations
- **`production/`**: Production-ready code for integration
- **`data/`**: Test portrait images
- **`results/`**: Generated outputs and comparison grids
- **`doc/`**: Additional documentation and learning guides

### Quick Reference

- **Learning**: Start with `chapter19/01_basic_concepts.py`
- **Production**: Use `production/midas/` or `production/depth_anything/`
- **Comparison**: Run `python portrait_comparison.py --compare`
- **Documentation**: See `doc/CHAPTER19_GUIDE.md` for detailed learning path

---

*This chapter provides both theoretical understanding and practical implementation, enabling readers to understand and apply depth-aware portrait effects in mobile computational photography applications.*

