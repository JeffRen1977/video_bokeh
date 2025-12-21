#!/usr/bin/env python3
"""
Chapter 19 Tutorial 4: Depth Estimation with Depth-Anything

This tutorial demonstrates depth estimation using the Depth-Anything model:
- Loading and using Depth-Anything model
- Understanding DINOv2 backbone architecture
- Depth map interpretation and comparison with MiDaS
- Performance advantages

Reference: Chapter 19, Section 19.2.2 - Depth-Anything Approach
"""

import cv2
import numpy as np
import torch
import time
import sys
from pathlib import Path

# Add parent directory to path for Depth-Anything imports
sys.path.append(str(Path(__file__).parent.parent))

# Create output directory
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)


def load_depth_anything_model(device='cpu', encoder='vits'):
    """
    Load Depth-Anything depth estimation model.
    
    This demonstrates Concept 1 from Chapter 19: Depth-Anything Model Architecture
    - Uses DPT-DINOv2 with Vision Transformer backbone
    - Pre-trained on large-scale datasets
    - Foundation model approach
    - 3x faster than MiDaS
    
    Reference: Chapter 19, Section 19.2.2 - Depth-Anything Approach
    
    Args:
        device: 'cpu' or 'cuda'
        encoder: 'vits' (small), 'vitb' (base), or 'vitl' (large)
        
    Returns:
        model: Loaded Depth-Anything model
        transform: Preprocessing transform
    """
    print(f"📥 Loading Depth-Anything model (ViT-{encoder})...")
    print("   This may take a moment on first run (downloading model)")
    
    try:
        # Add Depth-Anything repository to path
        depth_anything_repo = Path(__file__).parent.parent / 'Depth-Anything'
        if depth_anything_repo.exists():
            sys.path.insert(0, str(depth_anything_repo))
        else:
            print("   ⚠️  Depth-Anything repository not found")
            print("   Run: ./setup_depth_anything.sh")
            return None, None
        
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose
        from depth_anything.dpt import DepthAnything
        
        # Load model
        model_name = f'LiheYoung/depth_anything_{encoder}14'
        print(f"   📥 Loading model: {model_name}")
        model = DepthAnything.from_pretrained(model_name).eval()
        
        # Create transform
        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            model = model.to(device)
            print("   ✅ Model loaded on GPU")
        else:
            print("   ✅ Model loaded on CPU")
        
        return model, transform
        
    except Exception as e:
        print(f"   ❌ Failed to load Depth-Anything model: {e}")
        print("   Make sure Depth-Anything repository is set up:")
        print("   1. Run: ./setup_depth_anything.sh")
        print("   2. Or manually clone: git clone https://github.com/LiheYoung/Depth-Anything.git")
        return None, None


def preprocess_image_depth_anything(image_bgr, transform):
    """
    Preprocess image for Depth-Anything model.
    
    Depth-Anything expects:
    - RGB format (normalized to [0, 1])
    - Specific input size (518x518, handled by transform)
    - Normalized with ImageNet statistics
    
    Args:
        image_bgr: Input BGR image
        transform: Depth-Anything transform pipeline
        
    Returns:
        input_tensor: Preprocessed tensor ready for model
    """
    # Convert BGR to RGB and normalize to [0, 1]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0
    
    # Apply transform (resizes, normalizes, prepares for network)
    input_tensor = transform({'image': image_rgb})['image']
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
    
    return input_tensor


def estimate_depth_depth_anything(image_bgr, model, transform, device='cpu'):
    """
    Estimate depth map using Depth-Anything model.
    
    This demonstrates the complete depth estimation pipeline:
    1. Preprocess image
    2. Run model inference
    3. Post-process depth map
    4. Resize to original image size
    
    Reference: Chapter 19, Section 19.2.2 - Depth-Anything Depth Estimation
    
    Args:
        image_bgr: Input BGR image
        model: Depth-Anything model
        transform: Preprocessing transform
        device: 'cpu' or 'cuda'
        
    Returns:
        depth_map: Estimated depth map (numpy array)
        inference_time: Time taken for inference
    """
    print("⚡ Running Depth-Anything depth estimation...")
    
    # Preprocess
    input_tensor = preprocess_image_depth_anything(image_bgr, transform)
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        input_tensor = input_tensor.to(device)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        depth = model(input_tensor)
        depth_np = depth.squeeze().cpu().numpy()
    
    # Resize to original image size
    h, w = image_bgr.shape[:2]
    depth_resized = cv2.resize(depth_np, (w, h))
    
    inference_time = time.time() - start_time
    
    print(f"   ✅ Depth estimation completed in {inference_time:.2f} seconds")
    print(f"   📊 Depth range: [{depth_resized.min():.3f}, {depth_resized.max():.3f}]")
    
    return depth_resized, inference_time


def visualize_depth_anything_depth(depth_map, original_image):
    """
    Visualize Depth-Anything depth map with original image.
    
    Args:
        depth_map: Depth-Anything depth map
        original_image: Original BGR image
        
    Returns:
        visualization: Combined visualization
    """
    # Normalize depth for visualization
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply INFERNO colormap (as used in Chapter 19)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    
    # Resize images to same size for side-by-side comparison
    h, w = original_image.shape[:2]
    max_h = 400
    if h > max_h:
        scale = max_h / h
        new_h, new_w = int(h * scale), int(w * scale)
        original_resized = cv2.resize(original_image, (new_w, new_h))
        depth_resized = cv2.resize(depth_colored, (new_w, new_h))
    else:
        original_resized = original_image
        depth_resized = depth_colored
    
    # Combine side by side
    combined = np.hstack([original_resized, depth_resized])
    
    # Add labels
    cv2.putText(combined, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Depth-Anything Depth Map", (original_resized.shape[1] + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined, depth_colored


def analyze_depth_characteristics(depth_map):
    """
    Analyze characteristics of Depth-Anything depth map.
    
    This helps understand what Depth-Anything depth maps represent:
    - Metric depth (more accurate than MiDaS)
    - Higher values = closer to camera
    - Lower values = farther from camera
    - Better detail preservation
    
    Reference: Chapter 19, Section 19.2.2 - Depth Map Interpretation
    
    Args:
        depth_map: Depth-Anything depth map
        
    Returns:
        analysis: Dictionary with statistics
    """
    analysis = {
        'min': depth_map.min(),
        'max': depth_map.max(),
        'mean': depth_map.mean(),
        'std': depth_map.std(),
        'percentiles': {
            '25th': np.percentile(depth_map, 25),
            '50th': np.percentile(depth_map, 50),
            '75th': np.percentile(depth_map, 75),
            '85th': np.percentile(depth_map, 85),
        }
    }
    
    return analysis


def main():
    """
    Main tutorial function demonstrating Depth-Anything depth estimation from Chapter 19.
    """
    print("=" * 70)
    print("Chapter 19 Tutorial 4: Depth Estimation with Depth-Anything")
    print("=" * 70)
    print("\nThis tutorial demonstrates:")
    print("1. Loading Depth-Anything DPT-DINOv2 model")
    print("2. Preprocessing images for Depth-Anything")
    print("3. Running depth estimation inference")
    print("4. Understanding Depth-Anything depth maps")
    print("5. Performance comparison with MiDaS")
    print("\nReference: Chapter 19, Section 19.2.2\n")
    
    # Load test image
    image_path = Path(__file__).parent.parent / "data" / "portrait1.jpg"
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📷 Loading image: {image_path.name}")
    image = cv2.imread(str(image_path))
    if image is None:
        print("❌ Failed to load image")
        return
    
    # Resize if too large
    max_size = 800
    if max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"📏 Resized to: {new_h}x{new_w}")
    
    print("\n" + "=" * 70)
    print("Step 1: Loading Depth-Anything Model")
    print("=" * 70)
    print("Depth-Anything:")
    print("- Architecture: DPT-DINOv2 with ViT-Small backbone")
    print("- Backbone: Vision Transformer with DINOv2 features")
    print("- Training: Large-scale foundation model")
    print("- Output: Metric depth maps")
    print("- Typical speed: ~0.35s per image (CPU) - 3x faster than MiDaS")
    
    model, transform = load_depth_anything_model(device='cpu', encoder='vits')
    if model is None:
        return
    
    print("\n" + "=" * 70)
    print("Step 2: Depth Estimation")
    print("=" * 70)
    print("Running Depth-Anything inference on the image...")
    
    depth_map, inference_time = estimate_depth_depth_anything(image, model, transform, device='cpu')
    
    print("\n" + "=" * 70)
    print("Step 3: Depth Map Analysis")
    print("=" * 70)
    
    analysis = analyze_depth_characteristics(depth_map)
    print(f"Depth Statistics:")
    print(f"  Min: {analysis['min']:.3f}")
    print(f"  Max: {analysis['max']:.3f}")
    print(f"  Mean: {analysis['mean']:.3f}")
    print(f"  Std: {analysis['std']:.3f}")
    print(f"\nPercentiles (useful for threshold selection):")
    print(f"  25th: {analysis['percentiles']['25th']:.3f}")
    print(f"  50th: {analysis['percentiles']['50th']:.3f}")
    print(f"  75th: {analysis['percentiles']['75th']:.3f}")
    print(f"  85th: {analysis['percentiles']['85th']:.3f} (used in adaptive strategy)")
    
    print("\n" + "=" * 70)
    print("Step 4: Visualization")
    print("=" * 70)
    print("Creating depth map visualization...")
    
    combined_viz, depth_colored = visualize_depth_anything_depth(depth_map, image)
    
    # Save results
    combined_path = output_dir / "04_depth_anything_depth_comparison.jpg"
    depth_path = output_dir / "04_depth_anything_depth_map.jpg"
    
    cv2.imwrite(str(combined_path), combined_viz)
    cv2.imwrite(str(depth_path), depth_colored)
    
    print(f"✅ Comparison saved: {combined_path}")
    print(f"✅ Depth map saved: {depth_path}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ You've learned:")
    print("  1. How to load and use Depth-Anything DPT-DINOv2 model")
    print("  2. How Depth-Anything preprocesses images")
    print("  3. How to interpret Depth-Anything depth maps")
    print("  4. Performance advantages over MiDaS")
    print(f"\n📊 Performance: {inference_time:.2f}s per image")
    print("   (Typically 3x faster than MiDaS)")
    print("\n📖 Key Points from Chapter 19:")
    print("  - Depth-Anything uses metric depth (more accurate)")
    print("  - Higher values = closer to camera")
    print("  - Lower values = farther from camera")
    print("  - Superior detail preservation")
    print("  - Foundation model approach")
    print("\n📖 Next: Tutorial 5 - Portrait Effect Creation (05_portrait_effect.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()

