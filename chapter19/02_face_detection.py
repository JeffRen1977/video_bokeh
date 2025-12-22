#!/usr/bin/env python3
"""
Chapter 19 Tutorial 2: Face Detection for Portrait Effects

This tutorial demonstrates face detection and protection techniques from Chapter 19:
- OpenCV face detection using Haar cascades
- Creating face protection masks
- Face-aware portrait processing

Reference: Chapter 19, Section 19.3 - Face Detection and Protection
"""

import cv2
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)


def detect_faces(image_bgr):
    """
    Detect faces in an image using OpenCV Haar cascades.
    
    This demonstrates Concept 3 from Chapter 19: Face Detection
    - Uses pre-trained Haar cascade classifiers
    - Detects frontal faces in grayscale images
    - Returns bounding boxes for detected faces
    
    Reference: Chapter 19, Section 19.3.1 - Face Detection Methods
    
    Args:
        image_bgr: Input BGR image
        
    Returns:
        faces: List of (x, y, width, height) tuples for detected faces
    """
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        print("❌ Failed to load face cascade classifier")
        return []
    
    # Convert to grayscale (required for Haar cascades)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    # Parameters:
    #   scaleFactor: How much the image size is reduced at each scale (1.1 = 10% reduction)
    #   minNeighbors: Minimum neighbors required for a detection (4 = more strict)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    return faces


def create_face_mask(image_shape, faces, margin=0.2):
    """
    Create a face protection mask from detected faces.
    
    This demonstrates Concept 4 from Chapter 19: Face Protection Masks
    - Expands face regions with margin for better protection
    - Creates smooth masks using Gaussian blur
    - Ensures faces remain sharp in portrait effects
    
    Reference: Chapter 19, Section 19.3.2 - Face Protection Strategies
    
    Args:
        image_shape: (height, width) of the image
        faces: List of (x, y, width, height) tuples for detected faces
        margin: Expansion margin around faces (0.2 = 20% expansion)
        
    Returns:
        face_mask: 2D array (1.0 where faces are, 0.0 elsewhere)
    """
    h, w = image_shape[:2]
    face_mask = np.zeros((h, w), dtype=np.float32)
    
    if len(faces) == 0:
        return face_mask
    
    for (x, y, w_face, h_face) in faces:
        # Expand face area with margin
        # This ensures we protect not just the face but also hair, neck, etc.
        x_expand = int(x - w_face * margin)
        y_expand = int(y - h_face * margin)
        w_expand = int(w_face * (1 + 2 * margin))
        h_expand = int(h_face * (1 + 2 * margin))
        
        # Ensure coordinates are within image bounds
        x_expand = max(0, x_expand)
        y_expand = max(0, y_expand)
        w_expand = min(w - x_expand, w_expand)
        h_expand = min(h - y_expand, h_expand)
        
        # Create smooth mask for face area
        # Using Gaussian blur creates smooth transitions at edges
        face_region = np.ones((h_expand, w_expand), dtype=np.float32)
        face_region = cv2.GaussianBlur(face_region, (21, 21), 0)
        
        face_mask[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand] = face_region
    
    return face_mask


def visualize_face_detection(image_bgr, faces):
    """
    Visualize detected faces by drawing bounding boxes.
    
    Args:
        image_bgr: Input BGR image
        faces: List of (x, y, width, height) tuples for detected faces
        
    Returns:
        visualization: Image with bounding boxes drawn
    """
    visualization = image_bgr.copy()
    
    for (x, y, w, h) in faces:
        # Draw green rectangle around detected face
        cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add label
        cv2.putText(visualization, 'Face', (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return visualization


def apply_face_aware_blur(image_bgr, depth_map, face_mask, threshold=0.5):
    """
    Apply blur with face protection.
    
    This demonstrates the complete face-aware processing from Chapter 19:
    - Combines depth-based blurring with face protection
    - Ensures faces remain sharp even if depth estimation is imperfect
    - Creates professional portrait effects
    
    Reference: Chapter 19, Section 19.3.3 - Face-Aware Portrait Processing
    
    Args:
        image_bgr: Input BGR image
        depth_map: Depth map
        face_mask: Face protection mask (1.0 = face, 0.0 = background)
        threshold: Depth threshold for blurring
        
    Returns:
        result: Portrait effect with face protection
    """
    # Normalize depth
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    # Create blur mask based on depth (low depth = background = blur)
    blur_mask = (depth_normalized < threshold).astype(np.float32)
    blur_mask = cv2.GaussianBlur(blur_mask, (15, 15), 0)
    
    # Apply face protection: remove blur from face regions
    # face_mask is 1.0 where faces are, so we subtract it from blur_mask
    if face_mask is not None and face_mask.max() > 0:
        face_protection = 1.0 - face_mask  # 0.0 where faces are, 1.0 elsewhere
        blur_mask = blur_mask * face_protection  # Remove blur from face areas
    
    # Apply blur
    blurred = cv2.GaussianBlur(image_bgr, (51, 51), 0)
    
    # Composite
    result = image_bgr.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1 - blur_mask) +  # Sharp areas
            blurred[:, :, c] * blur_mask          # Blurred areas
        )
    
    return result.astype(np.uint8), blur_mask


def main():
    """
    Main tutorial function demonstrating face detection from Chapter 19.
    """
    print("=" * 70)
    print("Chapter 19 Tutorial 2: Face Detection for Portrait Effects")
    print("=" * 70)
    print("\nThis tutorial demonstrates:")
    print("1. Face detection using OpenCV Haar cascades")
    print("2. Creating face protection masks")
    print("3. Face-aware portrait processing")
    print("\nReference: Chapter 19, Section 19.3\n")
    
    # Load a test image
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
    max_size = 600
    if max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"📏 Resized to: {new_h}x{new_w}")
    
    print("\n" + "=" * 70)
    print("Step 1: Face Detection")
    print("=" * 70)
    print("Detecting faces using OpenCV Haar cascade classifier...")
    
    faces = detect_faces(image)
    print(f"✅ Detected {len(faces)} face(s)")
    
    if len(faces) == 0:
        print("⚠️  No faces detected. This might affect portrait effect quality.")
    else:
        for i, (x, y, w, h) in enumerate(faces, 1):
            print(f"   Face {i}: position=({x}, {y}), size={w}x{h}")
    
    # Visualize face detection
    face_detection_viz = visualize_face_detection(image, faces)
    detection_path = output_dir / "02_face_detection.jpg"
    cv2.imwrite(str(detection_path), face_detection_viz)
    print(f"✅ Face detection visualization saved: {detection_path}")
    
    print("\n" + "=" * 70)
    print("Step 2: Creating Face Protection Mask")
    print("=" * 70)
    print("Creating smooth mask to protect face regions from blurring...")
    
    face_mask = create_face_mask(image.shape, faces, margin=0.2)
    
    if face_mask.max() > 0:
        mask_visualization = (face_mask * 255).astype(np.uint8)
        mask_visualization = cv2.cvtColor(mask_visualization, cv2.COLOR_GRAY2BGR)
        mask_path = output_dir / "02_face_mask.jpg"
        cv2.imwrite(str(mask_path), mask_visualization)
        print(f"✅ Face mask saved: {mask_path}")
        print(f"   Mask coverage: {face_mask.sum() / face_mask.size * 100:.1f}% of image")
    else:
        print("⚠️  No face mask created (no faces detected)")
    
    print("\n" + "=" * 70)
    print("Step 3: Face-Aware Portrait Processing")
    print("=" * 70)
    print("Applying portrait effect with face protection...")
    print("(Using simple depth map for demonstration)")
    
    # Create simple depth map (from Tutorial 1)
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    depth_map = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            normalized_dist = dist / max_dist
            depth_map[y, x] = 1.0 - normalized_dist * 0.7
    
    # Apply face-aware blur
    result, blur_mask = apply_face_aware_blur(image, depth_map, face_mask, threshold=0.5)
    
    result_path = output_dir / "02_face_aware_portrait.jpg"
    cv2.imwrite(str(result_path), result)
    print(f"✅ Face-aware portrait effect saved: {result_path}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ You've learned:")
    print("  1. How to detect faces using OpenCV Haar cascades")
    print("  2. How to create face protection masks")
    print("  3. How to combine face detection with depth-based blurring")
    print("\n📖 Next: Tutorial 3 - Depth Estimation with MiDaS (03_depth_estimation_midas.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()


