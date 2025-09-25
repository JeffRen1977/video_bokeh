#!/usr/bin/env python3
"""
MiDaS Portrait Effect Module

This module provides depth-aware portrait effects using the MiDaS model.
It implements the corrected depth interpretation logic to ensure proper face sharpening.

Key Features:
- MiDaS depth estimation
- Corrected depth interpretation (low values = background, high values = face)
- Face detection for additional protection
- Multiple threshold strategies
- Professional portrait quality
"""

import cv2
import numpy as np
import torch
import time
import os
from pathlib import Path


class MiDaSPortraitProcessor:
    """MiDaS-based portrait effect processor with corrected logic."""
    
    def __init__(self, device='cpu'):
        """
        Initialize MiDaS portrait processor.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None
        self.transform = None
        self.face_cascade = None
        
        print("📥 Initializing MiDaS Portrait Processor...")
        self._load_model()
        self._load_face_detector()
        print("✅ MiDaS Portrait Processor initialized!")
    
    def _load_model(self):
        """Load MiDaS model and transform."""
        print("  📥 Loading MiDaS model...")
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.model.eval()
        
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.transform.dpt_transform
        
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.to(self.device)
            print("  🚀 Model loaded on GPU")
        else:
            print("  💻 Model loaded on CPU")
    
    def _load_face_detector(self):
        """Load OpenCV face detection cascade."""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("  👤 Face detector loaded")
        except Exception as e:
            print(f"  ⚠️  Face detector not available: {e}")
            self.face_cascade = None
    
    def _detect_faces(self, image_bgr):
        """
        Detect faces in the image.
        
        Args:
            image_bgr: Input BGR image
            
        Returns:
            Face mask (1.0 where faces are detected, 0.0 elsewhere)
        """
        if self.face_cascade is None:
            return None
        
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        h, w = image_bgr.shape[:2]
        face_mask = np.zeros((h, w), dtype=np.float32)
        
        for (x, y, w_face, h_face) in faces:
            # Expand face area slightly
            margin = 0.2
            x_expand = int(x - w_face * margin)
            y_expand = int(y - h_face * margin)
            w_expand = int(w_face * (1 + 2 * margin))
            h_expand = int(h_face * (1 + 2 * margin))
            
            # Ensure coordinates are within bounds
            x_expand = max(0, x_expand)
            y_expand = max(0, y_expand)
            w_expand = min(w - x_expand, w_expand)
            h_expand = min(h - y_expand, h_expand)
            
            # Create smooth mask for face area
            face_region = np.ones((h_expand, w_expand), dtype=np.float32)
            face_region = cv2.GaussianBlur(face_region, (21, 21), 0)
            face_mask[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand] = face_region
        
        return face_mask if len(faces) > 0 else None
    
    def estimate_depth(self, image_bgr):
        """
        Estimate depth map from input image using MiDaS.
        
        Args:
            image_bgr: Input BGR image
            
        Returns:
            Depth map as numpy array
        """
        # Convert to RGB for MiDaS
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(img_rgb)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        
        return prediction
    
    def create_portrait_effect(self, image_bgr, depth_map, face_mask=None, 
                             strategy='conservative', threshold=None):
        """
        Create portrait effect using corrected depth logic.
        
        Args:
            image_bgr: Input BGR image
            depth_map: Depth map from MiDaS
            face_mask: Optional face detection mask
            strategy: Strategy to use ('conservative', 'face_aware', 'adaptive')
            threshold: Custom threshold (if None, uses strategy defaults)
            
        Returns:
            Portrait effect result
        """
        # Normalize depth
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Determine threshold based on strategy
        if threshold is None:
            if strategy == 'conservative':
                threshold = 0.8  # Blur only very distant areas
            elif strategy == 'face_aware':
                threshold = 0.2  # More aggressive for face detection
            elif strategy == 'adaptive':
                threshold = np.percentile(depth_normalized, 85)  # Blur top 15%
            else:
                threshold = 0.3  # Default
        
        # CORRECTED LOGIC: Blur areas with LOW depth values (background)
        bg_mask = (depth_normalized < threshold).astype(np.float32)
        bg_mask = cv2.GaussianBlur(bg_mask, (15, 15), 0)
        
        # Apply face protection if available
        if face_mask is not None and strategy == 'face_aware':
            face_protection = 1.0 - face_mask
            bg_mask = bg_mask * face_protection
        
        # Apply blur to background areas
        blurred_bg = cv2.GaussianBlur(image_bgr, (51, 51), 0)
        
        # Composite: keep foreground sharp, blur background
        result = image_bgr.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = (
                result[:, :, c] * (1 - bg_mask) +  # Keep sharp (foreground)
                blurred_bg[:, :, c] * bg_mask      # Blur (background)
            )
        
        return result.astype(np.uint8), bg_mask, depth_normalized
    
    def process_image(self, image_path, output_dir="results/midas", 
                     strategies=['conservative', 'face_aware', 'adaptive']):
        """
        Process a single image with multiple strategies.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
            strategies: List of strategies to apply
            
        Returns:
            Dictionary with results for each strategy
        """
        print(f"🎭 Processing with MiDaS: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize if too large
        max_size = 800
        if max(img.shape[:2]) > max_size:
            scale = max_size / max(img.shape[:2])
            new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"📏 Resized to: {new_h}x{new_w}")
        
        # Estimate depth
        print("⚡ Running MiDaS depth estimation...")
        start_time = time.time()
        depth_map = self.estimate_depth(img)
        inference_time = time.time() - start_time
        print(f"✅ Depth estimation completed in {inference_time:.2f} seconds")
        
        # Detect faces
        face_mask = self._detect_faces(img)
        if face_mask is not None:
            print(f"👤 Detected faces for protection")
        
        # Process with different strategies
        results = {}
        img_name = Path(image_path).stem
        
        os.makedirs(output_dir, exist_ok=True)
        
        for strategy in strategies:
            print(f"🎯 Applying {strategy} strategy...")
            
            portrait_result, bg_mask, depth_normalized = self.create_portrait_effect(
                img, depth_map, face_mask, strategy
            )
            
            # Save results
            base_name = os.path.join(output_dir, f"{img_name}_midas_{strategy}")
            cv2.imwrite(f"{base_name}.jpg", portrait_result)
            
            # Create visualizations
            depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            mask_colored = cv2.applyColorMap((bg_mask * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            
            cv2.imwrite(f"{base_name}_depth.jpg", depth_colored)
            cv2.imwrite(f"{base_name}_mask.jpg", mask_colored)
            
            results[strategy] = {
                'result': portrait_result,
                'depth_map': depth_colored,
                'mask': mask_colored,
                'processing_time': inference_time
            }
        
        # Save original and face mask
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_midas_original.jpg"), img)
        if face_mask is not None:
            face_mask_viz = (face_mask * 255).astype(np.uint8)
            face_mask_viz = cv2.cvtColor(face_mask_viz, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{img_name}_midas_face_mask.jpg"), face_mask_viz)
        
        print(f"💾 MiDaS results saved to {output_dir}/")
        return results


def main():
    """Main function for testing MiDaS approach."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python midas_portrait.py path/to/image.jpg")
        return
    
    processor = MiDaSPortraitProcessor()
    results = processor.process_image(sys.argv[1])
    
    print("✅ MiDaS processing completed!")
    for strategy, result in results.items():
        print(f"  - {strategy}: {result['processing_time']:.2f}s")


if __name__ == "__main__":
    main()
