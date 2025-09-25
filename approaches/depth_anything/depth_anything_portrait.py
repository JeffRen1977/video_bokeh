#!/usr/bin/env python3
"""
Depth-Anything Portrait Effect Module

This module provides depth-aware portrait effects using the Depth-Anything model.
It implements the corrected depth interpretation logic to ensure proper face sharpening.

Key Features:
- Depth-Anything depth estimation
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


class DepthAnythingPortraitProcessor:
    """Depth-Anything-based portrait effect processor with corrected logic."""
    
    def __init__(self, device='cpu', encoder='vits'):
        """
        Initialize Depth-Anything portrait processor.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            encoder: Encoder size ('vits', 'vitb', 'vitl')
        """
        self.device = device
        self.encoder = encoder
        self.model = None
        self.transform = None
        self.face_cascade = None
        
        print("📥 Initializing Depth-Anything Portrait Processor...")
        self._load_model()
        self._load_face_detector()
        print("✅ Depth-Anything Portrait Processor initialized!")
    
    def _load_model(self):
        """Load Depth-Anything model and transform."""
        try:
            # Add the Depth-Anything repository to the path
            import sys
            from pathlib import Path
            depth_anything_repo = Path(__file__).parent.parent.parent / 'Depth-Anything'
            if depth_anything_repo.exists():
                sys.path.insert(0, str(depth_anything_repo))
            
            from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
            from torchvision.transforms import Compose
            import torch
            
            print("  📥 Loading Depth-Anything model...")
            
            # Try to use the original DepthAnything.from_pretrained method
            from depth_anything.dpt import DepthAnything
            
            model_name = f'LiheYoung/depth_anything_{self.encoder}14'
            print(f"  📥 Loading model: {model_name}")
            
            # This should work with the cached DINOv2
            self.model = DepthAnything.from_pretrained(model_name).eval()
            
            # Create transform
            self.transform = Compose([
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
            
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to(self.device)
                print("  🚀 Model loaded on GPU")
            else:
                print("  💻 Model loaded on CPU")
                
        except Exception as e:
            print(f"  ❌ Depth-Anything model loading failed: {e}")
            print("  🔄 Using enhanced simulation approach as fallback")
            self.model = None
            self.transform = None
    
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
    
    def _create_simulated_depth(self, image_shape, face_center=None):
        """
        Create enhanced simulated depth map that mimics Depth-Anything behavior.
        
        Args:
            image_shape: (height, width, channels) of the input image
            face_center: (x, y) center of face region
            
        Returns:
            Simulated depth map
        """
        h, w = image_shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        if face_center is None:
            face_center = (w // 2, h // 2)
        
        # Create a more realistic depth distribution
        # Background: low depth values (far from camera)
        depth_map[:, :] = 0.1
        
        # Create depth gradient from center to edges (common in portrait photography)
        center_y, center_x = h // 2, w // 2
        for y in range(h):
            for x in range(w):
                # Distance from center
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                normalized_center_dist = dist_from_center / max_dist
                
                # Base depth value (closer to center = closer to camera)
                base_depth = 0.8 - 0.4 * normalized_center_dist
                depth_map[y, x] = max(base_depth, 0.1)
        
        # Face region: high depth values (close to camera)
        face_size = min(w, h) // 3  # Larger face region
        y_start = max(0, face_center[1] - face_size)
        y_end = min(h, face_center[1] + face_size)
        x_start = max(0, face_center[0] - face_size)
        x_end = min(w, face_center[0] + face_size)
        
        # Create smooth face region with high depth values
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                dist = np.sqrt((x - face_center[0])**2 + (y - face_center[1])**2)
                normalized_dist = dist / face_size
                
                if normalized_dist <= 1.0:
                    # Face gets highest depth values (closest to camera)
                    depth_value = 0.9 - 0.2 * normalized_dist
                    depth_map[y, x] = max(depth_value, 0.7)
        
        # Add realistic depth variations
        # Create some background objects with different depths
        for _ in range(3):
            obj_x = np.random.randint(0, w)
            obj_y = np.random.randint(0, h)
            obj_size = np.random.randint(20, 60)
            
            for dy in range(-obj_size, obj_size):
                for dx in range(-obj_size, obj_size):
                    px, py = obj_x + dx, obj_y + dy
                    if 0 <= px < w and 0 <= py < h:
                        obj_dist = np.sqrt(dx**2 + dy**2)
                        if obj_dist <= obj_size:
                            # Background objects have medium depth
                            depth_map[py, px] = 0.3 + 0.2 * (1 - obj_dist / obj_size)
        
        # Smooth the depth map
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
        # Add subtle noise for realism
        noise = np.random.normal(0, 0.02, (h, w))
        depth_map = np.clip(depth_map + noise, 0, 1)
        
        return depth_map
    
    def estimate_depth(self, image_bgr):
        """
        Estimate depth map from input image using Depth-Anything.
        
        Args:
            image_bgr: Input BGR image
            
        Returns:
            Depth map as numpy array
        """
        if self.model is None:
            # Use simulated depth for testing
            print("  🔄 Using simulated depth map (Depth-Anything not available)")
            return self._create_simulated_depth(image_bgr.shape)
        
        # Convert to RGB and normalize to [0,1]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0
        
        # Preprocess using the transform
        input_tensor = self.transform({'image': img_rgb})['image']
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            depth = self.model(input_tensor)
            depth_np = depth.squeeze().cpu().numpy()
        
        # Resize to original image size
        h, w = image_bgr.shape[:2]
        depth_resized = cv2.resize(depth_np, (w, h))
        
        print(f"  📊 Depth range: {depth_resized.min():.3f} to {depth_resized.max():.3f}")
        return depth_resized
    
    def create_portrait_effect(self, image_bgr, depth_map, face_mask=None, 
                             strategy='conservative', threshold=None):
        """
        Create portrait effect using corrected depth logic.
        
        Args:
            image_bgr: Input BGR image
            depth_map: Depth map from Depth-Anything
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
    
    def process_image(self, image_path, output_dir="results/depth_anything", 
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
        print(f"🎭 Processing with Depth-Anything: {image_path}")
        
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
        print("⚡ Running Depth-Anything depth estimation...")
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
            base_name = os.path.join(output_dir, f"{img_name}_depth_anything_{strategy}")
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
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_depth_anything_original.jpg"), img)
        if face_mask is not None:
            face_mask_viz = (face_mask * 255).astype(np.uint8)
            face_mask_viz = cv2.cvtColor(face_mask_viz, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{img_name}_depth_anything_face_mask.jpg"), face_mask_viz)
        
        print(f"💾 Depth-Anything results saved to {output_dir}/")
        return results


def main():
    """Main function for testing Depth-Anything approach."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python depth_anything_portrait.py path/to/image.jpg")
        return
    
    processor = DepthAnythingPortraitProcessor()
    results = processor.process_image(sys.argv[1])
    
    print("✅ Depth-Anything processing completed!")
    for strategy, result in results.items():
        print(f"  - {strategy}: {result['processing_time']:.2f}s")


if __name__ == "__main__":
    main()
