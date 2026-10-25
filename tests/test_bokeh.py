"""
Pytest tests for the video_bokeh Chapter 19 modules.

Tests basic depth map creation, visualization, blurring, face mask creation,
and face-aware blurring using synthetic data (no external files needed).
"""

import sys
import os
import importlib.util
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

chapter19_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chapter19")


def _load_module(filename):
    path = os.path.join(chapter19_dir, filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


basic_concepts = _load_module("01_basic_concepts.py")
face_detection = _load_module("02_face_detection.py")

create_simple_depth_map = basic_concepts.create_simple_depth_map
visualize_depth_map = basic_concepts.visualize_depth_map
apply_simple_blur = basic_concepts.apply_simple_blur

create_face_mask = face_detection.create_face_mask
apply_face_aware_blur = face_detection.apply_face_aware_blur


# --- Tests for 01_basic_concepts ---

class TestCreateSimpleDepthMap:
    def test_shape_matches_input(self):
        depth = create_simple_depth_map((100, 80))
        assert depth.shape == (100, 80)

    def test_values_in_range(self):
        depth = create_simple_depth_map((64, 64))
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_center_is_brightest(self):
        depth = create_simple_depth_map((64, 64))
        center_val = depth[32, 32]
        corner_val = depth[0, 0]
        assert center_val > corner_val

    def test_dtype_float32(self):
        depth = create_simple_depth_map((50, 50))
        assert depth.dtype == np.float32


class TestVisualizeDepthMap:
    def test_output_is_bgr(self):
        depth = np.random.rand(60, 80).astype(np.float32)
        colored = visualize_depth_map(depth)
        assert colored.ndim == 3
        assert colored.shape == (60, 80, 3)

    def test_output_uint8(self):
        depth = np.random.rand(40, 40).astype(np.float32)
        colored = visualize_depth_map(depth)
        assert colored.dtype == np.uint8

    def test_uniform_depth_does_not_crash(self):
        depth = np.ones((30, 30), dtype=np.float32) * 0.5
        colored = visualize_depth_map(depth)
        assert colored.shape == (30, 30, 3)


class TestApplySimpleBlur:
    def _make_inputs(self, h=64, w=64):
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        depth = create_simple_depth_map((h, w))
        return image, depth

    def test_output_shape_matches_input(self):
        image, depth = self._make_inputs()
        result, mask = apply_simple_blur(image, depth, threshold=0.5)
        assert result.shape == image.shape

    def test_result_dtype_uint8(self):
        image, depth = self._make_inputs()
        result, _ = apply_simple_blur(image, depth, threshold=0.5)
        assert result.dtype == np.uint8

    def test_mask_is_2d_float(self):
        image, depth = self._make_inputs()
        _, mask = apply_simple_blur(image, depth, threshold=0.5)
        assert mask.ndim == 2
        assert mask.dtype == np.float32

    def test_zero_threshold_blurs_nothing(self):
        image, depth = self._make_inputs()
        result, mask = apply_simple_blur(image, depth, threshold=0.0)
        assert mask.max() < 0.01


# --- Tests for 02_face_detection ---

class TestCreateFaceMask:
    def test_no_faces_returns_zeros(self):
        mask = create_face_mask((100, 100), [])
        assert mask.shape == (100, 100)
        assert mask.max() == 0.0

    def test_single_face_creates_nonzero_mask(self):
        faces = [(20, 20, 30, 30)]
        mask = create_face_mask((100, 100), faces)
        assert mask.max() > 0.0

    def test_face_region_is_marked(self):
        faces = [(20, 20, 40, 40)]
        mask = create_face_mask((100, 100), faces, margin=0.0)
        assert mask[40, 40] > 0.0

    def test_multiple_faces(self):
        faces = [(10, 10, 20, 20), (60, 60, 20, 20)]
        mask = create_face_mask((100, 100), faces)
        assert mask[20, 20] > 0.0
        assert mask[70, 70] > 0.0

    def test_dtype_float32(self):
        mask = create_face_mask((64, 64), [(10, 10, 20, 20)])
        assert mask.dtype == np.float32


class TestApplyFaceAwareBlur:
    def _make_inputs(self, h=64, w=64):
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        depth = create_simple_depth_map((h, w))
        face_mask = create_face_mask((h, w), [(h // 4, w // 4, h // 2, w // 2)])
        return image, depth, face_mask

    def test_output_shape(self):
        image, depth, face_mask = self._make_inputs()
        result, blur_mask = apply_face_aware_blur(image, depth, face_mask)
        assert result.shape == image.shape

    def test_result_dtype_uint8(self):
        image, depth, face_mask = self._make_inputs()
        result, _ = apply_face_aware_blur(image, depth, face_mask)
        assert result.dtype == np.uint8

    def test_face_region_less_blurred(self):
        image, depth, face_mask = self._make_inputs(128, 128)
        _, blur_mask = apply_face_aware_blur(image, depth, face_mask, threshold=0.5)
        face_area_blur = blur_mask[32:96, 32:96].mean()
        assert face_area_blur < blur_mask.mean() or blur_mask.mean() < 0.01

    def test_none_face_mask(self):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        depth = create_simple_depth_map((64, 64))
        result, _ = apply_face_aware_blur(image, depth, None, threshold=0.5)
        assert result.shape == image.shape
