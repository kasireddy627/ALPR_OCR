import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.pipeline import ALPRPipeline

pipeline = ALPRPipeline()

def test_process_frame_returns_annotated_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out   = pipeline.process_frame(frame, frame_number=1)
    assert isinstance(out, np.ndarray), "Output should be numpy array"
    assert out.shape == frame.shape,    "Output shape should match input"

def test_no_crash_on_empty_frame():
    empty = np.zeros((720, 1280, 3), dtype=np.uint8)
    try:
        pipeline.process_frame(empty, frame_number=0)
        assert True
    except Exception as e:
        pytest.fail(f"Crashed on empty frame: {e}")