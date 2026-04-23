import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.preprocessor import ALPRPreprocessor

p = ALPRPreprocessor()

def test_all_four_stages_run():
    img = np.random.randint(0, 255, (60, 200, 3), dtype=np.uint8)
    out = p.process(img)
    assert out.shape == (96, 320), f"Expected (96,320), got {out.shape}"

def test_grayscale_output():
    img = np.random.randint(0, 255, (60, 200, 3), dtype=np.uint8)
    out = p.process(img)
    assert len(out.shape) == 2, "Output should be single channel"

def test_graceful_fallback():
    black = np.zeros((60, 200, 3), dtype=np.uint8)
    try:
        out = p.process(black)
        assert True
    except Exception as e:
        pytest.fail(f"Raised exception on black image: {e}")