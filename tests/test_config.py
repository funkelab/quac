from quac.config import DataConfig
import pytest


@pytest.fixture
def source():
    return "test_source"


def test_default(source):
    """By default, DataConfig should have rgb=True and grayscale=False."""
    config = DataConfig(source=source)
    assert config.rgb is True, "Default RGB should be True"
    assert config.grayscale is False, "Default Grayscale should be False"


def test_grayscale_true(source):
    """
    If we make a DataConfig with grayscale=True, that should set rgb=False.
    """
    config = DataConfig(source=source, grayscale=True)
    assert config.rgb is False, "Grayscale should set RGB to False"


@pytest.mark.xfail(
    reason="This test is expected to fail due to both rgb and grayscale being set."
)
def test_both_set():
    """
    If we make a DataConfig with both rgb and grayscale set, it should raise an error.
    """
    config = DataConfig(source=source, rgb=True, grayscale=True)
    assert config.rgb != config.grayscale, "RGB and Grayscale should not be both True"

    config = DataConfig(source=source, rgb=False, grayscale=False)
    assert config.rgb != config.grayscale, "RGB and Grayscale should not be both False"
