import pytest
import torch
from torchcp.llm.utils.metrics import Metrics, METRICS_REGISTRY_LLM

def test_metrics_registry():
    metrics = Metrics()
    assert "SSCL" in METRICS_REGISTRY_LLM.registered_names()
    assert "average_size" in METRICS_REGISTRY_LLM.registered_names()
    assert "average_sample_size" in METRICS_REGISTRY_LLM.registered_names() 
    assert "average_set_loss" in METRICS_REGISTRY_LLM.registered_names()

def test_SSCL():
    # Create sample data
    prediction_sets = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
    prediction_set_loss = torch.tensor([0.1, 0.2, 0.3])
    
    # Test with default num_bins
    result = METRICS_REGISTRY_LLM.get("SSCL")(
        prediction_sets, 
        prediction_set_loss
    )
    assert isinstance(result, torch.Tensor)
    
    # Test with custom num_bins
    result_custom = METRICS_REGISTRY_LLM.get("SSCL")(
        prediction_sets,
        prediction_set_loss,
        num_bins=5
    )
    assert isinstance(result_custom, torch.Tensor)

def test_average_metrics():
    prediction_sets = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float32)
    
    # Test average_size
    avg_size = METRICS_REGISTRY_LLM.get("average_size")(prediction_sets)
    assert isinstance(avg_size, torch.Tensor)
    assert avg_size.dtype == torch.float32
    
    # Test average_sample_size
    avg_sample_size = METRICS_REGISTRY_LLM.get("average_sample_size")(prediction_sets)
    assert isinstance(avg_sample_size, torch.Tensor)

def test_average_set_loss():
    prediction_sets = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
    prediction_set_loss = torch.tensor([[0.1, 0.2,0.5], [0.2, 0.3, 0.4],
                                      [0.3, 0.4,0.6]], dtype=torch.float32)
    
    result = METRICS_REGISTRY_LLM.get("average_set_loss")(
        prediction_sets,
        prediction_set_loss
    )
    assert isinstance(result, torch.Tensor)

if __name__ == "__main__":
    pytest.main(["-v"])