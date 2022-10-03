import pytest
from ser.transforms import transforms, normalize, flip
import torch

# Add a pytest unit test for 
def test_flip():
    img = torch.FloatTensor([[[1,2,3],[4,5,6], [7,8,9]]])
    expectation = torch.FloatTensor([[[9,8,7],[6,5,4], [3,2,1]]])
    assert torch.equal(flip()(img), expectation)