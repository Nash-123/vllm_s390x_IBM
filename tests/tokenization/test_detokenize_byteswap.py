# test_detokenizer.py
import pytest
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.sequence import Sequence, SequenceGroup, SamplingParams
from .tokenizer_group import MockTokenizerGroup

@pytest.fixture
def tokenizer_group():
    # Mock or create a real tokenizer group as needed
    return MockTokenizerGroup()

@pytest.fixture
def detokenizer(tokenizer_group):
    return Detokenizer(tokenizer_group)

def test_decode_sequence_inplace(detokenizer):
    # Create a mock sequence and sequence group
    sequence = Sequence(id="test_seq", lora_request={"model": "test_model"})
    sequence.set_token_ids([101, 102, 103, 104])  # Mock token IDs
    sequence_group = SequenceGroup(sequences=[sequence])
    
    # Define mock sampling parameters
    sampling_params = SamplingParams(skip_special_tokens=False, spaces_between_special_tokens=True)
    
    # Call the decode method
    detokenizer.decode_sequence_inplace(sequence, sampling_params)
    
    # Assertions to verify the output
    assert sequence.output_text == "expected output"

def test_process_tokens():
    # Example test to verify byte order processing
    input_tokens = [104, 101, 108, 108, 111]  # ASCII for 'hello' in big endian
    expected = [111, 108, 108, 101, 104]  # 'hello' in little endian
    
    result = Detokenizer.process_tokens(input_tokens)
    assert result == expected, "Token byte order not swapped correctly"

