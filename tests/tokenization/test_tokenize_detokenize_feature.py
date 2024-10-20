import pytest
from transformers import AutoTokenizer

from vllm.sequence import Logprob, SamplingParams, Sequence, SequenceGroup
from vllm.transformers_utils.detokenizer import Detokenizer, detokenize_incrementally
from vllm.transformers_utils.tokenizer_group import get_tokenizer_group

# Custom test data to focus on potential issues
CUSTOM_TEST_DATA = [
    ("This is a test sequence, including special characters like ñ and emojis 🚀.", "bigscience/bloom-560m"),
    ("Here we include some high-value Unicode points like 😂, 🥺, 👍 to see how they are processed.", "gpt2"),
    ("Testing byte order issues with multibyte characters: 漢字, 한글, кирилица.", "facebook/opt-125m")
]


@pytest.fixture
def detokenizer(tokenizer_name: str) -> Detokenizer:
    # Providing default values for missing parameters
    max_num_seqs = 100  # Default value for maximum number of sequences
    max_input_length = 512  # Default maximum input length
    tokenizer_group = get_tokenizer_group(
        None,
        tokenizer_id=tokenizer_name,
        enable_lora=False,
        max_num_seqs=max_num_seqs,
        max_input_length=max_input_length
    )
    return Detokenizer(tokenizer_group)

@pytest.mark.parametrize("test_string, tokenizer_name", CUSTOM_TEST_DATA)
def test_special_character_handling(test_string: str, tokenizer_name: str, detokenizer: Detokenizer):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.encode(test_string, add_special_tokens=True)
    decoded_text = ""
    prev_tokens = None
    last_known_length = 0  # Track the length of decoded_text after each append

    for i in range(1, len(tokens) + 1):
        new_tokens, text, _, _ = detokenize_incrementally(
            tokenizer,
            tokens[:i],
            prev_tokens,
            0,
            0,
            skip_special_tokens=False)

        if prev_tokens is None:
            prev_tokens = new_tokens
            decoded_text = text
            last_known_length = len(text)
        else:
            # Append only new text beyond the last known length
            if len(text) > last_known_length:
                decoded_text += text[last_known_length:]
                last_known_length = len(decoded_text)

        prev_tokens = new_tokens  # Update the tokens for the next iteration

    assert decoded_text == test_string, f"Failed to correctly detokenize for tokenizer {tokenizer_name}"

@pytest.mark.parametrize("test_string, tokenizer_name", [
    ("Simple test", "gpt2"),  # A very basic case
    ("😂🥺👍", "gpt2"),  # Only emojis
    ("ñ and 🚀", "gpt2")  # Special characters mixed with emojis
])
def test_diagnostic_character_handling(test_string: str, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.encode(test_string, add_special_tokens=True)
    decoded_text = ""
    prev_tokens = None

    for i in range(1, len(tokens) + 1):
        new_tokens, text, _, _ = detokenize_incrementally(
            tokenizer,
            tokens[:i],
            prev_tokens,
            0,
            0,
            skip_special_tokens=False)

        # Append full text on first iteration, then only changes
        if prev_tokens is None:
            decoded_text = text
        else:
            new_part = text[len(decoded_text):]  # Capture new text
            decoded_text += new_part

        prev_tokens = new_tokens  # Update the tokens for the next iteration

    assert decoded_text == test_string, f"Mismatch for tokenizer {tokenizer_name}"

