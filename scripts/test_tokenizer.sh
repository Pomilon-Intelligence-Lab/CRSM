# Test tokenizer works
python << 'EOF'
from crsm.tokenizer import Tokenizer

tok = Tokenizer(hf_name='gpt2')
text = "The CRSM architecture enables autonomous reasoning"
ids = tok.encode(text)
decoded = tok.decode(ids)

print(f"Original: {text}")
print(f"Token IDs: {ids}")
print(f"Decoded: {decoded}")
print(f"Vocab size: {tok.vocab_size}")
print("✓ Tokenizer working!")
EOF