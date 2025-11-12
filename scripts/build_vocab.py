"""
Build vocabulary properly from training corpus.
This version DOES NOT pre-populate and lets vocab grow naturally.
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def build_vocab_properly(corpus_dir: str, vocab_size: int, output_path: str):
    """Build vocabulary the right way - from actual training text."""
    
    print(f"Building vocabulary from: {corpus_dir}")
    print(f"Target vocab size: {vocab_size}")
    
    # Step 1: Read all text
    corpus_path = Path(corpus_dir)
    all_words = []
    
    for txt_file in sorted(corpus_path.glob('*.txt')):
        print(f"  Reading: {txt_file.name}")
        text = txt_file.read_text(encoding='utf-8')
        words = text.split()
        all_words.extend(words)
    
    print(f"  Total words: {len(all_words):,}")
    
    # Step 2: Count word frequencies
    word_counts = Counter(all_words)
    unique_words = len(word_counts)
    print(f"  Unique words: {unique_words:,}")
    
    # Step 3: Build vocabulary from most common words
    # Start with special tokens
    itos = ['<pad>', '<unk>']
    
    # Add most common words up to vocab_size
    max_words = vocab_size - len(itos)
    most_common = word_counts.most_common(max_words)
    
    for word, count in most_common:
        itos.append(word)
    
    # Pad to exact vocab_size with placeholder tokens
    # This ensures the model trains on vocab_size embeddings
    while len(itos) < vocab_size:
        itos.append(f'<pad_{len(itos)}>')
    
    # Create reverse mapping
    stoi = {token: idx for idx, token in enumerate(itos)}
    
    actual_vocab_size = len(itos)
    actual_words = sum(1 for t in itos if not t.startswith('<'))
    print(f"\n✓ Vocabulary built: {actual_vocab_size} tokens")
    
    # Show sample
    print("\nFirst 30 tokens in vocabulary:")
    for i in range(min(30, len(itos))):
        token = itos[i]
        count = word_counts.get(token, 0) if token not in ['<pad>', '<unk>'] else '-'
        print(f"  {i:3d}: {token:20s} (count: {count})")
    
    # Step 4: Save vocabulary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    vocab_data = {
        'itos': itos,
        'fixed_vocab_size': vocab_size
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved vocabulary to: {output_path}")
    print(f"  Actual size: {actual_vocab_size} tokens")
    print(f"  Coverage: {actual_vocab_size - 2}/{unique_words} unique words")
    
    return itos, stoi


def test_vocab(vocab_path: str, test_text: str):
    """Test the vocabulary by encoding/decoding."""
    
    print(f"\n{'='*70}")
    print("Testing Vocabulary")
    print('='*70)
    
    # Load vocab
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    itos = vocab_data['itos']
    stoi = {token: idx for idx, token in enumerate(itos)}
    
    print(f"✓ Loaded vocab: {len(itos)} tokens")
    
    # Encode test text
    print(f"\nTest text: {test_text}")
    words = test_text.split()
    token_ids = []
    
    for word in words:
        if word in stoi:
            token_ids.append(stoi[word])
        else:
            token_ids.append(stoi['<unk>'])
    
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded_words = []
    for tid in token_ids:
        decoded_words.append(itos[tid])
    
    decoded = ' '.join(decoded_words)
    print(f"Decoded: {decoded}")
    
    # Check coverage
    unk_count = sum(1 for tid in token_ids if tid == 1)
    known_count = len(token_ids) - unk_count
    
    if unk_count > 0:
        print(f"\n⚠ Coverage: {known_count}/{len(token_ids)} tokens recognized")
        print(f"  Unknown words:")
        for word in words:
            if word not in stoi:
                print(f"    - {word}")
    else:
        print(f"\n✓ Perfect coverage: all {len(token_ids)} tokens recognized!")
    
    return token_ids, decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-dir', type=str, required=True)
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--test-text', type=str, 
                       default="The continuous reasoning state model")
    
    args = parser.parse_args()
    
    # Build vocab
    itos, stoi = build_vocab_properly(
        corpus_dir=args.corpus_dir,
        vocab_size=args.vocab_size,
        output_path=args.output
    )
    
    # Test vocab
    test_vocab(args.output, args.test_text)
    
    print(f"\n{'='*70}")
    print("✓ Vocabulary ready for training!")
    print('='*70)


if __name__ == '__main__':
    main()