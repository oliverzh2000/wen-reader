import json
from datasets import load_dataset
from collections import Counter

# Load translation cache to understand cedict polysemy
with open('ml/data/translation_cache.json') as f:
    cache = json.load(f)

# Parse cache keys to get word -> senses mapping
cedict_words = Counter()
for key in cache.keys():
    parts = key.split('|')
    word = parts[0]
    cedict_words[word] += 1

print(f'Cedict polysemous words in cache: {len(cedict_words)}')
print(f'Words with 2+ senses: {sum(1 for w, c in cedict_words.items() if c >= 2)}')
print(f'Words with 3+ senses: {sum(1 for w, c in cedict_words.items() if c >= 3)}')
print(f'Words with 5+ senses: {sum(1 for w, c in cedict_words.items() if c >= 5)}')

# Check word length distribution
len_dist = Counter(len(w) for w in cedict_words.keys())
print(f'\nCedict word length distribution:')
for l in sorted(len_dist.keys()):
    print(f'  {l}-char: {len_dist[l]}')

# Load MiCLS
ds = load_dataset('wyy209/MiCLS')
micls_words = set(ex['word'] for ex in ds['train'])
print(f'\nMiCLS unique words: {len(micls_words)}')
print(f'MiCLS word length distribution:')
micls_len = Counter(len(w) for w in micls_words)
for l in sorted(micls_len.keys()):
    print(f'  {l}-char: {micls_len[l]}')

# Overlap
overlap = micls_words & set(cedict_words.keys())
print(f'\nOverlap (MiCLS words in cedict cache): {len(overlap)}')
print(f'MiCLS words NOT in cedict cache: {len(micls_words - set(cedict_words.keys()))}')
