# WSD Biencoder — Findings

## Best results (hard eval, 239 examples)

| Model | TOP-1 | TOP-3 | MRR |
|-------|:---:|:---:|:---:|
| GTE-base fp32 | **88.7%** | 99.6% | 0.936 |
| GTE-base fp16 model + int8 emb | **88.7%** | 99.6% | 0.936 |
| Distilled small (base→small) | 79.9% | 98.3% | 0.887 |
| Distilled small int8 model + int8 emb | 80.8% | 98.3% | 0.889 |
| GTE-base int8 model (broken) | 73.2% | 93.7% | 0.839 |

## Key findings

- **Data > architecture**: merging v1+v2 dataset (183k examples) jumped base from 64%→88.7%
- **Large doesn't beat base**: GTE-large matched but didn't exceed base — task saturated at this data quality
- **MLP scoring head doesn't help**: dot product is sufficient, removed
- **int8 model weights break GTE-base** (88.7→73.2%) but are fine on small
- **fp16 model weights are lossless** on base
- **Embedding quantization (fp16 or int8) is completely lossless** — unit-normalized vectors with 768 dims have huge margin
- **Distillation works**: base→small gives 79.9% vs raw small 64.9%

## Sense cluster storage

10,092 polysemous words, 23,579 total sense clusters (avg 2.3 clusters/word).

| Format | GTE-base (768d) | GTE-small (512d) |
|--------|:---:|:---:|
| fp32 | 69 MB | 46 MB |
| fp16 | 35 MB | 23 MB |
| int8 + scale | 17 MB | 12 MB |

## Deployment options

| Config | Accuracy | Model size | Embeddings | Total |
|--------|:---:|:---:|:---:|:---:|
| Base fp16 + int8 emb | **88.7%** | ~220 MB | ~17 MB | ~237 MB |
| Distilled small int8 + int8 emb | **80.8%** | ~8 MB | ~12 MB | ~20 MB |
