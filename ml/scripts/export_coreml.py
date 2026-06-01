#!/usr/bin/env python3
"""Export span scorer and WSD models to CoreML .mlpackage format with int8 quantization.

Usage:
    python scripts/export_coreml.py span --model-dir models/cws_span_scorer_electra_small/final
    python scripts/export_coreml.py wsd --model-dir models/wsd_distilled_gte_v2_small/20260529_021557/final
"""

import sys
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ML_DIR / "output" / "coreml"

WSD_OUTPUT_PATH = OUTPUT_DIR / "wsd_encoder.mlpackage"
SPAN_ENCODER_OUTPUT_PATH = OUTPUT_DIR / "span_scorer_encoder.mlpackage"
SPAN_HEAD_OUTPUT_PATH = OUTPUT_DIR / "span_head_weights.bin"

# Tolerance is looser for int8 quantized models
TOLERANCE_FP16 = 6e-2
TOLERANCE_INT8 = 2.5e-1

WSD_TEST_SENTENCES = ["他★打★了我一拳", "请★开★门让我进去吧", "★以★德报怨"]
SPAN_TEST_SENTENCES = ["今天天气很好", "研究生命的起源", "南京市长江大桥"]

# Fixed sequence lengths (single shape per model, no enumerated shapes)
WSD_SEQ_LENGTH = 16
SPAN_SEQ_LENGTH = 32


# -- Model wrappers -----------------------------------------------------------

class WSDWrapper(torch.nn.Module):
    """Wraps a transformer to return L2-normalized CLS embedding.

    Passes pre-computed position_ids and attention_mask to avoid int casts
    inside the embedding layer that coremltools can't convert.
    """
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).to(dtype=input_ids.dtype)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        cls_emb = out.last_hidden_state[:, 0, :]
        return F.normalize(cls_emb, p=2, dim=1)


class SpanEncoderWrapper(torch.nn.Module):
    """Wraps an encoder to return last_hidden_state for span scoring.

    Passes pre-computed position_ids and attention_mask to avoid int casts
    inside Electra's embedding layer that coremltools can't convert.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).to(dtype=input_ids.dtype)
        # Pre-compute position_ids: 0,1,2,...,seq_len-1 (avoids Electra's
        # internal create_position_ids_from_input_ids which uses int casts)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state


# -- CoreML conversion --------------------------------------------------------

def _patch_coremltools_fp16_overflow():
    """Monkey-patch coremltools in memory to clamp values to fp16 range before casting.

    The attention mask in transformers uses -3.4e38 which overflows fp16.
    This patches the cast op's value_inference to clamp before downcasting.
    """
    from coremltools.converters.mil.mil.ops.defs.iOS15 import elementwise_unary as eu

    cast_class = eu.cast
    original_value_inference = cast_class.value_inference

    def _safe_value_inference(self):
        input_var = self.x
        dtype_val = self.dtype.val
        if hasattr(input_var.val, "astype"):
            from coremltools.converters.mil.mil.types.type_mapping import string_to_nptype
            target_dtype = string_to_nptype(dtype_val)
            if target_dtype == np.float16:
                return np.clip(input_var.val, -65504.0, 65504.0).astype(np.float16)
        return original_value_inference(self)

    cast_class.value_inference = _safe_value_inference
    print("Patched coremltools cast op (in-memory) for fp16-safe conversion")


def _to_coreml_fixed(wrapper, coreml_path, output_names, seq_length, quantize_int8=True):
    """Export with a single fixed input shape and optional int8 weight quantization."""
    wrapper.eval()
    dummy_ids = torch.randint(1, 1000, (1, seq_length), dtype=torch.int32)
    traced = torch.jit.trace(wrapper, (dummy_ids,))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_length), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name=n) for n in output_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quantize_int8:
        print(f"Applying int8 weight quantization (per-block, block_size=32)...")
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_block",
                block_size=32,
            )
        )
        mlmodel = linear_quantize_weights(mlmodel, config=config)
        print(f"  Int8 quantization applied")

    coreml_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(coreml_path))
    quant_label = "int8-perblock" if quantize_int8 else "fp16"
    print(f"Saved {coreml_path} (fixed shape: [1, {seq_length}], {quant_label})")


# -- Verification -------------------------------------------------------------

def _verify(coreml_path, tokenizer, pytorch_fn, sentences, output_name, seq_length, tolerance):
    """Verify CoreML output matches PyTorch. Pads/truncates inputs to fixed length."""
    # Use CPU for verification to avoid Metal/ANE shader compilation issues with int8 models
    coreml_model = ct.models.MLModel(str(coreml_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    for s in sentences:
        enc = tokenizer(s, return_tensors="pt")
        ids = enc["input_ids"]
        seq_len = ids.shape[1]

        with torch.no_grad():
            pt_out = pytorch_fn(ids).numpy()

        # Pad or truncate to fixed length
        if seq_len < seq_length:
            pad_len = seq_length - seq_len
            ids = torch.cat([ids, torch.zeros(1, pad_len, dtype=ids.dtype)], dim=1)
        elif seq_len > seq_length:
            ids = ids[:, :seq_length]
            pt_out = pytorch_fn(ids).numpy()

        cm_out = coreml_model.predict({
            "input_ids": ids.numpy().astype(np.int32),
        })[output_name]

        # For sequence outputs, only compare real tokens (ignore padding positions)
        actual_len = min(seq_len, seq_length)
        if len(pt_out.shape) == 3:
            cm_out = cm_out[:, :actual_len, :]
            pt_out = pt_out[:, :actual_len, :]

        diff = np.max(np.abs(pt_out - cm_out))
        status = "PASS" if not np.isnan(diff) and diff < tolerance else "FAIL"
        print(f"  {status}: '{s}' (max diff: {diff:.2e}, tol: {tolerance:.2e})")
        if np.isnan(diff) or diff >= tolerance:
            sys.exit(1)


# -- Export functions ----------------------------------------------------------

def export_wsd(model_dir):
    model_dir = Path(model_dir)
    print(f"Loading WSD from {model_dir}")
    st = SentenceTransformer(str(model_dir), device="cpu")
    st.eval()
    transformer = st[0].auto_model
    transformer.eval()

    wrapper = WSDWrapper(transformer)
    _to_coreml_fixed(wrapper, WSD_OUTPUT_PATH, ["embedding"], WSD_SEQ_LENGTH, quantize_int8=True)

    print("Verifying WSD (int8 quantized)...")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    _verify(WSD_OUTPUT_PATH, tok, wrapper, WSD_TEST_SENTENCES, "embedding", WSD_SEQ_LENGTH, TOLERANCE_INT8)
    print("WSD OK")


def export_span_scorer(model_dir):
    """Export span scorer encoder to CoreML (int8) + head weights to binary."""
    model_dir = Path(model_dir)
    print(f"Loading span scorer from {model_dir}")

    encoder = AutoModel.from_pretrained(str(model_dir))
    encoder.eval()

    wrapper = SpanEncoderWrapper(encoder)
    _to_coreml_fixed(wrapper, SPAN_ENCODER_OUTPUT_PATH, ["hidden_states"], SPAN_SEQ_LENGTH, quantize_int8=True)

    print("Verifying span scorer encoder (int8 quantized)...")
    tok = BertTokenizerFast.from_pretrained(str(model_dir))
    _verify(SPAN_ENCODER_OUTPUT_PATH, tok, wrapper, SPAN_TEST_SENTENCES, "hidden_states", SPAN_SEQ_LENGTH, TOLERANCE_INT8)
    print("Span scorer encoder OK")

    # Export span head weights as raw binary
    head_state = torch.load(
        model_dir / "span_head.pt", map_location="cpu", weights_only=True,
    )
    SPAN_HEAD_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import struct
    key_order = [
        "width_embed.weight",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.3.weight",
        "mlp.3.bias",
    ]
    with open(SPAN_HEAD_OUTPUT_PATH, "wb") as f:
        for key in key_order:
            tensor = head_state[key].flatten().tolist()
            f.write(struct.pack(f"<{len(tensor)}f", *tensor))

    file_size = SPAN_HEAD_OUTPUT_PATH.stat().st_size
    print(f"Saved span head weights to {SPAN_HEAD_OUTPUT_PATH} ({file_size / 1024:.0f} KB)")
    print("Span head weight shapes:")
    for key in key_order:
        print(f"  {key}: {list(head_state[key].shape)}")


# -- Main ---------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["span", "wsd"])
    p.add_argument("--model-dir", required=True, help="Path to the model directory")
    args = p.parse_args()

    # Patch coremltools to handle fp16 overflow in attention mask constants
    _patch_coremltools_fp16_overflow()

    if args.mode == "span":
        export_span_scorer(args.model_dir)
    else:
        export_wsd(args.model_dir)


if __name__ == "__main__":
    main()
