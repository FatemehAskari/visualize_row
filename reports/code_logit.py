#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL logit-lens over image patches (single forward, no per-token loop).

For each image:
  - Build a chat with (image + question).
  - One forward pass with output_hidden_states=True.
  - For EACH hidden-state index (embedding + all transformer layers):
      For EACH image token (patch) in the visual span:
        * Apply final norm -> lm_head -> softmax (full vocab)
        * Save Top-K vocab (id, piece, prob)
        * Save digit ('0'..'9') probability mass:
            - digits_abs: sum of probs over single-token variants ('3', ' 3', ...)
            - digits_renorm: digits_abs renormalized to sum to 1
        * Save patch location: (row, col), linear index, bbox on resized canvas
Writes one JSON per image with this schema:

{
  "image_path": str,
  "prompt": str,
  "vision": {
    "span": {"start": int, "end": int, "length": int},
    "grid_h": int, "grid_w": int,
    "merged_patch_px": int,
    "resized_w": int, "resized_h": int
  },
  "layers": [
    {
      "layer_idx": int,
      "patches": [
        {
          "patch_index": int, "row": int, "col": int,
          "bbox_resized": [x0, y0, x1, y1],
          "topk": [{"id": int, "piece": str, "prob": float}, ...],
          "digits_abs": {"0": float, ... "9": float},
          "digits_renorm": {"0": float, ... "9": float}
        }, ...
      ]
    }, ...
  ]
}
"""

import os
import glob
import json
from typing import List, Dict, Any, Tuple

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# ========================= CONFIG =========================
MODEL_DIR   = "/home/mmd/qwen/models/Qwen2.5-VL-7B-Instruct"
IMAGES_DIR  = "/home/mmd/qwen/logit_lens/data/images"
OUTPUT_DIR  = "logitlens_json_once"
QUESTION    = "How many circles are there in this image?"

TOPK = 5                    # Top-K vocab to store per (layer, patch)
INCLUDE_EMBED_IDX = True    # include hidden_states[0] (embedding) as layer_idx=0
# =========================================================


# -------------------- Vision helper --------------------
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    # Fallback: load from path(s) directly
    def process_vision_info(messages: List[dict]):
        imgs, sizes = [], []
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "image":
                    path = part["image"]
                    img = Image.open(path).convert("RGB")
                    imgs.append(img)
                    sizes.append((img.height, img.width))
        return imgs, sizes


# -------------------- IO helpers --------------------
def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff")
    files: List[str] = []
    for p in exts:
        files.extend(glob.glob(os.path.join(folder, p)))
    return sorted(files)


# -------------------- Model / Processor --------------------
def load_model_and_processor(model_dir: str):
    """
    Load Qwen2.5-VL model + processor. Use bf16 on CUDA, else fp32.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",     # harmless; kept for compatibility
        local_files_only=os.path.isdir(model_dir),
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=os.path.isdir(model_dir)
    )
    model.eval()
    return model, processor, device


# -------------------- Prompt packer --------------------
def build_messages(image_path: str, question: str) -> List[dict]:
    """Single user turn: image + text question."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path, "max_pixels": 512*28*28},
            {"type": "text", "text": question},
        ],
    }]


# -------------------- Grid derivation --------------------
def derive_grid(processor, image_inputs, model) -> Tuple[int, int, int, int, int]:
    """
    Recover merged-patch grid geometry from the processor+model config.
    Returns:
      grid_h, grid_w, merged_patch_px, resized_w, resized_h
    """
    meta = processor.image_processor(images=image_inputs)
    thw = meta["image_grid_thw"].to("cpu").numpy().squeeze(0)  # [T, H_raw, W_raw]
    patch_size = int(getattr(model.config.vision_config, "patch_size", 14))
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    grid_h = int(thw[1] // merge_size)
    grid_w = int(thw[2] // merge_size)
    resized_h = int(thw[1] * patch_size)
    resized_w = int(thw[2] * patch_size)
    merged_patch_px = int(patch_size * merge_size)
    return grid_h, grid_w, merged_patch_px, resized_w, resized_h


# -------------------- Image token span --------------------
def find_image_span(processor, input_ids: torch.Tensor) -> Tuple[int, int]:
    """
    Locate visual token span (start,end) between <|vision_start|> and <|vision_end|>.
    """
    tok = processor.tokenizer
    vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tok.convert_tokens_to_ids("<|vision_end|>")
    seq = input_ids[0].tolist()
    try:
        start = seq.index(vs_id) + 1
        end   = seq.index(ve_id)
    except ValueError:
        raise RuntimeError("Missing <|vision_start|> / <|vision_end|> in input_ids.")
    if not (0 <= start < end <= len(seq)):
        raise RuntimeError("Invalid visual span indices.")
    return start, end


# -------------------- Final norm + lm_head --------------------
def _apply_final_norm(model, h: torch.Tensor) -> torch.Tensor:
    """
    Apply model's final norm before projecting to vocab.
    Handles common attribute names for Qwen variants.
    """
    if h.dim() == 1:
        h = h.unsqueeze(0)  # [1, hidden]
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(h)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(h)
    return h  # fallback (less ideal)


def logits_from_hidden(model, h: torch.Tensor) -> torch.Tensor:
    """hidden -> final_norm -> lm_head -> logits."""
    h = _apply_final_norm(model, h)
    logits = model.lm_head(h)  # [1, vocab]
    return logits


# -------------------- Digit utilities --------------------
def build_digit_id_sets(tokenizer) -> Dict[str, List[int]]:
    """
    Map each digit '0'..'9' to a set of single-token ids (e.g., '3', ' 3').
    """
    out: Dict[str, List[int]] = {}
    for d in range(10):
        s = str(d)
        cand = [s, " " + s]
        ids: List[int] = []
        for c in cand:
            enc = tokenizer.encode(c, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        out[s] = sorted(set(ids))
    return out


def digit_probs_from_softmax(probs: torch.Tensor, digit_id_sets: Dict[str, List[int]]):
    """
    From full-vocab softmax (1D), compute:
      - digits_abs: prob mass for each digit (sum over its variants)
      - digits_renorm: digits_abs renormalized to sum to 1
    """
    abs_probs: Dict[str, float] = {}
    for d, ids in digit_id_sets.items():
        p = float(sum(probs[i].item() for i in ids)) if ids else 0.0
        abs_probs[d] = p
    s = sum(abs_probs.values())
    renorm = {d: (abs_probs[d] / s if s > 0 else 0.0) for d in abs_probs}
    return abs_probs, renorm


# -------------------- Core: one image (single forward) --------------------
def run_one_image_once(image_path: str,
                       model: Qwen2_5_VLForConditionalGeneration,
                       processor: AutoProcessor,
                       device: str,
                       question: str,
                       topk: int) -> Dict[str, Any]:
    """
    Single forward pass to get hidden_states; for each layer × image patch:
      compute Top-K and digit probabilities. No generation / no replay.
    """
    # Pack (image + prompt)
    messages = build_messages(image_path, question)
    image_inputs, _ = process_vision_info(messages)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch = processor(
        text=[chat_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Grid + span
    grid_h, grid_w, merged_patch_px, resized_w, resized_h = derive_grid(processor, image_inputs, model)
    pos, pos_end = find_image_span(processor, batch["input_ids"])
    n_img_tokens = pos_end - pos
    assert n_img_tokens == grid_h * grid_w, f"Span {n_img_tokens} != grid_h*grid_w {grid_h*grid_w}"

    # One forward with hidden states
    with torch.no_grad():
        out = model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )

    # Choose which hidden_state indices to use (0=embedding if present)
    H_idxs = list(range(len(out.hidden_states)))
    if not INCLUDE_EMBED_IDX and len(H_idxs) > 0:
        H_idxs = H_idxs[1:]

    tok = processor.tokenizer
    digit_id_sets = build_digit_id_sets(tok)

    # Build per-layer outputs
    layers_out: List[Dict[str, Any]] = []
    for hidx in H_idxs:
        patch_entries: List[Dict[str, Any]] = []
        HS = out.hidden_states[hidx][0]  # [seq_len, hidden]

        for offset in range(n_img_tokens):
            patch_pos = pos + offset
            row = offset // grid_w
            col = offset %  grid_w

            # bbox on resized canvas (for visualization overlays if needed)
            x0 = col * merged_patch_px
            y0 = row * merged_patch_px
            x1 = x0 + merged_patch_px
            y1 = y0 + merged_patch_px

            # hidden -> logits -> softmax
            h = HS[patch_pos, :]                          # [hidden]
            logits = logits_from_hidden(model, h)         # [1, vocab]
            probs = torch.softmax(logits, dim=-1)[0]      # [vocab]

            # Top-K vocab
            pvals, tids = probs.topk(topk)
            pieces = tok.convert_ids_to_tokens(tids.tolist())
            topk_list = [
                {"id": int(tids[i]), "piece": pieces[i], "prob": float(pvals[i])}
                for i in range(len(pvals))
            ]

            # Digits (abs + renorm)
            digits_abs, digits_renorm = digit_probs_from_softmax(probs, digit_id_sets)

            patch_entries.append({
                "patch_index": offset,
                "row": row,
                "col": col,
                "bbox_resized": [int(x0), int(y0), int(x1), int(y1)],
                "topk": topk_list,
                "digits_abs": digits_abs,
                "digits_renorm": digits_renorm
            })

        layers_out.append({
            "layer_idx": hidx,
            "patches": patch_entries
        })

    # JSON payload (no generated tokens/text here)
    payload: Dict[str, Any] = {
        "image_path": image_path,
        "prompt": QUESTION,
        "vision": {
            "span": {"start": pos, "end": pos_end, "length": n_img_tokens},
            "grid_h": grid_h,
            "grid_w": grid_w,
            "merged_patch_px": merged_patch_px,
            "resized_w": resized_w,
            "resized_h": resized_h
        },
        "layers": layers_out
    }
    return payload


# -------------------- Batch driver --------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, processor, device = load_model_and_processor(MODEL_DIR)
    print(f"Model & processor ready on {device}")

    images = list_images(IMAGES_DIR)
    if not images:
        raise FileNotFoundError(f"No images found in: {IMAGES_DIR}")
    print(f"Found {len(images)} images.")

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path}")
        try:
            payload = run_one_image_once(
                image_path=img_path,
                model=model,
                processor=processor,
                device=device,
                question=QUESTION,
                topk=TOPK
            )
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_json = os.path.join(OUTPUT_DIR, f"logitlens_once_{stem}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  -> saved {out_json}")

    print("Done.")


if __name__ == "__main__":
    main()
