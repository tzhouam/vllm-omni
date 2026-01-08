#!/usr/bin/env python3
"""
Generate Qwen3 Omni Model Structure Visualization

This script loads the Qwen3 Omni MoE model, runs a simple inference with tracing
enabled, and outputs a mermaid diagram showing:
- Model structure and layer hierarchy
- Layer dependencies (call order)
- Input/Output shapes and dtypes for each layer

Usage:
    python generate_model_structure.py --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct
    python generate_model_structure.py --model_path /path/to/local/model --output_dir ./results
"""

import argparse
import os
import sys

import torch
from qwen3_omni_moe_transformers import (
    Qwen3OmniMoeForConditionalGenerationWithTracing,
)
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Qwen3 Omni Model Structure Mermaid Diagram"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Path to the model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_structure_output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="Maximum module nesting depth to trace (default: 4)",
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        nargs="*",
        default=None,
        help="Regex patterns for module names to include (e.g., 'thinker' 'embed')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Text prompt for test inference",
    )
    parser.add_argument(
        "--video_url",
        type=str,
        default=None,
        help="Optional video URL to test multimodal input",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Max new tokens for generation (keep small for faster tracing)",
    )
    parser.add_argument(
        "--no_audio",
        action="store_true",
        help="Disable audio generation (only generate text)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)",
    )
    parser.add_argument(
        "--print_mermaid",
        action="store_true",
        help="Print mermaid diagram to stdout",
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=499,
        help="Maximum edges per mermaid diagram (default: 499)",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=200,
        help="Maximum nodes per mermaid diagram (default: 200)",
    )
    parser.add_argument(
        "--max_layer_depth",
        type=int,
        default=3,
        help="Maximum layer nesting depth to show (default: 3)",
    )
    parser.add_argument(
        "--no_compact",
        action="store_true",
        help="Disable compact mode (show full layer names)",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Don't split mermaid by component (save as single file)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Qwen3 Omni Model Structure Generator")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model with tracing capability
    print(f"\n[1/4] Loading model from {args.model_path}...")
    print("      This may take a while for large models...")

    try:
        model = Qwen3OmniMoeForConditionalGenerationWithTracing.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map=args.device_map,
            attn_implementation="flash_attention_2",
        )
        print(f"      Model loaded successfully on {model.device}")
    except Exception as e:
        print(f"      Error loading model: {e}")
        print("      Trying without flash_attention_2...")
        model = Qwen3OmniMoeForConditionalGenerationWithTracing.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map=args.device_map,
        )
        print(f"      Model loaded successfully on {model.device}")

    # Load processor
    print(f"\n[2/4] Loading processor...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    print("      Processor loaded successfully")

    # Enable tracing
    print(f"\n[3/4] Enabling tracing (max_depth={args.max_depth})...")
    model.enable_tracing(
        max_depth=args.max_depth,
        include_patterns=args.include_patterns,
    )

    # Prepare input
    print(f"\n[4/4] Running inference with tracing...")
    print(f"      Prompt: {args.prompt[:50]}..." if len(args.prompt) > 50 else f"      Prompt: {args.prompt}")

    # Build conversation
    if args.video_url:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": args.video_url},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]
        use_audio_in_video = True
    else:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            },
        ]
        use_audio_in_video = False

    # Process inputs
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=use_audio_in_video,
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    print(f"      Input IDs shape: {inputs['input_ids'].shape}")

    # Run generation with tracing
    print("      Running generation (this traces all forward passes)...")
    try:
        result = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=not args.no_audio,
            thinker_max_new_tokens=args.max_new_tokens,
            thinker_return_dict_in_generate=True,
        )
        print("      Generation completed!")

        # Decode output
        if isinstance(result, tuple):
            text_result = result[0]
        else:
            text_result = result

        if hasattr(text_result, "sequences"):
            output_ids = text_result.sequences
        else:
            output_ids = text_result

        output_text = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"      Generated text: {output_text[:100]}..." if len(output_text) > 100 else f"      Generated text: {output_text}")

    except Exception as e:
        print(f"      Warning: Generation failed with error: {e}")
        print("      Continuing to save partial tracing results...")

    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    split_by_component = not args.no_split
    compact = not args.no_compact
    model.save_tracing_results(
        args.output_dir,
        prefix="qwen3_omni",
        max_edges=args.max_edges,
        max_nodes=args.max_nodes,
        compact=compact,
        max_layer_depth=args.max_layer_depth,
        split_by_component=split_by_component,
    )

    # Print mermaid to stdout if requested
    if args.print_mermaid:
        print("\n" + "=" * 70)
        print("Mermaid Diagram")
        print("=" * 70)
        if model.tracer:
            diagrams = model.tracer.generate_mermaid_by_component(
                max_edges=args.max_edges,
                max_nodes=args.max_nodes,
                compact=compact,
                max_layer_depth=args.max_layer_depth,
            )
            for component, mermaid in diagrams.items():
                print(f"\n### {component.capitalize()}\n")
                print("```mermaid")
                print(mermaid)
                print("```\n")

    # Print summary
    if model.tracer:
        summary = model.tracer.get_summary()
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"  Total hooks registered: {summary['total_hooks']}")
        print(f"  Total layers traced:    {summary['total_layers']}")
        print(f"  Max nodes per diagram:  {args.max_nodes}")
        print(f"  Max edges per diagram:  {args.max_edges}")
        print(f"  Max layer depth:        {args.max_layer_depth}")
        print(f"  Compact mode:           {compact}")
        print(f"  Split by component:     {split_by_component}")


    print(f"\nOutput files saved to: {args.output_dir}/")
    if split_by_component:
        print("  - qwen3_omni_thinker_overall_mermaid.md  (Thinker Complete Overview)")
        print("  - qwen3_omni_thinker_mermaid.md          (Thinker Base Model)")
        print("  - qwen3_omni_thinker_moe_mermaid.md      (Thinker MoE)")
        print("  - qwen3_omni_talker_overall_mermaid.md   (Talker Complete Overview)")
        print("  - qwen3_omni_talker_lm_mermaid.md        (Talker Language Model)")
        print("  - qwen3_omni_talker_moe_mermaid.md       (Talker MoE)")
        print("  - qwen3_omni_talker_mtp_mermaid.md       (Talker MTP)")
        print("  - qwen3_omni_code2wav_mermaid.md         (Code2Wav Vocoder)")
    else:
        print("  - qwen3_omni_mermaid.md  (Combined mermaid diagram)")
    print("  - qwen3_omni_layers.md   (Layer details table)")
    print("  - qwen3_omni_data.json   (Raw tracing data)")

    # Disable tracing and cleanup
    model.disable_tracing()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
