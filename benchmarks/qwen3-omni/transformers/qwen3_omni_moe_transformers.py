import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import soundfile as sf
import torch
import torch.nn as nn
from qwen3_omni_moe_model import Qwen3OmniMoeForConditionalGenerationWithLogging
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


# ==============================================================================
# LayerTracer: Track model layer info during forward passes
# ==============================================================================


@dataclass
class LayerInfo:
    """Information about a single layer's forward pass."""

    name: str
    module_type: str
    input_shapes: list[tuple]
    input_dtypes: list[str]
    output_shapes: list[tuple]
    output_dtypes: list[str]
    call_order: int


class LayerTracer:
    """
    Traces model layers during forward passes, distinguishing between
    prefill and decode phases in autoregressive generation.
    """

    def __init__(self, max_depth: int = 5, include_patterns: list[str] | None = None):
        """
        Args:
            max_depth: Maximum module nesting depth to trace (to avoid too much detail)
            include_patterns: List of regex patterns for module names to include.
                              If None, traces all modules up to max_depth.
        """
        self.prefill_info: list[LayerInfo] = []
        self.decode_info: list[LayerInfo] = []
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.current_phase: str = "prefill"
        self.decode_recorded: bool = False
        self.call_counter: int = 0
        self.max_depth: int = max_depth
        self.include_patterns: list[re.Pattern] = (
            [re.compile(p) for p in include_patterns] if include_patterns else None
        )
        self._module_to_name: dict[int, str] = {}

    def _get_tensor_info(self, tensor: Any) -> tuple[tuple | None, str | None]:
        """Extract shape and dtype from a tensor or nested structure."""
        if isinstance(tensor, torch.Tensor):
            return tuple(tensor.shape), str(tensor.dtype)
        elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
            # Get info from first element if it's a tensor
            first = tensor[0]
            if isinstance(first, torch.Tensor):
                return tuple(first.shape), str(first.dtype)
        return None, None

    def _extract_shapes_dtypes(
        self, data: Any
    ) -> tuple[list[tuple], list[str]]:
        """Extract all shapes and dtypes from input/output data."""
        shapes = []
        dtypes = []

        if isinstance(data, torch.Tensor):
            shapes.append(tuple(data.shape))
            dtypes.append(str(data.dtype))
        elif isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, torch.Tensor):
                    shapes.append(tuple(item.shape))
                    dtypes.append(str(item.dtype))
                elif isinstance(item, (list, tuple)):
                    # Nested structure
                    sub_shapes, sub_dtypes = self._extract_shapes_dtypes(item)
                    shapes.extend(sub_shapes)
                    dtypes.extend(sub_dtypes)
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    shapes.append(tuple(v.shape))
                    dtypes.append(str(v.dtype))

        return shapes, dtypes

    def _should_trace(self, name: str, depth: int) -> bool:
        """Determine if a module should be traced based on depth and patterns."""
        if depth > self.max_depth:
            return False
        if self.include_patterns is None:
            return True
        return any(p.search(name) for p in self.include_patterns)

    def _create_hook(self, name: str):
        """Create a forward hook for a specific module."""

        def hook_fn(module: nn.Module, input: Any, output: Any):
            # Skip if in decode phase and already recorded
            if self.current_phase == "decode" and self.decode_recorded:
                return

            input_shapes, input_dtypes = self._extract_shapes_dtypes(input)
            output_shapes, output_dtypes = self._extract_shapes_dtypes(output)

            layer_info = LayerInfo(
                name=name,
                module_type=module.__class__.__name__,
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
                call_order=self.call_counter,
            )
            self.call_counter += 1

            if self.current_phase == "prefill":
                self.prefill_info.append(layer_info)
            else:
                self.decode_info.append(layer_info)

        return hook_fn

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on all relevant modules."""
        self.clear()

        def _register_recursive(module: nn.Module, prefix: str, depth: int):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if self._should_trace(full_name, depth):
                    hook = child.register_forward_hook(self._create_hook(full_name))
                    self.hooks.append(hook)
                    self._module_to_name[id(child)] = full_name
                _register_recursive(child, full_name, depth + 1)

        _register_recursive(model, "", 0)
        print(f"[LayerTracer] Registered {len(self.hooks)} hooks")

    def switch_to_decode(self):
        """Switch to decode phase tracking."""
        self.current_phase = "decode"
        self.call_counter = 0

    def mark_decode_complete(self):
        """Mark that decode phase has been recorded (to avoid duplicates)."""
        self.decode_recorded = True

    def clear(self):
        """Remove all hooks and clear recorded data."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.prefill_info.clear()
        self.decode_info.clear()
        self._module_to_name.clear()
        self.current_phase = "prefill"
        self.decode_recorded = False
        self.call_counter = 0

    def _sanitize_node_id(self, name: str) -> str:
        """Convert module name to a valid mermaid node ID."""
        return name.replace(".", "_").replace("-", "_")

    def _format_shape(self, shape: tuple) -> str:
        """Format a shape tuple for display."""
        return "x".join(str(d) for d in shape)

    def _format_dtype(self, dtype: str) -> str:
        """Format dtype string (remove 'torch.' prefix)."""
        return dtype.replace("torch.", "")

    def _filter_by_component(
        self, info_list: list[LayerInfo], component: str | None
    ) -> list[LayerInfo]:
        """Filter layer info by component (thinker, talker, code2wav)."""
        if component is None:
            return info_list
        return [info for info in info_list if info.name.startswith(component)]

    def _filter_by_depth(
        self, info_list: list[LayerInfo], max_layer_depth: int | None
    ) -> list[LayerInfo]:
        """Filter layers by nesting depth (number of dots in name)."""
        if max_layer_depth is None:
            return info_list
        return [info for info in info_list if info.name.count(".") <= max_layer_depth]

    def _shorten_name(self, name: str) -> str:
        """Shorten layer name for compact display."""
        # Remove common prefixes and simplify
        parts = name.split(".")
        if len(parts) > 3:
            # Keep component, model, and last 2 parts
            return f"{parts[0]}...{'.'.join(parts[-2:])}"
        return name

    def _shorten_type(self, module_type: str) -> str:
        """Shorten module type name."""
        # Remove common prefixes
        for prefix in ["Qwen3OmniMoe", "Qwen3Omni", "Qwen3", "Qwen2"]:
            if module_type.startswith(prefix):
                module_type = module_type[len(prefix):]
        return module_type[:20] if len(module_type) > 20 else module_type

    def _generate_mermaid_for_list(
        self,
        info_list: list[LayerInfo],
        phase_name: str,
        node_prefix: str,
        include_details: bool = True,
        max_edges: int = 499,
        max_nodes: int = 200,
        compact: bool = False,
    ) -> tuple[list[str], int, int]:
        """
        Generate mermaid lines for a list of layer info.

        Returns:
            tuple: (lines, edge_count, node_count)
        """
        lines = []
        edge_count = 0
        node_count = 0

        if not info_list:
            return lines, edge_count, node_count

        lines.append(f"    subgraph {node_prefix}Phase [{phase_name}]")
        prev_node = None

        for info in info_list:
            if edge_count >= max_edges or node_count >= max_nodes:
                lines.append(f'        {node_prefix}_truncated["... truncated (max {max_nodes} nodes / {max_edges} edges)"]')
                if prev_node:
                    lines.append(f"        {prev_node} --> {node_prefix}_truncated")
                break

            node_id = f"{node_prefix}_{self._sanitize_node_id(info.name)}"

            if compact:
                # Compact mode: shorter labels
                short_name = self._shorten_name(info.name)
                short_type = self._shorten_type(info.module_type)
                if include_details and info.input_shapes and info.output_shapes:
                    in_shape = self._format_shape(info.input_shapes[0]) if info.input_shapes else "?"
                    out_shape = self._format_shape(info.output_shapes[0]) if info.output_shapes else "?"
                    label = f'{short_name}\\n{short_type}\\n{in_shape}->{out_shape}'
                else:
                    label = f"{short_name}\\n{short_type}"
            else:
                # Full mode
                if include_details and info.input_shapes and info.output_shapes:
                    in_shape = self._format_shape(info.input_shapes[0]) if info.input_shapes else "?"
                    out_shape = self._format_shape(info.output_shapes[0]) if info.output_shapes else "?"
                    in_dtype = self._format_dtype(info.input_dtypes[0]) if info.input_dtypes else "?"
                    label = f'{info.name}\\nType: {info.module_type}\\nIn: {in_shape} {in_dtype}\\nOut: {out_shape}'
                else:
                    label = f"{info.name}\\n{info.module_type}"

            lines.append(f'        {node_id}["{label}"]')
            node_count += 1
            if prev_node:
                lines.append(f"        {prev_node} --> {node_id}")
                edge_count += 1
            prev_node = node_id

        lines.append("    end")
        return lines, edge_count, node_count

    def generate_mermaid(
        self,
        include_details: bool = True,
        component: str | None = None,
        max_edges: int = 499,
        max_nodes: int = 200,
        compact: bool = False,
        max_layer_depth: int | None = None,
    ) -> str:
        """
        Generate a mermaid flowchart showing model structure and data flow.

        Args:
            include_details: Whether to include shape/dtype details in nodes
            component: Filter by component name (e.g., "thinker", "talker", "code2wav").
                       If None, includes all components.
            max_edges: Maximum number of edges in the diagram (default: 499)
            max_nodes: Maximum number of nodes in the diagram (default: 200)
            compact: Use compact labels (shorter names/types)
            max_layer_depth: Maximum layer nesting depth (None = no limit)

        Returns:
            Mermaid diagram string
        """
        lines = ["flowchart TD"]
        total_edges = 0
        total_nodes = 0

        # Filter by component and depth
        prefill_filtered = self._filter_by_component(self.prefill_info, component)
        prefill_filtered = self._filter_by_depth(prefill_filtered, max_layer_depth)
        decode_filtered = self._filter_by_component(self.decode_info, component)
        decode_filtered = self._filter_by_depth(decode_filtered, max_layer_depth)

        # Generate Prefill Phase subgraph
        if prefill_filtered:
            prefill_lines, prefill_edges, prefill_nodes = self._generate_mermaid_for_list(
                prefill_filtered,
                "Prefill Phase",
                "P",
                include_details,
                max_edges - total_edges,
                max_nodes - total_nodes,
                compact,
            )
            lines.extend(prefill_lines)
            total_edges += prefill_edges
            total_nodes += prefill_nodes

        # Generate Decode Phase subgraph
        if decode_filtered and total_edges < max_edges and total_nodes < max_nodes:
            decode_lines, decode_edges, decode_nodes = self._generate_mermaid_for_list(
                decode_filtered,
                "Decode Phase - Step 1",
                "D",
                include_details,
                max_edges - total_edges,
                max_nodes - total_nodes,
                compact,
            )
            lines.extend(decode_lines)
            total_edges += decode_edges
            total_nodes += decode_nodes

        # Connect phases
        if prefill_filtered and decode_filtered and total_edges < max_edges:
            lines.append("    PPhase --> DPhase")
            total_edges += 1

        return "\n".join(lines)

    def generate_mermaid_by_component(
        self,
        include_details: bool = True,
        max_edges: int = 499,
        max_nodes: int = 200,
        compact: bool = False,
        max_layer_depth: int | None = None,
    ) -> dict[str, str]:
        """
        Generate separate mermaid diagrams for each component.

        Args:
            include_details: Whether to include shape/dtype details in nodes
            max_edges: Maximum number of edges per diagram (default: 499)
            max_nodes: Maximum number of nodes per diagram (default: 200)
            compact: Use compact labels (shorter names/types)
            max_layer_depth: Maximum layer nesting depth (None = no limit)

        Returns:
            Dict mapping component name to mermaid diagram string
        """
        # Identify components from layer names
        components = set()
        for info in self.prefill_info + self.decode_info:
            # Extract top-level component (thinker, talker, code2wav)
            parts = info.name.split(".")
            if parts:
                components.add(parts[0])

        result = {}
        for component in sorted(components):
            mermaid = self.generate_mermaid(
                include_details=include_details,
                component=component,
                max_edges=max_edges,
                max_nodes=max_nodes,
                compact=compact,
                max_layer_depth=max_layer_depth,
            )
            result[component] = mermaid

        return result

    def generate_layer_table(self) -> str:
        """Generate a markdown table with layer details."""
        lines = []

        def _add_table(phase_name: str, info_list: list[LayerInfo]):
            if not info_list:
                return
            lines.append(f"\n## {phase_name}\n")
            lines.append("| # | Layer Name | Type | Input Shape | Input Dtype | Output Shape | Output Dtype |")
            lines.append("|---|------------|------|-------------|-------------|--------------|--------------|")
            for i, info in enumerate(info_list):
                in_shapes = ", ".join(self._format_shape(s) for s in info.input_shapes) or "-"
                in_dtypes = ", ".join(self._format_dtype(d) for d in info.input_dtypes) or "-"
                out_shapes = ", ".join(self._format_shape(s) for s in info.output_shapes) or "-"
                out_dtypes = ", ".join(self._format_dtype(d) for d in info.output_dtypes) or "-"
                lines.append(f"| {i+1} | {info.name} | {info.module_type} | {in_shapes} | {in_dtypes} | {out_shapes} | {out_dtypes} |")

        _add_table("Prefill Phase", self.prefill_info)
        _add_table("Decode Phase (Step 1)", self.decode_info)

        return "\n".join(lines)

    def get_summary(self) -> dict:
        """Get a summary of traced layers."""
        return {
            "prefill_layers": len(self.prefill_info),
            "decode_layers": len(self.decode_info),
            "total_hooks": len(self.hooks),
        }


# ==============================================================================
# Qwen3OmniMoeForConditionalGenerationWithTracing: Model with layer tracing
# ==============================================================================


class Qwen3OmniMoeForConditionalGenerationWithTracing(Qwen3OmniMoeForConditionalGeneration):
    """
    Qwen3 Omni MoE model with layer tracing capabilities.

    This class wraps the original model and adds hooks to trace:
    - Model structure
    - Layer dependencies
    - Input/Output shapes and dtypes for each layer
    - Distinguishes between prefill and decode phases
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.tracer: LayerTracer | None = None
        self._tracing_enabled: bool = False

    def enable_tracing(
        self,
        max_depth: int = 4,
        include_patterns: list[str] | None = None,
    ):
        """
        Enable layer tracing.

        Args:
            max_depth: Maximum module nesting depth to trace
            include_patterns: List of regex patterns for module names to include.
                              Useful patterns:
                              - "thinker" - trace thinker components
                              - "talker" - trace talker components
                              - "code2wav" - trace code2wav components
                              - "embed|attn|mlp|norm" - trace specific layer types
        """
        if self.tracer is not None:
            self.tracer.clear()
        self.tracer = LayerTracer(max_depth=max_depth, include_patterns=include_patterns)
        self.tracer.register_hooks(self)
        self._tracing_enabled = True
        print(f"[Tracing] Enabled with max_depth={max_depth}")

    def disable_tracing(self):
        """Disable layer tracing and clear hooks."""
        if self.tracer is not None:
            self.tracer.clear()
            self.tracer = None
        self._tracing_enabled = False
        print("[Tracing] Disabled")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        speaker: str = "Ethan",
        use_audio_in_video: bool = False,
        return_audio: bool | None = None,
        thinker_max_new_tokens: int = 1024,
        thinker_eos_token_id: int = 151645,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 50,
        talker_top_p: float = 1.0,
        talker_temperature: float = 0.9,
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        """
        Generate with tracing support.

        This method wraps the parent generate() to track prefill and decode phases.
        """
        if not self._tracing_enabled or self.tracer is None:
            # No tracing - use parent generate directly
            return super().generate(
                input_ids=input_ids,
                speaker=speaker,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
                thinker_max_new_tokens=thinker_max_new_tokens,
                thinker_eos_token_id=thinker_eos_token_id,
                talker_max_new_tokens=talker_max_new_tokens,
                talker_do_sample=talker_do_sample,
                talker_top_k=talker_top_k,
                talker_top_p=talker_top_p,
                talker_temperature=talker_temperature,
                talker_repetition_penalty=talker_repetition_penalty,
                **kwargs,
            )

        # Reset tracer state for new generation
        self.tracer.prefill_info.clear()
        self.tracer.decode_info.clear()
        self.tracer.current_phase = "prefill"
        self.tracer.decode_recorded = False
        self.tracer.call_counter = 0

        # Store original generate method behavior
        # The parent class handles the AR loop internally
        # We hook into the forward passes to capture prefill vs decode

        # For Qwen3 Omni, the generate flow is:
        # 1. Thinker generates (prefill + decode loop internally)
        # 2. Talker generates (prefill + decode loop internally)
        # 3. Code2Wav decodes

        # We'll capture whatever the hooks record during the full generate
        print("[Tracing] Starting traced generation (prefill phase)...")

        # Run the actual generation
        result = super().generate(
            input_ids=input_ids,
            speaker=speaker,
            use_audio_in_video=use_audio_in_video,
            return_audio=return_audio,
            thinker_max_new_tokens=thinker_max_new_tokens,
            thinker_eos_token_id=thinker_eos_token_id,
            talker_max_new_tokens=talker_max_new_tokens,
            talker_do_sample=talker_do_sample,
            talker_top_k=talker_top_k,
            talker_top_p=talker_top_p,
            talker_temperature=talker_temperature,
            talker_repetition_penalty=talker_repetition_penalty,
            **kwargs,
        )

        summary = self.tracer.get_summary()
        print(f"[Tracing] Generation complete. Captured {summary['prefill_layers']} layer calls.")

        return result

    def get_mermaid_diagram(self, include_details: bool = True) -> str:
        """
        Get the mermaid diagram of the traced model structure.

        Args:
            include_details: Whether to include shape/dtype details

        Returns:
            Mermaid diagram string
        """
        if self.tracer is None:
            return "# No tracing data available. Call enable_tracing() and run generate() first."
        return self.tracer.generate_mermaid(include_details=include_details)

    def get_layer_table(self) -> str:
        """
        Get a markdown table with layer details.

        Returns:
            Markdown table string
        """
        if self.tracer is None:
            return "# No tracing data available. Call enable_tracing() and run generate() first."
        return self.tracer.generate_layer_table()

    def save_tracing_results(
        self,
        output_dir: str,
        prefix: str = "model_structure",
        max_edges: int = 499,
        max_nodes: int = 200,
        compact: bool = True,
        max_layer_depth: int | None = 3,
        split_by_component: bool = True,
    ):
        """
        Save tracing results to files.

        Args:
            output_dir: Directory to save results
            prefix: Filename prefix
            max_edges: Maximum edges per mermaid diagram (default: 499)
            max_nodes: Maximum nodes per mermaid diagram (default: 200)
            compact: Use compact labels (default: True)
            max_layer_depth: Maximum layer depth to show (default: 3)
            split_by_component: If True, save separate mermaid files for thinker/talker/code2wav
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.tracer is None:
            print("[Tracing] No tracing data to save.")
            return

        if split_by_component:
            # Save separate mermaid diagrams for each component
            component_diagrams = self.tracer.generate_mermaid_by_component(
                include_details=True,
                max_edges=max_edges,
                max_nodes=max_nodes,
                compact=compact,
                max_layer_depth=max_layer_depth,
            )
            for component, mermaid in component_diagrams.items():
                mermaid_path = os.path.join(output_dir, f"{prefix}_{component}_mermaid.md")
                with open(mermaid_path, "w", encoding="utf-8") as f:
                    f.write(f"# Qwen3 Omni Model Structure - {component.capitalize()}\n\n")
                    f.write("```mermaid\n")
                    f.write(mermaid)
                    f.write("\n```\n")
                print(f"[Tracing] Saved {component} mermaid diagram to {mermaid_path}")
        else:
            # Save combined mermaid diagram
            mermaid_path = os.path.join(output_dir, f"{prefix}_mermaid.md")
            with open(mermaid_path, "w", encoding="utf-8") as f:
                f.write("# Qwen3 Omni Model Structure\n\n")
                f.write("```mermaid\n")
                f.write(self.tracer.generate_mermaid(
                    include_details=True,
                    max_edges=max_edges,
                    max_nodes=max_nodes,
                    compact=compact,
                    max_layer_depth=max_layer_depth,
                ))
                f.write("\n```\n")
            print(f"[Tracing] Saved mermaid diagram to {mermaid_path}")

        # Save layer table
        table_path = os.path.join(output_dir, f"{prefix}_layers.md")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("# Qwen3 Omni Layer Details\n")
            f.write(self.get_layer_table())
        print(f"[Tracing] Saved layer table to {table_path}")

        # Save JSON data
        json_path = os.path.join(output_dir, f"{prefix}_data.json")
        data = {
            "summary": self.tracer.get_summary(),
            "prefill_layers": [
                {
                    "name": info.name,
                    "type": info.module_type,
                    "input_shapes": [list(s) for s in info.input_shapes],
                    "input_dtypes": info.input_dtypes,
                    "output_shapes": [list(s) for s in info.output_shapes],
                    "output_dtypes": info.output_dtypes,
                    "call_order": info.call_order,
                }
                for info in self.tracer.prefill_info
            ],
            "decode_layers": [
                {
                    "name": info.name,
                    "type": info.module_type,
                    "input_shapes": [list(s) for s in info.input_shapes],
                    "input_dtypes": info.input_dtypes,
                    "output_shapes": [list(s) for s in info.output_shapes],
                    "output_dtypes": info.output_dtypes,
                    "call_order": info.call_order,
                }
                for info in self.tracer.decode_info
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Tracing] Saved JSON data to {json_path}")


# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"


def load_prompts(prompts_file: str) -> list[str]:
    """Load prompts from a text file, one prompt per line."""
    prompts = []
    with open(prompts_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def run_benchmark(
    model,
    processor,
    prompts: list[str],
    output_dir: str = "benchmark_results",
    speaker: str = "Ethan",
    use_audio_in_video: bool = True,
):
    """
    Run benchmark on a list of prompts and collect performance stats.

    Args:
        model: The Qwen3OmniMoe model
        processor: The Qwen3OmniMoe processor
        prompts: List of text prompts to process
        output_dir: Directory to save results
        speaker: Speaker voice for audio output
        use_audio_in_video: Whether to use audio in video

    Returns:
        tuple: (aggregated_stats, results, audio_outputs)
            - aggregated_stats: dict with aggregated performance statistics
            - results: list of dicts with per-prompt results
            - audio_outputs: list of audio tensors/arrays (or None if no audio)
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    all_stats = []
    results = []
    audio_outputs = []

    for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        # Preparation for inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
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

        # Inference: Generation of the output text and audio
        text_ids, audio = model.generate(
            **inputs, speaker=speaker, thinker_return_dict_in_generate=True, use_audio_in_video=use_audio_in_video
        )

        # Decode output text
        output_text = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Collect performance stats
        perf_stats = None
        if hasattr(model, "_perf_stats_last"):
            perf_stats = model._perf_stats_last.copy()
            perf_stats["prompt_idx"] = idx
            perf_stats["prompt"] = prompt
            all_stats.append(perf_stats)

        # Save audio and collect audio output
        audio_path = None
        audio_data = None
        if audio is not None:
            audio_data = audio.reshape(-1).detach().cpu().numpy()
            audio_path = os.path.join(audio_dir, f"output_{idx:04d}.wav")
            sf.write(
                audio_path,
                audio_data,
                samplerate=24000,
            )
            audio_outputs.append(audio_data)
        else:
            audio_outputs.append(None)

        # Save result
        result = {
            "idx": idx,
            "prompt": prompt,
            "output": output_text,
            "audio_path": audio_path,
            "perf_stats": perf_stats,
        }
        results.append(result)

    # Aggregate statistics
    aggregated_stats = aggregate_stats(all_stats)

    # Save all results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save aggregated stats
    stats_path = os.path.join(output_dir, "perf_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"aggregated": aggregated_stats, "per_prompt": all_stats}, f, ensure_ascii=False, indent=2)

    # Count saved audio files
    num_audio_saved = sum(1 for a in audio_outputs if a is not None)
    print(f"\nSaved {num_audio_saved} audio files to {audio_dir}/")

    return aggregated_stats, results, audio_outputs


def aggregate_stats(all_stats: list[dict]) -> dict:
    """Aggregate performance statistics from multiple runs."""
    if not all_stats:
        return {}

    keys = [
        "thinker_tokens",
        "thinker_time_s",
        "thinker_tps",
        "talker_tokens",
        "talker_time_s",
        "talker_tps",
        "code2wav_tokens",
        "code2wav_time_s",
        "code2wav_tps",
        "total_tokens",
        "total_time_s",
        "total_tps",
    ]

    aggregated = {
        "num_samples": len(all_stats),
    }

    for key in keys:
        values = [s.get(key, 0) for s in all_stats if key in s]
        if values:
            aggregated[f"{key}_sum"] = sum(values)
            aggregated[f"{key}_avg"] = sum(values) / len(values)
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)

    # Calculate overall throughput
    total_tokens = aggregated.get("total_tokens_sum", 0)
    total_time = aggregated.get("total_time_s_sum", 0)
    if total_time > 0:
        aggregated["overall_tps"] = total_tokens / total_time

    return aggregated


def print_stats(stats: dict):
    """Print performance statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("Performance Statistics Summary")
    print("=" * 60)

    print(f"\nNumber of samples: {stats.get('num_samples', 0)}")

    print("\n--- Thinker ---")
    print(f"  Total tokens:  {stats.get('thinker_tokens_sum', 0):.0f}")
    print(f"  Total time:    {stats.get('thinker_time_s_sum', 0):.2f}s")
    print(f"  Avg TPS:       {stats.get('thinker_tps_avg', 0):.2f}")
    print(f"  Min TPS:       {stats.get('thinker_tps_min', 0):.2f}")
    print(f"  Max TPS:       {stats.get('thinker_tps_max', 0):.2f}")

    print("\n--- Talker ---")
    print(f"  Total tokens:  {stats.get('talker_tokens_sum', 0):.0f}")
    print(f"  Total time:    {stats.get('talker_time_s_sum', 0):.2f}s")
    print(f"  Avg TPS:       {stats.get('talker_tps_avg', 0):.2f}")
    print(f"  Min TPS:       {stats.get('talker_tps_min', 0):.2f}")
    print(f"  Max TPS:       {stats.get('talker_tps_max', 0):.2f}")

    print("\n--- Code2Wav ---")
    print(f"  Total tokens:  {stats.get('code2wav_tokens_sum', 0):.0f}")
    print(f"  Total time:    {stats.get('code2wav_time_s_sum', 0):.2f}s")
    print(f"  Avg TPS:       {stats.get('code2wav_tps_avg', 0):.2f}")
    print(f"  Min TPS:       {stats.get('code2wav_tps_min', 0):.2f}")
    print(f"  Max TPS:       {stats.get('code2wav_tps_max', 0):.2f}")

    print("\n--- Overall ---")
    print(f"  Total tokens:  {stats.get('total_tokens_sum', 0):.0f}")
    print(f"  Total time:    {stats.get('total_time_s_sum', 0):.2f}s")
    print(f"  Overall TPS:   {stats.get('overall_tps', 0):.2f}")
    print(f"  Avg TPS:       {stats.get('total_tps_avg', 0):.2f}")
    print(f"  Min TPS:       {stats.get('total_tps_min', 0):.2f}")
    print(f"  Max TPS:       {stats.get('total_tps_max', 0):.2f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Omni Benchmark Script")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="benchmark/build_dataset/top100.txt",
        help="Path to the prompts file (one prompt per line)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="benchmark_results", help="Directory to save benchmark results"
    )
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model")
    parser.add_argument("--speaker", type=str, default="Ethan", help="Speaker voice for audio output")
    parser.add_argument("--num_prompts", type=int, default=None, help="Number of prompts to process (default: all)")
    args = parser.parse_args()

    # Load model and processor
    print(f"Loading model from {args.model_path}...")
    model = Qwen3OmniMoeForConditionalGenerationWithLogging.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    # Benchmark mode
    print(f"Loading prompts from {args.prompts_file}...")
    prompts = load_prompts(args.prompts_file)

    if args.num_prompts:
        prompts = prompts[: args.num_prompts]

    print(f"Running benchmark on {len(prompts)} prompts...")

    aggregated_stats, results, audio_outputs = run_benchmark(
        model=model,
        processor=processor,
        prompts=prompts,
        output_dir=args.output_dir,
        speaker=args.speaker,
    )

    print_stats(aggregated_stats)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()


__all__ = [
    "LayerInfo",
    "LayerTracer",
    "Qwen3OmniMoeForConditionalGenerationWithTracing",
    "load_prompts",
    "run_benchmark",
    "aggregate_stats",
    "print_stats",
    "main",
]
