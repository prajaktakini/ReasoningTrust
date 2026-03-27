"""
Comparative Linear Probe for Thinking vs Instruct Models

Updated to use Ridge Regression with Pearson Correlation (following Aligned Probing paper)
Instead of Logistic Regression with AUC.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from reasoning_trust.models.initialize_model import load_model_config
except ImportError:
    def load_model_config(model_name):
        return {"model_string": model_name, "tokenizer_string": model_name}


class HybridActivationExtractor:
    """
    Extract activations from both thinking and instruct modes for hybrid analysis.
    
    Uses the same model configuration as vLLM setup but loads with transformers
    for activation extraction (vLLM doesn't support hooks).
    """
    
    def __init__(self, model_name: str = None, device: str = "cuda"):
        """Initialize extractor using model_name.""" 
        self.device = device
        self.model_name = model_name

        # Load Model Config
        try:
            self.model_config = load_model_config(model_name)
            model_string = self.model_config.get("model_string", model_name)
            tokenizer_string = self.model_config.get("tokenizer_string", model_string)
        except:
            model_string = model_name
            tokenizer_string = model_name
        
        print(f"\n{'='*80}")
        print(f"Loading model: {model_string}")
        print(f"{'='*80}")
        
        # Load tokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_string, trust_remote_code=True)

        # Load Model
        print("Loading model for activation extraction...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_string,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.hf_model.eval()
        print("✓ Model loaded successfully\n")
        
        self.activations = {}
        self.hooks = []

    def _get_activation(self, name):
        """Hook to capture activations from specified layers."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if len(hidden.shape) == 3:
                self.activations[f"{name}"] = hidden[:, -1, :].detach().cpu().numpy()
        
        return hook

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """Register hooks to capture activations from specified layers."""
        self.activations = {}
        self.hooks = []

        # Find transformer layers
        if hasattr(self.hf_model, 'model') and hasattr(self.hf_model.model, 'layers'):
            layers = self.hf_model.model.layers
        elif hasattr(self.hf_model, 'transformer') and hasattr(self.hf_model.transformer, 'h'):
            layers = self.hf_model.transformer.h
        else:
            raise ValueError("Could not find transformer layers in model")
        
        num_layers = len(layers)
        if layer_indices is None:
            layer_indices = list(range(num_layers))

        for idx in layer_indices:
            if 0 <= idx < num_layers:
                layer = layers[idx]
                hook = layer.register_forward_hook(self._get_activation(f"layer_{idx}"))
                self.hooks.append(hook)

        return num_layers

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_from_full_text(
        self, 
        full_text: str, 
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """Extract activations from full text (prompt + continuation)."""

        self.register_hooks(layer_indices)
        self.activations = {}

        inputs = self.hf_tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)

        with torch.no_grad():
            _ = self.hf_model(**inputs)

        all_activations = {}
        for layer_name, activation in self.activations.items():
            layer_idx = int(layer_name.split("_")[1])
            all_activations[layer_idx] = activation

        self.remove_hooks()
        del inputs
        torch.cuda.empty_cache()
        
        return all_activations

    def extract_batch(
        self,
        full_texts: List[str],
        layer_indices: Optional[List[int]] = None,
        batch_size: int = 4,
    ) -> Dict[int, np.ndarray]:
        """Extract activations from multiple full texts"""
        all_layer_activations = {idx: [] for idx in (layer_indices if layer_indices else [])}

        for i in tqdm(range(0, len(full_texts), batch_size), desc="Extracting activations"):
            batch_texts = full_texts[i:i+batch_size]

            for text in batch_texts:
                try:
                    activations = self.extract_from_full_text(
                        text, layer_indices=layer_indices
                    )

                    for layer_idx, act in activations.items():
                        if layer_idx not in all_layer_activations:
                            all_layer_activations[layer_idx] = []
                        all_layer_activations[layer_idx].append(act[0])
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return {
            idx: np.vstack(acts) 
            for idx, acts in all_layer_activations.items() 
            if acts
        }
    
    def extract_output_components(
        self,
        full_texts: List[str],
        component: str = 'layer_norm',
        batch_size: int = 4,
    ) -> np.ndarray:
        """Extract activations from output components (layer_norm or lm_head)"""
        
        activations = []

        if component == 'layer_norm':
            target_module = self.hf_model.model.norm
        elif component == 'logits':
            target_module = self.hf_model.lm_head
        else:
            raise ValueError(f"Unknown component: {component}")

        def hook(module, input, output):
            if len(output.shape) == 3:
                activations.append(output[:, -1, :].detach().cpu().numpy())
            
        handle = target_module.register_forward_hook(hook)

        for i in tqdm(range(0, len(full_texts), batch_size), desc=f"Extracting {component}"):
            batch_texts = full_texts[i:i+batch_size]

            for text in batch_texts:
                try:
                    inputs = self.hf_tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096
                    ).to(self.device)

                    with torch.no_grad():
                        _ = self.hf_model(**inputs)

                    del inputs

                except Exception as e:
                    print(f"Error: {e}")
                    continue

            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        handle.remove()

        if len(activations) == 0:
            return np.array([])
        
        return np.vstack(activations)


def load_data_from_jsonl(
    jsonl_path: str, 
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[str], List[str], np.ndarray, Dict[str, np.ndarray]]:
    """
    Load data from JSONL with CONTINUOUS toxicity scores.
    
    Returns:
        prompts: Original prompts
        continuations: Generated continuations
        full_texts: prompt + continuation
        toxicity_scores: Continuous toxicity scores [num_samples]
        all_attributes: Dict of all toxicity attributes
    """

    prompts = []
    continuations = []
    full_texts = []
    toxicity_scores = []
    
    # Track all Perspective API attributes
    attributes = {
        'toxicity': [],
        'severe_toxicity': [],
        'identity_attack': [],
        'insult': [],
        'profanity': [],
        'threat': [],
        'sexually_explicit': []
    }

    print(f"Loading data from {jsonl_path}...")

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                continuation = data.get('continuation', '')
                content_score = data.get('content_score', {})

                if not content_score or not isinstance(content_score, dict):
                    continue

                # Get CONTINUOUS toxicity score (not binary!)
                toxicity = content_score.get('toxicity', 0.0)

                full_text = prompt + "\n" + continuation
                prompts.append(prompt)
                continuations.append(continuation)
                full_texts.append(full_text)
                toxicity_scores.append(toxicity)
                
                # Store all attributes
                for attr in attributes.keys():
                    attributes[attr].append(content_score.get(attr, 0.0))

            except Exception as e:
                continue

    toxicity_scores = np.array(toxicity_scores)
    
    # Convert attributes to numpy
    for attr in attributes:
        attributes[attr] = np.array(attributes[attr])

    print(f"  Loaded: {len(full_texts)} samples")
    print(f"  Mean toxicity: {toxicity_scores.mean():.3f} ± {toxicity_scores.std():.3f}")
    print(f"  Median toxicity: {np.median(toxicity_scores):.3f}")
    print(f"  Toxic (>0.5): {(toxicity_scores > 0.5).sum()} ({(toxicity_scores > 0.5).mean():.1%})")

    return prompts, continuations, full_texts, toxicity_scores, attributes


class ComparativeLinearProbe:
    """
    Train and compare linear probes using Ridge Regression.
    Uses Pearson correlation as metric (instead of AUC).
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: L2 regularization strength (1.0 = moderate, 0.1 = strong)
        """
        self.alpha = alpha
        self.probes_thinking = {}
        self.probes_instruct = {}
        self.results_thinking = {}
        self.results_instruct = {}
        self.differences = {}
        
    def train_probe(
        self, 
        activations: np.ndarray,
        toxicity_scores: np.ndarray,
        layer_idx: int,
        mode: str,
        test_size: float = 0.2
    ):
        """
        Train Ridge regression probe for continuous toxicity prediction.
        
        Args:
            activations: Hidden States [num_samples, hidden_dim]
            toxicity_scores: Continuous toxicity scores [num_samples] (0.0-1.0)
            layer_idx: Layer index 
            mode: "thinking" or "instruct"
        """

        if len(activations) < 10:
            print(f"Layer {layer_idx} ({mode}): Too few samples, skipping")
            return None
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                activations, toxicity_scores, 
                test_size=test_size, 
                random_state=42
            )
        except ValueError as e:
            print(f"Layer {layer_idx} ({mode}): Split failed: {e}")
            return None
        
        # Train Ridge Regression probe (linear regression with L2 regularization)
        probe = Ridge(alpha=self.alpha, random_state=42, max_iter=1000)
        probe.fit(X_train, y_train)

        # Predict on test set
        y_pred = probe.predict(X_test)

        # Compute Pearson correlation (main metric - replaces AUC)
        try:
            correlation, p_value = pearsonr(y_test, y_pred)
        except:
            correlation, p_value = 0.0, 1.0

        # Additional metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Also compute train correlation for overfitting check
        y_train_pred = probe.predict(X_train)
        train_correlation, _ = pearsonr(y_train, y_train_pred)

        results = {
            'correlation': correlation,  # ← Main metric (replaces AUC)
            'p_value': p_value,
            'mse': mse,
            'r2': r2,
            'train_correlation': train_correlation,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mean_toxicity': y_train.mean(),
            'std_toxicity': y_train.std(),
            'layer_idx': layer_idx
        }

        # Store
        if mode == "thinking":
            self.probes_thinking[layer_idx] = probe
            self.results_thinking[layer_idx] = results
        else:
            self.probes_instruct[layer_idx] = probe
            self.results_instruct[layer_idx] = results

        # Print with overfitting check
        overfit_gap = train_correlation - correlation
        overfit_marker = ""
        if overfit_gap > 0.2:
            overfit_marker = " OVERFIT"
        elif overfit_gap > 0.1:
            overfit_marker = " MILD-OVERFIT"
    
        print(f"Layer {layer_idx:2d} ({mode:9s}): "
            f"Test r={correlation:.3f}, Train r={train_correlation:.3f}, "
            f"Gap={overfit_gap:.3f}, p={p_value:.4f}{overfit_marker}")

        return results

    def compare_modes(self):
        """Compare probe performance between thinking and instruct modes."""
        all_layers = set(self.results_thinking.keys()) | set(self.results_instruct.keys())

        for layer_idx in all_layers:
            thinking_corr = self.results_thinking.get(layer_idx, {}).get('correlation', 0)
            instruct_corr = self.results_instruct.get(layer_idx, {}).get('correlation', 0)

            thinking_overfit = (self.results_thinking.get(layer_idx, {}).get('train_correlation', 0) - 
                           thinking_corr)
            instruct_overfit = (self.results_instruct.get(layer_idx, {}).get('train_correlation', 0) - 
                            instruct_corr)

            self.differences[layer_idx] = {
                'thinking_correlation': thinking_corr,
                'instruct_correlation': instruct_corr,
                'thinking_overfit': thinking_overfit,  
                'instruct_overfit': instruct_overfit,  
                'difference': thinking_corr - instruct_corr,
                'relative_diff_pct': (thinking_corr - instruct_corr) / max(abs(instruct_corr), 0.001) * 100
            }

    def analyze_emergence_and_suppression(self, threshold=0.5):
        """
        Analyze where toxicity emerges or is suppressed.
        Uses correlation threshold instead of AUC threshold.
        """
        thinking_emergence = [
            layer for layer, results in self.results_thinking.items()
            if results.get('correlation', 0) > threshold
        ]

        instruct_emergence = [
            layer for layer, results in self.results_instruct.items()
            if results.get('correlation', 0) > threshold
        ]

        thinking_better = [
            layer for layer, diff in self.differences.items()
            if diff.get('difference', 0) > 0.1  # Correlation difference threshold
        ]

        def find_suppression_layers(results_dict):
            drops = []
            sorted_layers = sorted(results_dict.keys())
            for i in range(1, len(sorted_layers)):
                prev_layer = sorted_layers[i-1]
                curr_layer = sorted_layers[i]
                prev_corr = results_dict[prev_layer].get('correlation', 0)
                curr_corr = results_dict[curr_layer].get('correlation', 0)
                if curr_corr < prev_corr - 0.1:  # Correlation drop threshold
                    drops.append(curr_layer)
            return drops
        
        thinking_suppression = find_suppression_layers(self.results_thinking)
        instruct_suppression = find_suppression_layers(self.results_instruct)

        return {
            'thinking_emergence_layers': sorted(thinking_emergence),
            'instruct_emergence_layers': sorted(instruct_emergence),
            'thinking_better_layers': sorted(thinking_better),
            'thinking_suppression_layers': sorted(thinking_suppression),
            'instruct_suppression_layers': sorted(instruct_suppression),
        }

    def visualize_comparison(self, output_dir: Path):
        """Create visualization comparing modes (using correlation)"""

        output_dir.mkdir(parents=True, exist_ok=True)

        all_layers = sorted(set(self.results_thinking.keys()) | set(self.results_instruct.keys()))
        thinking_corrs = [self.results_thinking.get(l, {}).get('correlation', 0) for l in all_layers]
        instruct_corrs = [self.results_instruct.get(l, {}).get('correlation', 0) for l in all_layers]

        thinking_gaps = [self.results_thinking.get(l, {}).get('train_correlation', 0) - 
                     self.results_thinking.get(l, {}).get('correlation', 0) for l in all_layers]
        instruct_gaps = [self.results_instruct.get(l, {}).get('train_correlation', 0) - 
                        self.results_instruct.get(l, {}).get('correlation', 0) for l in all_layers]


        fig, axes = plt.subplots(3, 1, figsize=(14, 14))

        # Plot 1: Layer-wise comparison
        ax1 = axes[0]
        ax1.plot(all_layers, thinking_corrs, marker='o', label='Thinking Mode', 
                linewidth=2.5, markersize=7, color='#d62728', alpha=0.8)
        ax1.plot(all_layers, instruct_corrs, marker='s', label='Instruct Mode',
                linewidth=2.5, markersize=7, color='#1f77b4', alpha=0.8)
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Moderate (r=0.5)')
        ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Strong (r=0.7)')
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.3, linewidth=1.5)
        ax1.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Pearson Correlation (Toxicity Prediction)', fontsize=13, fontweight='bold')
        ax1.set_title('Layer-wise Toxicity Encoding: Thinking vs Instruct Models', 
                     fontsize=15, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([-0.1, 1.0])

        # Plot 2: Difference
        ax2 = axes[1]
        differences = [self.differences.get(l, {}).get('difference', 0) for l in all_layers]
        colors = ['#d62728' if d > 0 else '#1f77b4' for d in differences]
        ax2.bar(all_layers, differences, alpha=0.7, color=colors, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax2.axhline(y=0.1, color='#d62728', linestyle='--', alpha=0.5, linewidth=1.5, label='Significant (+0.1)')
        ax2.axhline(y=-0.1, color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5, label='Significant (-0.1)')
        ax2.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Correlation Difference (Thinking - Instruct)', fontsize=13, fontweight='bold')
        ax2.set_title('Where Thinking Models Show Stronger Toxicity Encoding', 
                     fontsize=15, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        ax3 = axes[2]
        ax3.plot(all_layers, thinking_gaps, marker='o', label='Thinking Overfit Gap',
                linewidth=2.5, markersize=7, color='#d62728', alpha=0.8)
        ax3.plot(all_layers, instruct_gaps, marker='s', label='Instruct Overfit Gap',
                linewidth=2.5, markersize=7, color='#1f77b4', alpha=0.8)
        ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Mild Overfit (0.1)')
        ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Severe Overfit (0.2)')
        ax3.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Train Correlation - Test Correlation', fontsize=13, fontweight='bold')
        ax3.set_title('Overfitting Detection by Layer', fontsize=15, fontweight='bold', pad=15)
        ax3.legend(fontsize=11, loc='best')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim([0, 0.4])
        
        plt.tight_layout()
        plt.savefig(output_dir / "layer_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualizations saved to: {output_dir / 'layer_comparison.png'}")

    def generate_report(self, output_dir: Path) -> pd.DataFrame:
        """Generate comprehensive comparison report."""

        all_layers = sorted(set(self.results_thinking.keys()) | set(self.results_instruct.keys()))

        rows = []
        for layer in all_layers:
            thinking_res = self.results_thinking.get(layer, {})
            instruct_res = self.results_instruct.get(layer, {})
            diff = self.differences.get(layer, {})

            row = {
                'layer': layer,
                'thinking_correlation': thinking_res.get('correlation', 0),
                'thinking_train_correlation': thinking_res.get('train_correlation', 0), 
                'thinking_overfit_gap': thinking_res.get('train_correlation', 0) - thinking_res.get('correlation', 0), 
                'thinking_r2': thinking_res.get('r2', 0),
                'thinking_mse': thinking_res.get('mse', 0),
                'instruct_correlation': instruct_res.get('correlation', 0),
                'instruct_train_correlation': instruct_res.get('train_correlation', 0),  
                'instruct_overfit_gap': instruct_res.get('train_correlation', 0) - instruct_res.get('correlation', 0),  
                'instruct_r2': instruct_res.get('r2', 0),
                'instruct_mse': instruct_res.get('mse', 0),
                'correlation_difference': diff.get('difference', 0),
                'relative_diff_pct': diff.get('relative_diff_pct', 0),    
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        report_path = output_dir / "comparative_report.csv"
        df.to_csv(report_path, index=False)

        # Print Summary
        print("\n" + "="*80)
        print("SUMMARY FINDINGS")
        print("="*80)

        if len(df) > 0:
            best_thinking = df.loc[df['thinking_correlation'].idxmax()]
            best_instruct = df.loc[df['instruct_correlation'].idxmax()]

            print(f"\nBest Thinking Layer: {int(best_thinking['layer'])} (r={best_thinking['thinking_correlation']:.3f})")
            print(f"Best Instruct Layer: {int(best_instruct['layer'])} (r={best_instruct['instruct_correlation']:.3f})")

            print(f"\nOverfitting Analysis:")
            avg_think_gap = df['thinking_overfit_gap'].mean()
            avg_inst_gap = df['instruct_overfit_gap'].mean()
            max_think_gap = df['thinking_overfit_gap'].max()
            max_inst_gap = df['instruct_overfit_gap'].max()
            
            print(f"  Thinking - Avg gap: {avg_think_gap:.3f}, Max gap: {max_think_gap:.3f}")
            print(f"  Instruct - Avg gap: {avg_inst_gap:.3f}, Max gap: {max_inst_gap:.3f}")
            
            if max_think_gap > 0.2 or max_inst_gap > 0.2:
                print(f"  WARNING: Severe overfitting detected!")
            elif avg_think_gap > 0.1 or avg_inst_gap > 0.1:
                print(f" WARNING: Moderate overfitting detected")
            else:
                print(f"  Overfitting is under control")

            significant_diff = df[df['correlation_difference'].abs() > 0.1]
            if len(significant_diff) > 0:
                print(f"\nLayers with significant correlation differences (|Δ| > 0.1):")
                for _, row in significant_diff.iterrows():
                    print(f"  Layer {int(row['layer']):2d}: Δ={row['correlation_difference']:+.3f} "
                          f"({row['relative_diff_pct']:+.1f}%)")
            
            avg_diff = df['correlation_difference'].mean()
            print(f"\nAverage correlation difference: {avg_diff:+.3f} "
                  f"({'Thinking stronger' if avg_diff > 0 else 'Instruct stronger'})")
        
        print(f"\nReport saved to: {report_path}")
        
        return df


def main(
    jsonl_thinking: str,
    jsonl_instruct: str,
    model_name_thinking: str = None,
    model_name_instruct: str = None,
    output_dir: str = "results/linear_probes/comparative",
    max_samples: Optional[int] = None,
    batch_size: int = 4,
    layer_indices: Optional[List[int]] = None,
    template_id: Optional[int] = None,
    probe_output_components: bool = False,
    alpha: float = 1.0,
    attribute: str = 'toxicity'
):
    """Main function for comparative linear probing analysis."""

    output_dir = Path(output_dir)
    if template_id is not None:
        output_dir = output_dir / f"template_{template_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("COMPARATIVE LINEAR PROBING ANALYSIS")
    print("Using Ridge Regression with Pearson Correlation")
    print("="*80)
    print(f"Regularization (alpha): {alpha}")
    print(f"Analyzing attribute: {attribute}")

    # Load data with continuous scores
    prompts_think, conts_think, full_think, tox_think, attrs_think = load_data_from_jsonl(
        jsonl_thinking, max_samples=max_samples
    )

    prompts_inst, conts_inst, full_inst, tox_inst, attrs_inst = load_data_from_jsonl(
        jsonl_instruct, max_samples=max_samples
    )
    
    # Select attribute to analyze
    toxicity_scores_thinking = attrs_think[attribute]
    toxicity_scores_instruct = attrs_inst[attribute]

    # Initialize extractors
    extractor_thinking = HybridActivationExtractor(model_name=model_name_thinking)
    extractor_instruct = HybridActivationExtractor(model_name=model_name_instruct)

    # Extract activations
    print("\n" + "="*80)
    print("EXTRACTING ACTIVATIONS FROM GENERATED OUTPUTS")
    print("="*80)

    print("\nExtracting thinking mode activations...")
    activations_thinking = extractor_thinking.extract_batch(
        full_think,
        layer_indices=layer_indices,
        batch_size=batch_size
    )

    torch.cuda.empty_cache()
    gc.collect()

    print("\nExtracting instruct mode activations...")
    activations_instruct = extractor_instruct.extract_batch(
        full_inst,
        layer_indices=layer_indices,
        batch_size=batch_size
    )

    # Train probes
    print("\n" + "="*80)
    print("TRAINING LINEAR PROBES (Ridge Regression)")
    print("="*80)

    probe = ComparativeLinearProbe(alpha=alpha)

    print("\nTraining probes for thinking mode...")
    for layer_idx in sorted(activations_thinking.keys()):
        probe.train_probe(
            activations=activations_thinking[layer_idx],
            toxicity_scores=toxicity_scores_thinking,
            layer_idx=layer_idx,
            mode="thinking"
        )
    
    print("\nTraining probes for instruct mode...")
    for layer_idx in sorted(activations_instruct.keys()):
        probe.train_probe(
            activations=activations_instruct[layer_idx],
            toxicity_scores=toxicity_scores_instruct,
            layer_idx=layer_idx,
            mode="instruct"
        )
    
    # Optional: Output components
    if probe_output_components:
        print("\n" + "="*80)
        print("PROBING OUTPUT COMPONENTS")
        print("="*80)

        # Layer Norm
        print("\n[1/2] Extracting from final layer norm...")
        ln_thinking = extractor_thinking.extract_output_components(
            full_think, component='layer_norm', batch_size=batch_size
        )
        ln_instruct = extractor_instruct.extract_output_components(
            full_inst, component='layer_norm', batch_size=batch_size
        )

        if len(ln_thinking) > 0 and len(ln_instruct) > 0:
            probe.train_probe(ln_thinking, toxicity_scores_thinking, layer_idx=100, mode="thinking")
            probe.train_probe(ln_instruct, toxicity_scores_instruct, layer_idx=100, mode="instruct")

        # lm_head with PCA
        print("\n[2/2] Extracting from lm_head...")
        try:
            from sklearn.decomposition import PCA

            logits_thinking = extractor_thinking.extract_output_components(
                full_think, component='logits', batch_size=batch_size
            )
            logits_instruct = extractor_instruct.extract_output_components(
                full_inst, component='logits', batch_size=batch_size
            )

            if len(logits_thinking) > 0 and len(logits_instruct) > 0:
                n_samples = len(logits_thinking) + len(logits_instruct)
                n_components = min(1000, n_samples - 10)
                
                print(f"Applying PCA: {logits_thinking.shape[1]} → {n_components} dims...")
                pca = PCA(n_components=n_components, random_state=42)
                
                combined = np.vstack([logits_thinking, logits_instruct])
                pca.fit(combined)
                
                logits_thinking_pca = pca.transform(logits_thinking)
                logits_instruct_pca = pca.transform(logits_instruct)
                
                probe.train_probe(logits_thinking_pca, toxicity_scores_thinking, layer_idx=101, mode="thinking")
                probe.train_probe(logits_instruct_pca, toxicity_scores_instruct, layer_idx=101, mode="instruct")
                
        except Exception as e:
            print(f"lm_head extraction failed: {e}")

    # Analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    probe.compare_modes()
    analysis = probe.analyze_emergence_and_suppression()

    print(f"\nStrong Toxicity Encoding (r > 0.5):")
    print(f"  Thinking: {analysis['thinking_emergence_layers']}")
    print(f"  Instruct: {analysis['instruct_emergence_layers']}")
    
    print(f"\nLayers where Thinking shows significantly stronger encoding:")
    print(f"  {analysis['thinking_better_layers']}")
    
    print(f"\nEncoding Drops (correlation decreases):")
    print(f"  Thinking: {analysis['thinking_suppression_layers']}")
    print(f"  Instruct: {analysis['instruct_suppression_layers']}")

    # Visualize and report
    probe.visualize_comparison(output_dir=output_dir)
    report_df = probe.generate_report(output_dir=output_dir)

    # Save analysis summary
    analysis_path = output_dir / "analysis_summary.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✓ Analysis summary saved to: {analysis_path}")

    print("\n" + "="*80)
    print("COMPARATIVE LINEAR PROBING COMPLETE")
    print("="*80)

    return probe, report_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Comparative Linear Probing using Ridge Regression & Pearson Correlation"
    )

    parser.add_argument("--jsonl-thinking", required=True,
                       help="JSONL file from thinking mode evaluation")
    parser.add_argument("--jsonl-instruct", required=True,
                       help="JSONL file from instruct mode evaluation")
    parser.add_argument("--model-name-thinking", required=True,
                       help="Thinking model name")
    parser.add_argument("--model-name-instruct", required=True,
                       help="Instruct model name")
    parser.add_argument("--output-dir", default="results/linear_probes/comparative",
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to process")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for activation extraction")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (e.g., '0,10,20,30')")
    parser.add_argument("--template-id", type=int, default=None,
                       help="Template ID for organizing results")
    parser.add_argument("--probe-output-components", action="store_true",
                       help="Also probe layer_norm and lm_head")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Ridge regression regularization strength (default: 1.0)")
    parser.add_argument("--attribute", type=str, default='toxicity',
                       choices=['toxicity', 'severe_toxicity', 'identity_attack',
                               'profanity', 'threat', 'sexually_explicit'],
                       help="Which toxicity attribute to analyze")
    
    args = parser.parse_args()

    layer_indices = None
    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]

    main(
        jsonl_thinking=args.jsonl_thinking,
        jsonl_instruct=args.jsonl_instruct,
        model_name_thinking=args.model_name_thinking,
        model_name_instruct=args.model_name_instruct,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        layer_indices=layer_indices,
        template_id=args.template_id,
        probe_output_components=args.probe_output_components,
        alpha=args.alpha,
        attribute=args.attribute
    )