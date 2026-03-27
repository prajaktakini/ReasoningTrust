"""
Comparative Linear Probe for Thinking vs Instruct Models

This module implements linear probing to analyze layer-wise toxicity emergence
differences between thinking and instruct models.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import sys
import warnings
warnings.filterwarnings('ignore')
from reasoning_trust.models.initialize_model import load_model_config


# Add parent directory to path to import from reasoning_trust
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

class HybridActivationExtractor:
    """
    Extract activations from both thinking and instruct modes for hybrid analysis.
    
    Uses the same model configuration as vLLM setup but loads with transformers
    for activation extraction (vLLM doesn't support hooks).
    """
    
    def __init__(self, model_name: str = None, device: str = "cuda"):
        """
        Initialize extractor using either model_name or model_config.
        """ 
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

        print(f"✓ Model loaded: {type(self.hf_model)}")
        print(f"✓ Model is None: {self.hf_model is None}")
        print(f"✓ Tokenizer loaded: {type(self.hf_tokenizer)}")
        print(f"✓ Tokenizer is None: {self.hf_tokenizer is None}")

        # Test that tokenizer is callable
        try:
            test_output = self.hf_tokenizer("test", return_tensors="pt")
            print(f"✓ Tokenizer works!")
        except Exception as e:
            print(f"✗ Tokenizer error: {e}")

        print("Model loaded successfully.")
        self.activations = {}
        self.hooks = []


    def _get_activation(self, name):
        """Hook to capture activations from specified layers."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]  # [batch, seq_len, hidden_dim]
            else:
                hidden = output

            if len(hidden.shape) == 3:
                # Store last token representation
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
        """
        Extract activations from full text (prompt + continuation).
        This ensures we're analyzing the same text that was evaluated for toxicity.
        """

        self.register_hooks(layer_indices)
        self.activations = {}

        # Tokenize full text
        inputs = self.hf_tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)

        # Forward pass to extract activations
        with torch.no_grad():
            _ = self.hf_model(**inputs)

        # Collect activations by layer
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
        """
        Extract activations from multiple full texts"
        """
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
                        all_layer_activations[layer_idx].append(act[0])  # Remove batch dim
                except Exception as e:
                    print(f"Error extracting activations for text: {e}")
                    continue

            # Memory management
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # stack all activations
        return {
            idx: np.vstack(acts) 
            for idx, acts in all_layer_activations.items() 
            if acts
        }
    
    def extract_output_components(
            self,
            full_texts: List[str],
            component: str = 'layer_norm', # 'layer_norm' or 'logits'
            batch_size: int = 4,
    ) -> np.ndarray:
        """
        Extract activations from specified output component for given texts (after transformer layers).

        Args:
            full_texts: List of full texts (prompt + continuation)
            Component: 'layer_norm' or 'logits'
                - 'layer_norm': Final layer normalization output
                - 'logits': lm_head output (vocab logits)
            batch_size: Batch size for processing
        Returns:
            Stacked activations [num_samples, feature_dim]
            - For layer_norm: feature_dim = hidden_dim (3584)
            - For logits: feature_dim = vocab_size (e.g., 50432)
        """
        
        activations = []

        # Define hook based on component
        if component == 'layer_norm':
            target_module = self.hf_model.model.norm

            def hook(module, input, output):
                # output is the normalized hidden states
                # Shape: [batch, seq_len, hidden_dim]
                activations.append(output[:, -1, :].detach().cpu().numpy())
        
        elif component == 'logits':
            target_module = self.hf_model.lm_head

            def hook(module, input, output):
                # output is the logits
                # Shape: [batch, seq_len, vocab_size]
                activations.append(output[:, -1, :].detach().cpu().numpy())

        else:
            raise ValueError(f"Unknown component: {component}. Use 'layer_norm' or 'logits'")

        # Register hook
        handle = target_module.register_forward_hook(hook)

        # Process texts
        print("Extracting output component activations from component:", component)
        for i in tqdm(range(0, len(full_texts), batch_size), desc=f"Extracting {component}"):
            batch_texts = full_texts[i:i+batch_size]

            for text in batch_texts:
                try:
                    # Tokenize full text
                    inputs = self.hf_tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096
                    ).to(self.device)

                    # Forward pass
                    with torch.no_grad():
                        _ = self.hf_model(**inputs)

                    del inputs

                except Exception as e:
                    print(f"Error extracting {component} for text: {e}")
                    
                    # Add zero vector as placeholder
                    if component == 'layer_norm':
                        activations.append(np.zeroes(1, self.hf_model.config.hidden_size))
                    else:
                        activations.append(np.zeroes(1, self.hf_model.config.vocab_size))

            # Memory management
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # Remove hook
        handle.remove()

        # Stack and return
        if len(activations) == 0:
            return np.vstack([])
        
        return np.vstack(activations)

    
    
def load_data_from_jsonl(jsonl_path: str, 
                         max_samples: Optional[int] = None, 
                         toxicity_threshold: float = 0.5) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """
    Load data from JSONL evaluation file.
    
    Returns:
        prompts: Original prompts
        continuations: Generated continuations
        full_texts: prompt + continuation (for activation extraction)
        labels: Binary toxicity labels
    """

    prompts = []
    continuations = []
    full_texts = []
    labels = []

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

                toxicity = content_score.get('toxicity', 0.0)
                label = 1 if toxicity > toxicity_threshold else 0

                full_text = prompt + "\n" + continuation
                prompts.append(prompt)
                continuations.append(continuation)
                full_texts.append(full_text)
                labels.append(label)

            except Exception as e:
                print(f"Skipping it! Error processing line {i}: {e}")
                continue

    labels = np.array(labels)

    print(f"  Loaded: {len(full_texts)} samples")
    print(f"  Toxic: {labels.sum()} ({labels.mean():.1%})")
    print(f"  Non-toxic: {len(labels) - labels.sum()} ({1-labels.mean():.1%})")

    return prompts, continuations, full_texts, labels

class ComparativeLinearProbe:
    """Train and compare linear probes using Logistic Regression."""

    def __init__(self, use_cross_validation: bool = False):
        self.use_cv = use_cross_validation
        self.probes_thinking = {}
        self.probes_instruct = {}
        self.results_thinking = {}
        self.results_instruct = {}
        self.differences = {}
        
    def train_probe(
            self, 
            activations: np.ndarray,
            labels: np.ndarray,
            layer_idx: int,
            mode: str,
            test_size: float = 0.2
    ):
        """Train logistic regression probe for binary classification.
        
        Args:
            activations: Hidden States[num_samples, hidden_dim]
            labels: Binary labels [num_samples]
            layer_idx: Layer index 
            mode: "thinking" or "instruct"
        """

        # Check class balance
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            print(f"Warning: Layer {layer_idx} ({mode}) has only one class, skipping")
            return None

        class_counts = np.bincount(labels)
        minority_class = np.min(class_counts)

        if minority_class < 5:
            print(f"Warning: Layer {layer_idx} ({mode}) has very few samples in minority class, skipping")
            return None
        
        # Use cross validation for small datasets
        if self.use_cv and len(labels) < 180:
            return self._train_with_cv(activations, labels, layer_idx, mode)
        
        # standard train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                activations, labels, test_size=test_size, random_state=42, stratify=labels
            )
        except ValueError as e:
            print(f" Layer {layer_idx} ({mode}): Cannot stratify, using random split, {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                activations, labels, test_size=test_size, random_state=42
            )

        
        # Train Logistic Regression probe
        probe = LogisticRegression(max_iter=1000, C=0.1, random_state=42, class_weight='balanced', solver='lbfgs', n_jobs=-1)
        probe.fit(X_train, y_train)

        # Evaluate test set
        y_test_pred = probe.predict(X_test)
        y_test_pred_proba = probe.predict_proba(X_test)[:, 1]

        # Compute metrics
        try:
            test_auc = roc_auc_score(y_test, y_test_pred_proba)
        except:
            test_auc = 0.5

        # Train Set evaluation
        y_train_pred = probe.predict(X_train)
        y_train_pred_proba = probe.predict_proba(X_train)[:, 1]

        try:
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
        except:
            train_auc = 0.5

        # Compute overfitting gap
        overfit_gap = train_auc - test_auc

        # Check convergence
        n_iter = probe.n_iter_[0] if hasattr(probe, 'n_iter_') else 0

        # Compute test metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

        results = {
            'test_auc': test_auc,           
            'train_auc': train_auc,         
            'overfit_gap': overfit_gap,      
            'n_iter': n_iter,                
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_ratio': y_train.mean(),
            'layer_idx': layer_idx
        }

        # Store
        if mode == "thinking":
            self.probes_thinking[layer_idx] = probe
            self.results_thinking[layer_idx] = results
        else:
            self.probes_instruct[layer_idx] = probe
            self.results_instruct[layer_idx] = results

        
        overfit_marker = ""
        if overfit_gap > 0.15:
            overfit_marker = " OVERFIT"
        elif overfit_gap > 0.08:
            overfit_marker = " MILD-OVERFIT"
        
        converge_marker = ""
        if n_iter >= 1900:  # Close to max_iter
            converge_marker = " NO-CONVERGE"
        
        print(f"Layer {layer_idx:2d} ({mode:9s}): "
            f"Test AUC={test_auc:.3f}, Train AUC={train_auc:.3f}, "
            f"Gap={overfit_gap:.3f}, Iter={n_iter}{overfit_marker}{converge_marker}")


        return results


    def compare_modes(self):
        """Compare probe performance between thinking and instruct modes."""
        all_layers = set(self.results_thinking.keys()) | set(self.results_instruct.keys())

        for layer_idx in all_layers:
            thinking_auc = self.results_thinking.get(layer_idx, {}).get('test_auc', 0)  
            instruct_auc = self.results_instruct.get(layer_idx, {}).get('test_auc', 0)  

            self.differences[layer_idx] = {
                'thinking_auc': thinking_auc,
                'instruct_auc': instruct_auc,
                'thinking_train_auc': self.results_thinking.get(layer_idx, {}).get('train_auc', 0),  
                'instruct_train_auc': self.results_instruct.get(layer_idx, {}).get('train_auc', 0),  
                'thinking_overfit': self.results_thinking.get(layer_idx, {}).get('overfit_gap', 0),  
                'instruct_overfit': self.results_instruct.get(layer_idx, {}).get('overfit_gap', 0),  
                'difference': thinking_auc - instruct_auc,
                'relative_diff_pct': (thinking_auc - instruct_auc) / max(instruct_auc, 0.001) * 100
            }

    def analyze_emergence_and_suppression(self, threshold=0.7):
        """Analyze where toxicity emerges or is suppressed."""
        thinking_emergence = [
            layer for layer, results in self.results_thinking.items()
            if results.get('test_auc', 0) > threshold  
        ]

        instruct_emergence = [
            layer for layer, results in self.results_instruct.items()
            if results.get('test_auc', 0) > threshold 
        ]

        thinking_better = [
            layer for layer, diff in self.differences.items()
            if diff.get('difference', 0) > 0.05
        ]

        def find_suppression_layers(results_dict):
            drops = []
            sorted_layers = sorted(results_dict.keys())
            for i in range(1, len(sorted_layers)):
                prev_layer = sorted_layers[i-1]
                curr_layer = sorted_layers[i]
                prev_auc = results_dict[prev_layer].get('test_auc', 0)
                curr_auc = results_dict[curr_layer].get('test_auc', 0)
                if curr_auc < prev_auc - 0.05:
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
        """Create visualization comparing modes"""
        
        all_layers = sorted(set(self.results_thinking.keys()) | set(self.results_instruct.keys()))
        
        thinking_aucs = [self.results_thinking.get(l, {}).get('test_auc', 0) for l in all_layers]
        instruct_aucs = [self.results_instruct.get(l, {}).get('test_auc', 0) for l in all_layers]
        
        thinking_gaps = [self.results_thinking.get(l, {}).get('overfit_gap', 0) for l in all_layers]
        instruct_gaps = [self.results_instruct.get(l, {}).get('overfit_gap', 0) for l in all_layers]

        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        
        # Plot 1: AUC comparison 
        ax1 = axes[0]
        ax1.plot(all_layers, thinking_aucs, marker='o', label='Thinking Mode', 
                linewidth=2.5, markersize=7, color='#d62728', alpha=0.8)
        ax1.plot(all_layers, instruct_aucs, marker='s', label='Instruct Mode',
                linewidth=2.5, markersize=7, color='#1f77b4', alpha=0.8)
        ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Strong (0.7)')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, linewidth=1.5, label='Random (0.5)')
        ax1.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
        ax1.set_title('Layer-wise Toxicity Detection (Test Set)', 
                    fontsize=15, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0.4, 1.0])

        # Plot 2: Difference 
        ax2 = axes[1]
        differences = [self.differences.get(l, {}).get('difference', 0) for l in all_layers]
        colors = ['#d62728' if d > 0 else '#1f77b4' for d in differences]
        ax2.bar(all_layers, differences, alpha=0.7, color=colors, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax2.axhline(y=0.05, color='#d62728', linestyle='--', alpha=0.5, linewidth=1.5, label='Significant (+0.05)')
        ax2.axhline(y=-0.05, color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5, label='Significant (-0.05)')
        ax2.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
        ax2.set_ylabel('AUC Difference (Thinking - Instruct)', fontsize=13, fontweight='bold')
        ax2.set_title('Where Thinking Shows Stronger Signal', 
                    fontsize=15, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Plot 3: Overfitting gaps
        ax3 = axes[2]
        ax3.plot(all_layers, thinking_gaps, marker='o', label='Thinking Overfit Gap',
                linewidth=2.5, markersize=7, color='#d62728', alpha=0.8)
        ax3.plot(all_layers, instruct_gaps, marker='s', label='Instruct Overfit Gap',
                linewidth=2.5, markersize=7, color='#1f77b4', alpha=0.8)
        ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Mild Overfit (0.1)')
        ax3.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Severe Overfit (0.15)')
        ax3.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Train AUC - Test AUC', fontsize=13, fontweight='bold')
        ax3.set_title('Overfitting Detection by Layer', fontsize=15, fontweight='bold', pad=15)
        ax3.legend(fontsize=11, loc='best')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim([0, 0.3])
        
        plt.tight_layout()
        plt.savefig(output_dir / "layer_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    

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
                # Test metrics
                'thinking_test_auc': thinking_res.get('test_auc', 0),     
                'thinking_train_auc': thinking_res.get('train_auc', 0),    
                'thinking_overfit_gap': thinking_res.get('overfit_gap', 0),
                'thinking_n_iter': thinking_res.get('n_iter', 0),           
                'thinking_accuracy': thinking_res.get('accuracy', 0),
                'thinking_f1': thinking_res.get('f1', 0),
                
                'instruct_test_auc': instruct_res.get('test_auc', 0),      
                'instruct_train_auc': instruct_res.get('train_auc', 0),     
                'instruct_overfit_gap': instruct_res.get('overfit_gap', 0), 
                'instruct_n_iter': instruct_res.get('n_iter', 0),           
                'instruct_accuracy': instruct_res.get('accuracy', 0),
                'instruct_f1': instruct_res.get('f1', 0),
                
                'auc_difference': diff.get('difference', 0),
                'relative_diff_pct': diff.get('relative_diff_pct', 0),    
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        report_path = output_dir / "comparative_report.csv"
        df.to_csv(report_path, index=False)

        # Print overfitting summary
        print("\n" + "="*80)
        print("SUMMARY FINDINGS")
        print("="*80)

        if len(df) > 0:
            # Best layers
            best_thinking = df.loc[df['thinking_test_auc'].idxmax()]
            best_instruct = df.loc[df['instruct_test_auc'].idxmax()]

            print(f"\nBest Layers (Test AUC):")
            print(f"  Thinking: Layer {int(best_thinking['layer'])} (AUC={best_thinking['thinking_test_auc']:.3f})")
            print(f"  Instruct: Layer {int(best_instruct['layer'])} (AUC={best_instruct['instruct_test_auc']:.3f})")

            # Overfitting analysis
            print(f"\nOverfitting Analysis:")
            avg_think_gap = df['thinking_overfit_gap'].mean()
            avg_inst_gap = df['instruct_overfit_gap'].mean()
            max_think_gap = df['thinking_overfit_gap'].max()
            max_inst_gap = df['instruct_overfit_gap'].max()
            
            print(f"  Thinking - Avg gap: {avg_think_gap:.3f}, Max gap: {max_think_gap:.3f}")
            print(f"  Instruct - Avg gap: {avg_inst_gap:.3f}, Max gap: {max_inst_gap:.3f}")
            
            if max_think_gap > 0.15 or max_inst_gap > 0.15:
                print(f"  WARNING: Severe overfitting detected!")
            elif avg_think_gap > 0.1 or avg_inst_gap > 0.1:
                print(f"  WARNING: Moderate overfitting detected")
            else:
                print(f"  Overfitting is under control")
            
            # Convergence analysis
            print(f"\nConvergence Analysis:")
            avg_think_iter = df['thinking_n_iter'].mean()
            avg_inst_iter = df['instruct_n_iter'].mean()
            max_think_iter = df['thinking_n_iter'].max()
            max_inst_iter = df['instruct_n_iter'].max()
            
            print(f"  Thinking - Avg iterations: {avg_think_iter:.0f}, Max: {max_think_iter:.0f}")
            print(f"  Instruct - Avg iterations: {avg_inst_iter:.0f}, Max: {max_inst_iter:.0f}")
            
            if max_think_iter >= 1900 or max_inst_iter >= 1900:
                print(f"  WARNING: Some probes did not converge (hit max_iter)")
            else:
                print(f"  All probes converged")

            # Significant differences
            significant_diff = df[df['auc_difference'].abs() > 0.05]
            if len(significant_diff) > 0:
                print(f"\nSignificant AUC Differences (|Δ| > 0.05):")
                for _, row in significant_diff.iterrows():
                    print(f"  Layer {int(row['layer']):2d}: Δ={row['auc_difference']:+.3f} "
                        f"({row['relative_diff_pct']:+.1f}%)")
            
            avg_diff = df['auc_difference'].mean()
            print(f"\nAverage AUC difference: {avg_diff:+.3f}")
        
        print(f"\n Report saved to: {report_path}")
        
        return df


def main(jsonl_thinking: str,
         jsonl_instruct: str,
         model_name_thinking: str = None,
         model_name_instruct: str = None,
         output_dir: str = "results/linear_probes/comparative",
         max_samples: Optional[int] = None,
         batch_size: int = 4,
         layer_indices: Optional[List[int]] = None,
         template_id: Optional[int] = None,
         toxicity_threshold: float = 0.5,
         use_cross_validation: bool = False,
         probe_output_components: bool = True
):
    """Main function for comparative linear probing analysis."""

    output_dir = Path(output_dir)
    if template_id is not None:
        output_dir = output_dir / f"template_{template_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("COMPARATIVE LINEAR PROBING ANALYSIS")
    print("Using Logistic Regression for Binary Classification")
    print("="*80)

    _, _, full_texts_thinking, labels_thinking = load_data_from_jsonl(
        jsonl_thinking, max_samples=max_samples, toxicity_threshold=toxicity_threshold
    )

    _, _, full_texts_instruct, labels_instruct = load_data_from_jsonl(
        jsonl_instruct, max_samples=max_samples, toxicity_threshold=toxicity_threshold
    )

    # Initialize extractor
    extractor_thinking = HybridActivationExtractor(model_name=model_name_thinking)
    extractor_instruct = HybridActivationExtractor(model_name=model_name_instruct)

    # Extract activations
    print("\n" + "="*80)
    print("EXTRACTING ACTIVATIONS FROM GENERATED OUTPUTS")
    print("="*80)

    print("\nExtracting thinking mode activations...")
    activations_thinking = extractor_thinking.extract_batch(
        full_texts_thinking,
        layer_indices=layer_indices,
        batch_size=batch_size
    )

    torch.cuda.empty_cache()
    gc.collect()

    print("\nExtracting instruct mode activations...")
    activations_instruct = extractor_instruct.extract_batch(
        full_texts_instruct,
        layer_indices=layer_indices,
        batch_size=batch_size
    )


    # Train probes
    print("\n" + "="*80)
    print("TRAINING LINEAR PROBES (Logistic Regression)")
    print("="*80)

    probe = ComparativeLinearProbe(use_cross_validation=use_cross_validation)

    print("\nTraining probes for thinking mode...")
    for layer_idx in sorted(activations_thinking.keys()):
        probe.train_probe(
            activations=activations_thinking[layer_idx],
            labels=labels_thinking,
            layer_idx=layer_idx,
            mode="thinking"
        )
    
    print("\nTraining probes for instruct mode...")
    for layer_idx in sorted(activations_instruct.keys()):
        probe.train_probe(
            activations=activations_instruct[layer_idx],
            labels=labels_instruct,
            layer_idx=layer_idx,
            mode="instruct"
        )
    
    if probe_output_components:
        print("\n" + "="*80)
        print("PROBING OUTPUT COMPONENTS (Beyond Transformer Layers)")
        print("="*80)
        print("Testing: Layer Norm → lm_head (where filtering may occur)")

        # ------------------------------------
        # Component 1: Layer Norm
        # ------------------------------------
        print("\n[1/2] Extracting from final layer norm...")
        ln_thinking = extractor_thinking.extract_output_components(
            full_texts_thinking,
            component='layer_norm',
            batch_size=batch_size
        )

        ln_instruct = extractor_instruct.extract_output_components(
            full_texts_instruct,
            component='layer_norm',
            batch_size=batch_size
        )

        if len(ln_thinking) > 0 and len(ln_instruct) > 0:
            print("\nTraining probes on layer norm outputs...")

            # User layer_id=100 to distinguish from transformer layers
            probe.train_probe(ln_thinking, labels_thinking, layer_idx=100, mode="thinking")
            probe.train_probe(ln_instruct, labels_instruct, layer_idx=100, mode="instruct")

            ln_think_auc = probe.results_thinking[100]['test_auc']
            ln_inst_auc = probe.results_instruct[100]['test_auc']
            ln_diff = ln_think_auc - ln_inst_auc

            print(f"\nLayer Norm Results:")
            print(f"  Thinking AUC: {ln_think_auc:.3f}")
            print(f"  Instruct AUC: {ln_inst_auc:.3f}")
            print(f"  Difference: {ln_diff:+.3f}")

            if abs(ln_diff) > 0.05:
                print(f"  SIGNIFICANT DIFFERENCE FOUND IN LAYER NORM!")
                if ln_diff > 0:
                    print(f"     → Thinking maintains stronger toxicity signal")
                    print(f"     → Instruct suppresses toxicity at layer norm")
                else:
                    print(f"     → Instruct maintains stronger toxicity signal")
        else:
            print("Failed to extract layer norm activations")

        # ------------------------------------
        # Component 2: ln_head Logits
        # ------------------------------------
        print("\n[2/2] Extracting from lm_head (vocabulary logits)...")
        print("Note: High-dimensional (vocab_size=151936). Using PCA for efficiency...")

        try:
            from sklearn.decomposition import PCA

            logits_thinking = extractor_thinking.extract_output_components(
                full_texts_thinking,
                component='logits',
                batch_size=batch_size
            )

            logits_instruct = extractor_instruct.extract_output_components(
                full_texts_instruct,
                component='logits',
                batch_size=batch_size
            )

            if len(logits_thinking) > 0 and len(logits_instruct) > 0:
                # Reduce dimensionality with PCA
                # Calculate valid n_components
                n_samples = len(logits_thinking) + len(logits_instruct)
                max_components = min(n_samples, logits_thinking.shape[1])
                n_components = min(1000, max_components - 10)  # Keep buffer
                
                print(f"Applying PCA: {logits_thinking.shape[1]} → {n_components} dimensions...")
                print(f"  (Max possible: {max_components}, using {n_components} for stability)")
                
                pca = PCA(n_components=n_components, random_state=42)
                
                # Fit on combined data
                combined = np.vstack([logits_thinking, logits_instruct])
                pca.fit(combined)


                print("\nTraining probes on lm_head logits (PCA-reduced)...")

                logits_thinking_pca = pca.transform(logits_thinking)
                logits_instruct_pca = pca.transform(logits_instruct)

                print(f"✓ PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")


                  
                print("\nTraining probes on lm_head logits...")
                # Use layer_idx=101 to distinguish from other components
                probe.train_probe(logits_thinking_pca, labels_thinking, layer_idx=101, mode="thinking")
                probe.train_probe(logits_instruct_pca, labels_instruct, layer_idx=101, mode="instruct")
                
                logits_think_auc = probe.results_thinking[101]['test_auc']
                logits_inst_auc = probe.results_instruct[101]['test_auc']
                logits_diff = logits_think_auc - logits_inst_auc
                
                print(f"\nlm_head Logits Results:")
                print(f"  Thinking AUC: {logits_think_auc:.3f}")
                print(f"  Instruct AUC: {logits_inst_auc:.3f}")
                print(f"  Difference: {logits_diff:+.3f}")
                
                if abs(logits_diff) > 0.05:
                    print(f" SIGNIFICANT DIFFERENCE FOUND IN LM_HEAD!")
                    if logits_diff > 0:
                        print(f"     → Thinking maintains toxicity signal in logits")
                        print(f"     → Instruct filters toxic tokens at lm_head")
                    else:
                        print(f"     → Instruct maintains stronger signal")
            else:
                print("Failed to extract lm_head logits")
                
        except ImportError:
            print("sklearn PCA not available, skipping logits analysis")
        except Exception as e:
            print(f"Error in logits extraction: {e}")


    

    # Analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS OF PROBE PERFORMANCE")
    print("="*80)

    probe.compare_modes()
    analysis = probe.analyze_emergence_and_suppression()

    print(f"\nToxicity Emergence Layers (AUC > 0.7):")
    print(f"  Thinking: {analysis['thinking_emergence_layers']}")
    print(f"  Instruct: {analysis['instruct_emergence_layers']}")
    
    print(f"\nLayers where Thinking shows significantly stronger signal:")
    print(f"  {analysis['thinking_better_layers']}")
    
    print(f"\nSuppression Layers (AUC drops):")
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
    print("COMPARATIVE LINEAR PROBING ANALYSIS COMPLETED")
    print("="*80)


    return probe, report_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comparative Linear Probing for Toxicity Detection using Logistic Regression")

    parser.add_argument("--jsonl_thinking", type=str, required=True, help="JSONL file from thinking mode evaluation")
    parser.add_argument("--jsonl_instruct", type=str, required=True, help="JSONL file from instruct mode evaluation")
    parser.add_argument("--model_name_thinking", type=str, default=None, help="Model name Thinking (e.g., 'Qwen/Qwen3-4B-Thinking-2507')")
    parser.add_argument("--model_name_instruct", type=str, default=None, help="Model name Instruct (e.g., 'Qwen/Qwen3-4B-Instruct-2507')")
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
    parser.add_argument("--toxicity-threshold", type=float, default=0.5,
                       help="Threshold for binary classification (default: 0.5)")
    parser.add_argument("--use-cv", action="store_true",
                       help="Use cross-validation for small datasets")
    parser.add_argument("--probe-output-components", action="store_true",
                       help="Probe output components (layer norm, logits) in addition to transformer layers")
    
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
        toxicity_threshold=args.toxicity_threshold,
        use_cross_validation=args.use_cv,
        probe_output_components=args.probe_output_components,
    )

