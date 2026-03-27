"""
aligned_probing_comparative.py - 4-Scenario Aligned Probing Framework

Implements the methodology from "Aligned Probing: Relating Toxic Behavior and Model Internals"
Tests 4 scenarios to understand how thinking vs instruct models encode toxicity.

SCENARIOS:
1. Input: Prompt representations → Prompt toxicity (how well model encodes input)
2. Forward: Output representations → Prompt toxicity (does input info propagate?)
3. Output: Output representations → Output toxicity (how well model encodes output)
4. Backward: Prompt representations → Output toxicity (can model predict its toxicity?)

Key Improvements:
- Ridge regression with max_iter=1000 for consistency
- Overfitting detection via train/test correlation gap
- Enhanced visualizations with overfitting analysis
- Comprehensive reporting with statistical metrics
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
    """Extract activations for aligned probing scenarios"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        
        # Load model config
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
        
        # Load tokenizer and model
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_string, 
            trust_remote_code=True
        )
        
        print("Loading model...")
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
        """Hook to capture activations"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            if len(hidden.shape) == 3:
                self.activations[name] = hidden[:, -1, :].detach().cpu().numpy()
        
        return hook
    
    def extract_from_text(
        self,
        text: str,
        layer_idx: int
    ) -> np.ndarray:
        """
        Extract activation from specific text at specific layer.
        
        Args:
            text: Input text (prompt or prompt+continuation)
            layer_idx: Which layer to extract from
            
        Returns:
            Activation vector [hidden_dim]
        """
        self.activations = {}
        
        # Find layers
        if hasattr(self.hf_model, 'model') and hasattr(self.hf_model.model, 'layers'):
            layers = self.hf_model.model.layers
        elif hasattr(self.hf_model, 'transformer') and hasattr(self.hf_model.transformer, 'h'):
            layers = self.hf_model.transformer.h
        else:
            raise ValueError("Could not find transformer layers")
        
        # Register hook on specific layer
        hook = layers[layer_idx].register_forward_hook(
            self._get_activation(f"layer_{layer_idx}")
        )
        
        # Forward pass
        inputs = self.hf_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        with torch.no_grad():
            _ = self.hf_model(**inputs)
        
        # Get activation
        activation = self.activations.get(f"layer_{layer_idx}")
        
        # Cleanup
        hook.remove()
        del inputs
        torch.cuda.empty_cache()
        
        if activation is None:
            return np.array([])
        
        return activation[0]  # Remove batch dimension
    
    def extract_batch_from_texts(
        self,
        texts: List[str],
        layer_idx: int,
        batch_size: int = 4
    ) -> np.ndarray:
        """
        Extract activations from multiple texts at specific layer.
        
        Args:
            texts: List of texts
            layer_idx: Which layer
            batch_size: Batch size for processing
            
        Returns:
            Stacked activations [num_texts, hidden_dim]
        """
        all_activations = []
        
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc=f"Extracting layer {layer_idx}"):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                try:
                    activation = self.extract_from_text(text, layer_idx)
                    if len(activation) > 0:
                        all_activations.append(activation)
                except Exception as e:
                    print(f" Error: {e}")
                    continue
            
            # Memory management
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return np.vstack(all_activations) if all_activations else np.array([])


class AlignedProbe:
    """
    Implement aligned probing with continuous regression.
    Uses Ridge regression (linear regression with L2 regularization).
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: L2 regularization strength (1.0 = moderate, 0.1 = strong)
        """
        self.alpha = alpha
        self.probes = {}
        self.results = {}
    
    def train_probe(
        self,
        activations: np.ndarray,
        toxicity_scores: np.ndarray,
        probe_id: str,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train Ridge regression probe to predict continuous toxicity scores.
        
        Args:
            activations: Hidden states [num_samples, hidden_dim]
            toxicity_scores: Continuous toxicity scores [num_samples] (0.0-1.0)
            probe_id: Identifier for this probe (e.g., "thinking_layer_10_input")
            test_size: Fraction for test set
            
        Returns:
            Dictionary with correlation and other metrics
        """
        
        if len(activations) < 10:
            print(f" Probe {probe_id}: Too few samples ({len(activations)}), skipping")
            return None
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                activations,
                toxicity_scores,
                test_size=test_size,
                random_state=42
            )
        except Exception as e:
            print(f" Probe {probe_id}: Split failed: {e}")
            return None
        
        # Train Ridge regression probe with max_iter for consistency
        probe = Ridge(alpha=self.alpha, random_state=42, max_iter=1000)
        probe.fit(X_train, y_train)
        
        # Predict on test set
        y_test_pred = probe.predict(X_test)
        
        # Compute Pearson correlation (main metric)
        try:
            test_correlation, p_value = pearsonr(y_test, y_test_pred)
        except:
            test_correlation, p_value = 0.0, 1.0
        
        # Additional test metrics
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        # Train correlation for overfitting detection
        y_train_pred = probe.predict(X_train)
        try:
            train_correlation, _ = pearsonr(y_train, y_train_pred)
        except:
            train_correlation = 0.0
        
        # Compute overfitting gap
        overfit_gap = train_correlation - test_correlation
        
        results = {
            'correlation': test_correlation,
            'train_correlation': train_correlation,
            'overfit_gap': overfit_gap,
            'p_value': p_value,
            'mse': mse,
            'r2': r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mean_toxicity': y_train.mean(),
            'std_toxicity': y_train.std()
        }
        
        # Store
        self.probes[probe_id] = probe
        self.results[probe_id] = results
        
        # Print with overfitting detection
        overfit_marker = ""
        if overfit_gap > 0.2:
            overfit_marker = " OVERFIT"
        elif overfit_gap > 0.1:
            overfit_marker = " MILD-OVERFIT"
        
        print(f"  {probe_id}: r={test_correlation:.3f}, train_r={train_correlation:.3f}, "
              f"gap={overfit_gap:.3f}, p={p_value:.4f}{overfit_marker}")
        
        return results


def load_data_with_continuous_scores(
    jsonl_path: str,
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[str], np.ndarray, Dict[str, np.ndarray]]:
    """
    Load data with CONTINUOUS toxicity scores (not binary labels).
    
    Returns:
        prompts: Original prompts
        continuations: Generated continuations
        prompt_toxicity: Toxicity scores of prompts [num_samples]
        output_attributes: Dict of toxicity scores for different attributes
    """
    
    prompts = []
    continuations = []
    prompt_toxicity_scores = []
    
    # Fine-grained attributes (like the paper)
    attributes = {
        'toxicity': [],
        'severe_toxicity': [],
        'identity_attack': [],
        'profanity': [],
        'threat': [],
        'sexually_explicit': [],
        'insult': []
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
                
                # Get continuous toxicity scores (NOT binary!)
                output_tox = content_score.get('toxicity', 0.0)
                
                # Store data
                prompts.append(prompt)
                continuations.append(continuation)
                prompt_toxicity_scores.append(0.0)  # We don't have prompt toxicity in JSONL
                
                # Store fine-grained attributes
                for attr in attributes.keys():
                    attributes[attr].append(content_score.get(attr, 0.0))
                
            except Exception as e:
                continue
    
    # Convert to numpy arrays
    prompt_toxicity = np.array(prompt_toxicity_scores)
    
    # Convert attributes to numpy
    for attr in attributes:
        attributes[attr] = np.array(attributes[attr])
    
    output_toxicity = attributes['toxicity']
    
    print(f"  Loaded: {len(prompts)} samples")
    print(f"  Mean output toxicity: {output_toxicity.mean():.3f} ± {output_toxicity.std():.3f}")
    print(f"  Median: {np.median(output_toxicity):.3f}")
    print(f"  Toxic (>0.5): {(output_toxicity > 0.5).sum()} ({(output_toxicity > 0.5).mean():.1%})")
    
    return prompts, continuations, prompt_toxicity, attributes


def run_aligned_probing_scenarios(
    extractor: HybridActivationExtractor,
    prompts: List[str],
    continuations: List[str],
    prompt_toxicity: np.ndarray,
    output_toxicity: np.ndarray,
    layers_to_test: List[int],
    model_mode: str,
    batch_size: int = 4
) -> Dict[str, Dict[int, Dict]]:
    """
    Run all 4 aligned probing scenarios.
    
    Returns:
        Dict with structure: {scenario: {layer: results}}
    """
    
    print(f"\n{'='*80}")
    print(f"ALIGNED PROBING: {model_mode.upper()} MODEL")
    print(f"{'='*80}")
    
    results = {
        'input': {},      # Scenario 1: Input internals → Input toxicity
        'forward': {},    # Scenario 2: Output internals → Input toxicity
        'output': {},     # Scenario 3: Output internals → Output toxicity  
        'backward': {}    # Scenario 4: Input internals → Output toxicity (PREDICTIVE!)
    }
    
    probe_trainer = AlignedProbe(alpha=1.0)
    
    for layer_idx in layers_to_test:
        print(f"\n{'─'*80}")
        print(f"Layer {layer_idx}")
        print(f"{'─'*80}")
        
        # Extract representations
        print("Extracting representations...")
        
        # Input internals (from prompts only)
        input_internals = extractor.extract_batch_from_texts(
            prompts,
            layer_idx=layer_idx,
            batch_size=batch_size
        )
        
        # Output internals (from full text: prompt + continuation)
        full_texts = [p + "\n" + c for p, c in zip(prompts, continuations)]
        output_internals = extractor.extract_batch_from_texts(
            full_texts,
            layer_idx=layer_idx,
            batch_size=batch_size
        )
        
        if len(input_internals) == 0 or len(output_internals) == 0:
            print("Extraction failed, skipping layer")
            continue
        
        print(f" Extracted: {input_internals.shape} (input), {output_internals.shape} (output)")
        
        # ════════════════════════════════════════════════════════
        # SCENARIO 1: INPUT
        # Question: How well does model encode INPUT toxicity?
        # ════════════════════════════════════════════════════════
        if prompt_toxicity.sum() > 0:  # Only if we have prompt toxicity data
            print("\nScenario 1: INPUT (Input internals → Input toxicity)")
            results['input'][layer_idx] = probe_trainer.train_probe(
                input_internals,
                prompt_toxicity,
                probe_id=f"{model_mode}_layer_{layer_idx}_input"
            )
        
        # ════════════════════════════════════════════════════════
        # SCENARIO 2: FORWARD
        # Question: Does INPUT toxicity info propagate to OUTPUT?
        # ════════════════════════════════════════════════════════
        if prompt_toxicity.sum() > 0:
            print("\nScenario 2: FORWARD (Output internals → Input toxicity)")
            results['forward'][layer_idx] = probe_trainer.train_probe(
                output_internals,
                prompt_toxicity,
                probe_id=f"{model_mode}_layer_{layer_idx}_forward"
            )
        
        # ════════════════════════════════════════════════════════
        # SCENARIO 3: OUTPUT
        # Question: How well does model encode OUTPUT toxicity?
        # ════════════════════════════════════════════════════════
        print("\nScenario 3: OUTPUT (Output internals → Output toxicity)")
        results['output'][layer_idx] = probe_trainer.train_probe(
            output_internals,
            output_toxicity,
            probe_id=f"{model_mode}_layer_{layer_idx}_output"
        )
        
        # ════════════════════════════════════════════════════════
        # SCENARIO 4: BACKWARD MOST INTERESTING!
        # Question: Can model PREDICT its own output toxicity from prompt?
        # ════════════════════════════════════════════════════════
        print("\nScenario 4: BACKWARD (Input internals → Output toxicity)")
        results['backward'][layer_idx] = probe_trainer.train_probe(
            input_internals,
            output_toxicity,
            probe_id=f"{model_mode}_layer_{layer_idx}_backward"
        )
    
    return results


def compare_aligned_probing_results(
    results_thinking: Dict,
    results_instruct: Dict,
    output_dir: Path
):
    """
    Compare aligned probing results between thinking and instruct models.
    Focuses on BACKWARD scenario (most relevant for your research question).
    """
    
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: ALIGNED PROBING")
    print("="*80)
    
    scenarios = ['input', 'forward', 'output', 'backward']
    
    # Create comparison dataframe
    comparison_data = []
    
    for scenario in scenarios:
        print(f"\n{'─'*80}")
        print(f"{scenario.upper()} SCENARIO")
        print(f"{'─'*80}")
        
        # Get all layers tested in this scenario
        thinking_layers = set(results_thinking.get(scenario, {}).keys())
        instruct_layers = set(results_instruct.get(scenario, {}).keys())
        all_layers = sorted(thinking_layers & instruct_layers)  # Intersection
        
        if not all_layers:
            print("No common layers found")
            continue
        
        # Enhanced header with overfitting info
        print(f"{'Layer':<8} {'Think r':<10} {'Inst r':<10} {'Diff':<8} "
              f"{'Think Gap':<12} {'Inst Gap':<12}")
        print("-" * 70)
        
        for layer in all_layers:
            think_res = results_thinking[scenario].get(layer)
            inst_res = results_instruct[scenario].get(layer)
            
            if think_res is None or inst_res is None:
                continue
            
            think_corr = think_res['correlation']
            inst_corr = inst_res['correlation']
            diff = think_corr - inst_corr
            
            # Track overfitting
            think_gap = think_res.get('overfit_gap', 0)
            inst_gap = inst_res.get('overfit_gap', 0)
            
            marker = ""
            if abs(diff) > 0.1:
                marker = "  MAJOR"
            elif abs(diff) > 0.05:
                marker = " NOTABLE"
            
            print(f"{layer:<8} {think_corr:>8.3f}  {inst_corr:>8.3f}  {diff:>6.3f}  "
                  f"{think_gap:>10.3f}  {inst_gap:>10.3f}{marker}")
            
            comparison_data.append({
                'scenario': scenario,
                'layer': layer,
                'thinking_correlation': think_corr,
                'instruct_correlation': inst_corr,
                'difference': diff,
                'thinking_train_correlation': think_res.get('train_correlation', 0),
                'instruct_train_correlation': inst_res.get('train_correlation', 0),
                'thinking_overfit_gap': think_gap,
                'instruct_overfit_gap': inst_gap,
                'thinking_r2': think_res['r2'],
                'instruct_r2': inst_res['r2'],
                'thinking_mse': think_res['mse'],
                'instruct_mse': inst_res['mse']
            })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save results
    df.to_csv(output_dir / 'aligned_probing_comparison.csv', index=False)
    print(f"\n Comparison saved to: {output_dir / 'aligned_probing_comparison.csv'}")
    
    # Overfitting analysis per scenario
    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS BY SCENARIO")
    print("="*80)
    
    for scenario in scenarios:
        scenario_data = df[df['scenario'] == scenario]
        if len(scenario_data) > 0:
            think_avg_gap = scenario_data['thinking_overfit_gap'].mean()
            inst_avg_gap = scenario_data['instruct_overfit_gap'].mean()
            think_max_gap = scenario_data['thinking_overfit_gap'].max()
            inst_max_gap = scenario_data['instruct_overfit_gap'].max()
            
            print(f"\n{scenario.upper()} Scenario:")
            print(f"  Thinking - Avg gap: {think_avg_gap:.3f}, Max gap: {think_max_gap:.3f}")
            print(f"  Instruct - Avg gap: {inst_avg_gap:.3f}, Max gap: {inst_max_gap:.3f}")
            
            if think_max_gap > 0.2 or inst_max_gap > 0.2:
                print(f"  WARNING: Severe overfitting detected!")
            elif think_avg_gap > 0.1 or inst_avg_gap > 0.1:
                print(f"  WARNING: Moderate overfitting detected")
            else:
                print(f"  Overfitting is under control")
    
    # Special analysis of BACKWARD scenario
    print("\n" + "="*80)
    print(" BACKWARD SCENARIO ANALYSIS (Can models predict their toxicity?)")
    print("="*80)
    
    backward_results = df[df['scenario'] == 'backward']
    
    if len(backward_results) > 0:
        avg_thinking = backward_results['thinking_correlation'].mean()
        avg_instruct = backward_results['instruct_correlation'].mean()
        
        print(f"\nAverage Backward Correlation (predicting output toxicity from prompt):")
        print(f"  Thinking: r = {avg_thinking:.3f}")
        print(f"  Instruct: r = {avg_instruct:.3f}")
        print(f"  Difference: {avg_thinking - avg_instruct:+.3f}")
        
        if avg_instruct > avg_thinking + 0.1:
            print(f"\n CRITICAL FINDING:")
            print(f"   Instruct models can PREDICT their own toxicity better!")
            print(f"   → They can self-regulate and filter proactively")
            print(f"   → Thinking models CANNOT predict their toxicity as well")
            print(f"   → This likely explains why thinking is more toxic!")
        elif avg_thinking > avg_instruct + 0.1:
            print(f"\n UNEXPECTED:")
            print(f"   Thinking models predict toxicity better but are still more toxic")
            print(f"   → Suggests filtering failure despite awareness")
        else:
            print(f"\n SIMILAR PREDICTION ABILITY:")
            print(f"   Both models show similar predictive capability")
            print(f"   → Toxicity differences likely due to other mechanisms")
    
    # Visualize
    create_aligned_probing_visualizations(df, output_dir)
    
    return df


def create_aligned_probing_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations for aligned probing results"""
    
    print("\nCreating visualizations...")
    
    scenarios = df['scenario'].unique()
    
    # ═══════════════════════════════════════════════════════════
    # Figure 1: 4-scenario comparison + overfitting
    # ═══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 14))
    
    # Create grid: 3 rows x 2 cols
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Top 2 rows: scenario plots (2x2)
    for idx, scenario in enumerate(scenarios):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        scenario_data = df[df['scenario'] == scenario]
        
        if len(scenario_data) == 0:
            continue
        
        layers = scenario_data['layer'].values
        thinking_corr = scenario_data['thinking_correlation'].values
        instruct_corr = scenario_data['instruct_correlation'].values
        
        # Plot correlations
        ax.plot(layers, thinking_corr, marker='o', label='Thinking',
               linewidth=2.5, markersize=7, color='#d62728', alpha=0.8)
        ax.plot(layers, instruct_corr, marker='s', label='Instruct',
               linewidth=2.5, markersize=7, color='#1f77b4', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, 
                  linewidth=1.5, label='Moderate (r=0.5)')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5,
                  linewidth=1.5, label='Strong (r=0.7)')
        
        ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pearson Correlation', fontsize=11, fontweight='bold')
        ax.set_title(f'{scenario.upper()} Scenario', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.0])
    
    # Bottom row: Overfitting analysis (spans full width)
    ax_overfit = fig.add_subplot(gs[2, :])
    
    # Plot overfitting gaps for each scenario
    x_positions = []
    thinking_gaps = []
    instruct_gaps = []
    labels = []
    
    x = 0
    for scenario in scenarios:
        scenario_data = df[df['scenario'] == scenario]
        for _, row in scenario_data.iterrows():
            x_positions.append(x)
            thinking_gaps.append(row['thinking_overfit_gap'])
            instruct_gaps.append(row['instruct_overfit_gap'])
            labels.append(f"{scenario[:3].upper()}\nL{int(row['layer'])}")
            x += 1
        x += 0.5  # Gap between scenarios
    
    width = 0.35
    x_array = np.arange(len(x_positions))
    
    ax_overfit.bar(x_array - width/2, thinking_gaps, width, 
                   label='Thinking', color='#d62728', alpha=0.7)
    ax_overfit.bar(x_array + width/2, instruct_gaps, width,
                   label='Instruct', color='#1f77b4', alpha=0.7)
    
    ax_overfit.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, 
                      linewidth=1.5, label='Mild (0.1)')
    ax_overfit.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, 
                      linewidth=1.5, label='Severe (0.2)')
    ax_overfit.set_xlabel('Scenario-Layer', fontsize=12, fontweight='bold')
    ax_overfit.set_ylabel('Overfitting Gap (Train r - Test r)', fontsize=12, fontweight='bold')
    ax_overfit.set_title('Overfitting Detection Across Scenarios', fontsize=13, fontweight='bold')
    ax_overfit.set_xticks(x_array)
    ax_overfit.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax_overfit.legend(fontsize=10, loc='best')
    ax_overfit.grid(True, alpha=0.3, axis='y')
    ax_overfit.set_ylim([0, max(max(thinking_gaps + instruct_gaps) * 1.2, 0.3)])
    
    plt.savefig(output_dir / 'aligned_probing_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Scenario comparison plot created")
    
    # ═══════════════════════════════════════════════════════════
    # Figure 2: Heatmap of differences
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot table for heatmap
    pivot_thinking = df.pivot_table(
        index='layer', 
        columns='scenario', 
        values='thinking_correlation',
        aggfunc='first'
    )
    pivot_instruct = df.pivot_table(
        index='layer', 
        columns='scenario', 
        values='instruct_correlation',
        aggfunc='first'
    )
    
    # Difference heatmap
    pivot_diff = pivot_thinking - pivot_instruct
    
    sns.heatmap(
        pivot_diff.T,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.3,
        vmax=0.3,
        cbar_kws={'label': 'Correlation Difference (Thinking - Instruct)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_title('Aligned Probing: Where Thinking and Instruct Differ', 
                fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aligned_probing_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Heatmap created")
    
    # ═══════════════════════════════════════════════════════════
    # Figure 3: Backward scenario focus (most important)
    # ═══════════════════════════════════════════════════════════
    backward_data = df[df['scenario'] == 'backward']
    
    if len(backward_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Correlation comparison
        ax1 = axes[0]
        layers = backward_data['layer'].values
        thinking_corr = backward_data['thinking_correlation'].values
        instruct_corr = backward_data['instruct_correlation'].values
        
        ax1.plot(layers, thinking_corr, marker='o', label='Thinking',
                linewidth=3, markersize=9, color='#d62728', alpha=0.8)
        ax1.plot(layers, instruct_corr, marker='s', label='Instruct',
                linewidth=3, markersize=9, color='#1f77b4', alpha=0.8)
        
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
        ax1.set_title('BACKWARD: Can Models Predict Their Own Toxicity?', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Difference bars
        ax2 = axes[1]
        differences = backward_data['difference'].values
        colors = ['#d62728' if d > 0 else '#1f77b4' for d in differences]
        
        ax2.bar(layers, differences, alpha=0.7, color=colors, 
               edgecolor='black', linewidth=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax2.axhline(y=0.1, color='#d62728', linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.axhline(y=-0.1, color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Correlation Difference (Thinking - Instruct)', 
                      fontsize=12, fontweight='bold')
        ax2.set_title('Predictive Ability Difference', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'backward_scenario_focus.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Backward scenario focus plot created")
    
    print("All visualizations created successfully")


def main(
    jsonl_thinking: str,
    jsonl_instruct: str,
    model_name_thinking: str,
    model_name_instruct: str,
    output_dir: str = "results/aligned_probing",
    max_samples: Optional[int] = None,
    batch_size: int = 4,
    layers: Optional[str] = None,
    template_id: Optional[int] = None,
    attribute: str = 'toxicity',
    alpha: float = 1.0
):
    """
    Main function for aligned probing analysis.
    
    Implements 4-scenario framework from paper with improvements.
    """
    
    output_dir = Path(output_dir)
    if template_id is not None:
        output_dir = output_dir / f"template_{template_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("ALIGNED PROBING FRAMEWORK")
    print("Ridge Regression with Pearson Correlation + Overfitting Detection")
    print("="*80)
    print(f"\nAnalyzing attribute: {attribute}")
    print(f"Regularization (alpha): {alpha}")
    
    # Parse layers
    if layers:
        layers_to_test = [int(x) for x in layers.split(',')]
    else:
        layers_to_test = [0, 5, 10, 15, 20, 25, 30, 35]
    
    print(f"Testing layers: {layers_to_test}")
    
    # Load data with continuous scores
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    prompts_think, conts_think, prompt_tox_think, attributes_think = \
        load_data_with_continuous_scores(jsonl_thinking, max_samples)
    
    prompts_inst, conts_inst, prompt_tox_inst, attributes_inst = \
        load_data_with_continuous_scores(jsonl_instruct, max_samples)
    
    # Get specific attribute
    output_tox_think = attributes_think[attribute]
    output_tox_inst = attributes_inst[attribute]
    
    # Initialize extractors
    extractor_thinking = HybridActivationExtractor(model_name=model_name_thinking)
    
    # Run 4-scenario analysis for thinking model
    results_thinking = run_aligned_probing_scenarios(
        extractor_thinking,
        prompts_think,
        conts_think,
        prompt_tox_think,
        output_tox_think,
        layers_to_test,
        model_mode="thinking",
        batch_size=batch_size
    )
    
    # Clean up before loading second model
    del extractor_thinking
    torch.cuda.empty_cache()
    gc.collect()
    
    # Initialize instruct extractor
    extractor_instruct = HybridActivationExtractor(model_name=model_name_instruct)
    
    # Run 4-scenario analysis for instruct model
    results_instruct = run_aligned_probing_scenarios(
        extractor_instruct,
        prompts_inst,
        conts_inst,
        prompt_tox_inst,
        output_tox_inst,
        layers_to_test,
        model_mode="instruct",
        batch_size=batch_size
    )
    
    # Compare results
    comparison_df = compare_aligned_probing_results(
        results_thinking,
        results_instruct,
        output_dir
    )
    
    # Save complete results
    results_data = {
        'thinking': {
            scenario: {
                int(layer): {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in results.items()
                }
                for layer, results in layer_results.items()
                if results is not None
            }
            for scenario, layer_results in results_thinking.items()
        },
        'instruct': {
            scenario: {
                int(layer): {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in results.items()
                }
                for layer, results in layer_results.items()
                if results is not None
            }
            for scenario, layer_results in results_instruct.items()
        }
    }
    
    with open(output_dir / 'aligned_probing_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\n" + "="*80)
    print("ALIGNED PROBING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles created:")
    print("  - aligned_probing_comparison.csv (detailed comparison)")
    print("  - aligned_probing_results.json (all results)")
    print("  - aligned_probing_scenarios.png (4-scenario comparison + overfitting)")
    print("  - aligned_probing_heatmap.png (difference heatmap)")
    print("  - backward_scenario_focus.png (backward scenario detailed view)")
    
    return comparison_df, results_thinking, results_instruct


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Aligned Probing: 4-Scenario Framework for Thinking vs Instruct"
    )
    
    parser.add_argument("--jsonl-thinking", required=True,
                       help="JSONL file from thinking mode evaluation")
    parser.add_argument("--jsonl-instruct", required=True,
                       help="JSONL file from instruct mode evaluation")
    parser.add_argument("--model-name-thinking", required=True,
                       help="Thinking model name")
    parser.add_argument("--model-name-instruct", required=True,
                       help="Instruct model name")
    parser.add_argument("--output-dir", default="results/aligned_probing",
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to process")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for extraction")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (e.g., '0,10,20,30,35')")
    parser.add_argument("--template-id", type=int, default=None,
                       help="Template ID")
    parser.add_argument("--attribute", type=str, default='toxicity',
                       choices=['toxicity', 'severe_toxicity', 'identity_attack', 
                                'profanity', 'threat', 'sexually_explicit'],
                       help="Which toxicity attribute to analyze")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Ridge regression regularization strength (default: 1.0)")
    
    args = parser.parse_args()
    
    main(
        jsonl_thinking=args.jsonl_thinking,
        jsonl_instruct=args.jsonl_instruct,
        model_name_thinking=args.model_name_thinking,
        model_name_instruct=args.model_name_instruct,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        layers=args.layers,
        template_id=args.template_id,
        attribute=args.attribute,
        alpha=args.alpha
    )