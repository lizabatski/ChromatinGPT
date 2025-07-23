import torch
from model import NetDeepHistone              # Original DeepHistone
from experiments.enformer.mini_enformer import NetDeepHistoneEnformer  # Enformer version
from experiments.enformer.mini_dhica import SeparatePathwayModel    # Your dual-pathway model

def count_parameters(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}:")
    print(f"  Total Params     : {total_params:,}")
    print(f"  Trainable Params : {trainable_params:,}")
    
    # Calculate parameter size in MB
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"  Model Size (MB)  : {param_size_mb:.2f}")
    print()
    
    return total_params, trainable_params

def detailed_parameter_breakdown(model, name):
    print(f"\n{name} - Detailed Parameter Breakdown:")
    print("-" * 60)
    
    total = 0
    for module_name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                print(f"  {module_name:<35} : {module_params:>10,}")
                total += module_params
    
    print("-" * 60)
    print(f"  {'Total':<35} : {total:>10,}")
    print()

def analyze_dual_pathway_architecture(model, name):
    """Analyze the dual-pathway model architecture in detail"""
    print(f"\n{name} - Architecture Analysis:")
    print("-" * 60)
    
    # Analyze DNA pathway
    dna_params = sum(p.numel() for p in model.dna_pathway.parameters())
    print(f"  DNA Pathway              : {dna_params:>10,} parameters")
    
    # Analyze DNase pathway  
    dnase_params = sum(p.numel() for p in model.dnase_pathway.parameters())
    print(f"  DNase Pathway            : {dnase_params:>10,} parameters")
    
    # Analyze fusion layer
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"  Fusion Layer             : {fusion_params:>10,} parameters")
    
    # Analyze transformer layers
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    print(f"  Transformer Layers       : {transformer_params:>10,} parameters")
    
    # Analyze final layers
    final_conv_params = sum(p.numel() for p in model.final_conv.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    final_params = final_conv_params + classifier_params
    print(f"  Final Layers             : {final_params:>10,} parameters")
    
    # Calculate percentages
    total = dna_params + dnase_params + fusion_params + transformer_params + final_params
    print("-" * 60)
    print(f"  DNA Pathway              : {dna_params/total*100:>9.1f}%")
    print(f"  DNase Pathway            : {dnase_params/total*100:>9.1f}%") 
    print(f"  Fusion Layer             : {fusion_params/total*100:>9.1f}%")
    print(f"  Transformer Layers       : {transformer_params/total*100:>9.1f}%")
    print(f"  Final Layers             : {final_params/total*100:>9.1f}%")
    print("-" * 60)
    print(f"  Total                    : {total:>10,} parameters")
    print()

# === Instantiate DeepHistone (Original) ===
print("Instantiating DeepHistone (Original)...")
deep_histone = NetDeepHistone()

# === Instantiate DeepHistone-Enformer ===
print("Instantiating DeepHistone-Enformer...")
deep_histone_enformer = NetDeepHistoneEnformer(
    input_channels=5,
    channels=1536,             # Width
    num_transformer_layers=10,  # Layers
    num_heads=12,               # Attention heads
    dropout=0.1,               # Updated dropout
    num_histones=7,
    pooling_type='attention',
    num_conv_blocks=5         # Updated conv blocks
)

# === Instantiate Dual-Pathway Model ===
print("Instantiating Dual-Pathway Model...")
dual_pathway_model = SeparatePathwayModel(
    channels=1024,              # Smaller than Enformer for comparison
    num_transformer_layers=6,  # Fewer layers than full Enformer
    num_heads=8,
    dropout=0.4,
    num_histones=7,
    pooling_type='attention',
    num_conv_blocks=5,
    fusion_type='concat'       # Using concat fusion
)

# === Count and print parameters ===
print("=" * 60)
print("Model Parameter Counts")
print("=" * 60)

original_params, _ = count_parameters(deep_histone, "DeepHistone (Original)")
enformer_params, _ = count_parameters(deep_histone_enformer, "DeepHistone-Enformer (Hybrid)")
dual_pathway_params, _ = count_parameters(dual_pathway_model, "Dual-Pathway Model")

print(f"Parameter Ratios:")
print(f"  Enformer/Original    : {enformer_params/original_params:.2f}x")
print(f"  Dual-Pathway/Original: {dual_pathway_params/original_params:.2f}x")
print(f"  Dual-Pathway/Enformer: {dual_pathway_params/enformer_params:.2f}x")
print("=" * 60)

# === Detailed breakdown (optional) ===
print("\n" + "=" * 60)
print("DETAILED PARAMETER BREAKDOWN")
print("=" * 60)

detailed_parameter_breakdown(deep_histone, "DeepHistone (Original)")
detailed_parameter_breakdown(deep_histone_enformer, "DeepHistone-Enformer (Hybrid)")
detailed_parameter_breakdown(dual_pathway_model, "Dual-Pathway Model")

# === Dual-pathway architecture analysis ===
analyze_dual_pathway_architecture(dual_pathway_model, "Dual-Pathway Model")

# === Model comparison summary ===
print("=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

print(f"DeepHistone (Original)     : {original_params:>12,} parameters")
print(f"DeepHistone-Enformer       : {enformer_params:>12,} parameters")
print(f"Dual-Pathway Model         : {dual_pathway_params:>12,} parameters")
print()
print(f"Enformer vs Original       : +{enformer_params - original_params:>11,} parameters ({enformer_params/original_params:.2f}x)")
print(f"Dual-Pathway vs Original   : +{dual_pathway_params - original_params:>11,} parameters ({dual_pathway_params/original_params:.2f}x)")
print(f"Dual-Pathway vs Enformer   : {dual_pathway_params - enformer_params:>+11,} parameters ({dual_pathway_params/enformer_params:.2f}x)")

print("\nArchitecture Differences:")
print("- DeepHistone: Pure CNN architecture")
print("- DeepHistone-Enformer: CNN + Transformer hybrid")
print("- Dual-Pathway: Separate DNA & DNase pathways + fusion + transformer")

print("\nKey Innovations:")
print("- Dual-Pathway: Specialized processing for DNA sequence vs accessibility data")
print("- Fusion Layer: Combines complementary information from both pathways")
print("- Modular Design: Can experiment with different fusion strategies")

print("\nMemory Estimate (rough):")
original_memory = original_params * 4 / (1024**2)  # MB
enformer_memory = enformer_params * 4 / (1024**2)  # MB
dual_pathway_memory = dual_pathway_params * 4 / (1024**2)  # MB

print(f"DeepHistone memory         : ~{original_memory:.1f} MB")
print(f"DeepHistone-Enformer memory: ~{enformer_memory:.1f} MB")
print(f"Dual-Pathway memory        : ~{dual_pathway_memory:.1f} MB")

print("=" * 60)

# === Configuration comparison ===
print("\nModel Configurations:")
print("-" * 40)
print("DeepHistone (Original):")
print("  - Pure CNN architecture")
print("  - Single input pathway")

print("\nDeepHistone-Enformer:")
print(f"  - Channels: 1536")
print(f"  - Transformer layers: 10") 
print(f"  - Attention heads: 12")
print(f"  - Input channels: 5 (combined)")

print("\nDual-Pathway Model:")
print(f"  - Channels: 768")
print(f"  - Transformer layers: 4")
print(f"  - Attention heads: 8") 
print(f"  - DNA pathway: 4 input channels")
print(f"  - DNase pathway: 1 input channel")
print(f"  - Fusion type: concat")

print("=" * 60)