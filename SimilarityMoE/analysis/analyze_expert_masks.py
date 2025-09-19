import argparse
import os
import torch
import matplotlib.pyplot as plt

def make_parser():
    parser = argparse.ArgumentParser(description="MoE Mask analysis")
    parser.add_argument('--mask-dir',type=str, required=True,)
    parser.add_argument('--output-dir', type=str, default='plots', help='Directory to save plots')
    
    return parser

def plot_per_layer(experts_freq, step=None, output_dir='plots'):
    sorted_layers = sorted(experts_freq.keys(), key=lambda x: int(x.split('_')[1]))
    x = list(range(len(experts_freq)))
    num_experts = len(next(iter(experts_freq.values())))
    ys = [[] for _ in range(num_experts)]
    for layer in sorted_layers:
        freqs = experts_freq[layer]
        for i in range(num_experts):
            ys[i].append(freqs[i])
    plt.figure(figsize=(14, 6))
    for i in range(num_experts):
        plt.plot(x, ys[i], label=f'Expert {i}', marker='o')

    plt.xlabel('Layer')
    plt.ylabel('Activation Frequency')
    plt.title('Expert Activation Frequency per Layer')
    plt.xticks(x, sorted_layers, rotation=45, fontsize=8)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'activation_frequency_per_layer{"_step_" + str(step) if step else ""}.png'))
    
def load_masks(mask_dir):
    masks = []
    for filename in os.listdir(mask_dir):
        print(f"Processing file: {filename}")
        item = {}
        if filename.endswith(".pt"):
            try:
                item['step'] = filename.split('_')[-1].replace('.pt', '')
                item['mask'] = torch.load(os.path.join(mask_dir, filename),map_location=torch.device('cpu'))
                masks.append(item)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    return masks

def per_layer_freq(mask):
    layers = len(mask)
    experts_freq = {f'Layer_{i}': [] for i in range(layers)}
    for idx, layer in enumerate(mask):
        experts_freq[f'Layer_{idx}'] = (layer.sum(dim=0)/layer.shape[0]).tolist()
    return experts_freq

def per_step_freq(masks):
    steps = len(masks)
    experts_freq = {f'Step_{i}': [] for i in range(steps)}

def compute_activation_frequency(masks):
    pass


        
        

def main():
    parser = make_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    masks = load_masks(args.mask_dir)
    if not masks:
        print("No masks found in the specified directory.")
        return
    print(f"Loaded {len(masks)} masks from {args.mask_dir}")
    
    for mask in masks:
        experts_freq = per_layer_freq(mask['mask'])
        plot_per_layer(experts_freq, step=mask['step'], output_dir=output_dir)

if __name__ == "__main__":
    main()