import matplotlib.pyplot as plt
import json

def main():
    PATH_TO_TRAINER_STATE="/leonardo_work/EUHPC_A06_067/moe_models/ddam_qwen3_moe-base_12_bal-mix/checkpoint-300/trainer_state.json"
    with open(PATH_TO_TRAINER_STATE, 'r') as f:
        trainer_state = json.load(f)
    history = trainer_state['log_history']
    losses = [entry['loss'] for entry in history if 'loss' in entry]
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss over Steps')
    plt.legend()
    plt.savefig('./trash/training_loss_plot.png')
    
if __name__ == "__main__":
    main()