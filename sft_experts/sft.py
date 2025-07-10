# sft.py

import argparse
import os
import datetime

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import (
    SFTTrainer,
    SFTConfig,
    DataCollatorForCompletionOnlyLM,
    clone_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from peft import LoraConfig, AutoPeftModelForCausalLM
from accelerate import Accelerator
from accelerate.state import PartialState

import wandb

from sft_formatting import (  # format functions to convert datasets to prompt-completion
    map_mathinstruct_to_prompt_completion,
    map_pythonalpaca_to_prompt_completion,
    map_piqa_to_prompt_completion,
    map_pubmedqa_to_prompt_completion,
    map_aya_to_prompt_completion,
    map_aya_with_language_to_prompt_completion,
    map_alpaca_to_prompt_completion,
    map_copa_to_prompt_completion,
    map_socialiqa_to_prompt_completion,
    map_ecare_causal_reasoning_to_prompt_completion,
    map_ecare_explanation_generation_to_prompt_completion,
    map_moleculeqa_to_prompt_completion,
    map_casehold_to_prompt_completion,
    map_finqa_to_prompt_completion
)

def main():
    parser = argparse.ArgumentParser(description="MoE experts with SFT.")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./hf_cache")

    # Dataset
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--data_files", type=str, default=None)

    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, default="./sft_output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_liger", action="store_true")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--completion_only_loss", action="store_true") # NOTE: not compatible with packing!
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--neftune_noise_alpha", type=float, default=None)
    parser.add_argument("--activation_offloading", action="store_true")

    # Optimizer and scheduler
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # LoRA
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_rslora", action="store_true")

    # Other settings
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--instruction_format",
        type=str,
        choices=[
            "mathinstruct",
            "pythonalpaca",
            "piqa",
            "pubmedqa",
            "aya",
            "aya_with_language",
            "alpaca",
            # "copa",
            "socialiqa",
            "e-care_causal_reasoning",
            "e-care_explanation_generation",
            "moleculeqa",
            "casehold",
            "finqa"
        ],
        required=True,
    )

    args = parser.parse_args()

    if args.packing and args.completion_only_loss:
        raise ValueError("Cannot use --packing together with --completion_only_loss.")

    accelerator = Accelerator()
    accelerator.print(f"Training config:\n{args}\n")

    # Initialize wandb
    project_name = os.environ.get("WANDB_PROJECT", "default_project_name")
    run_name = f"{project_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if accelerator.is_main_process:
        wandb.init(
            project=project_name,
            name=run_name,
            # tags=["math", "sft", "qwen2"],
            mode="offline",
        )

    accelerator.print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if args.packing else "right"
    tokenizer.truncation_side = "right"
    # if tokenizer.chat_template is None: # set default chat template if needed
    #     model, tokenizer = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")

    PartialState().wait_for_everyone()

    # from transformers import BitsAndBytesConfig
    # quantization_config = None
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        local_files_only=True,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
    )
    model.config.attn_implementation = "flash_attention_2"
    model.config.use_cache = False if args.gradient_checkpointing else True

    # if args.completion_only_loss:
    #     accelerator.print("Using DataCollatorForCompletionOnlyLM (completions only)")
    #     collator = DataCollatorForCompletionOnlyLM(
    #         tokenizer=tokenizer,
    #         mlm=False,
    #         response_template="### Response:\n",
    #         pad_to_multiple_of=8, # needed?
    #     )
    # else:
    #     accelerator.print("Using classic causal LM DataCollator (full loss)")
    #     collator = DataCollatorForLanguageModeling(
    #         tokenizer=tokenizer,
    #         mlm=False,
    #     )

    PartialState().wait_for_everyone()

    accelerator.print("Loading dataset...")
    if args.data_files:
        import json
        data_files = json.loads(args.data_files)
        dataset = load_dataset(
            path=args.dataset_name,
            data_files=data_files,
            cache_dir=args.cache_dir
        )
    else:
        dataset = load_dataset(
            path=args.dataset_name,
            name=args.dataset_config_name,
            cache_dir=args.cache_dir
        )
    map_functions = {
        "mathinstruct": map_mathinstruct_to_prompt_completion,
        "pythonalpaca": map_pythonalpaca_to_prompt_completion,
        "piqa": map_piqa_to_prompt_completion,
        "pubmedqa": map_pubmedqa_to_prompt_completion,
        "aya": map_aya_to_prompt_completion,
        "aya_with_language": map_aya_with_language_to_prompt_completion,
        "alpaca": map_alpaca_to_prompt_completion,
        "copa": map_copa_to_prompt_completion,
        "socialiqa": map_socialiqa_to_prompt_completion,
        "e-care_causal_reasoning": map_ecare_causal_reasoning_to_prompt_completion,
        "e-care_explanation_generation": map_ecare_explanation_generation_to_prompt_completion,
        "moleculeqa": map_moleculeqa_to_prompt_completion,
        "casehold": map_casehold_to_prompt_completion,
        "finqa": map_finqa_to_prompt_completion
    }
    mapping_fn = map_functions[args.instruction_format] # convert dataset to prompt-completion format
    train_dataset = dataset[args.train_split].map(mapping_fn)
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in ["prompt", "completion"]])
    if args.eval_split:
        eval_dataset = dataset[args.eval_split].map(mapping_fn)
        eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c not in ["prompt", "completion"]])
    else:
        eval_dataset = None

    def concat_prompt_completion(example): # convert dataset to language modeling format (text)
        text = example["prompt"] + example["completion"]
        if "### Response:\n" not in text:
            accelerator.print("[WARNING] Response template missing in example.")
        return {"text": text}

    # train_dataset = train_dataset.map(
    #     concat_prompt_completion,
    #     remove_columns=["prompt", "completion"]
    # )
    # if eval_dataset is not None:
    #     eval_dataset = eval_dataset.map(
    #         concat_prompt_completion,
    #         remove_columns=["prompt", "completion"]
    #     )

    PartialState().wait_for_everyone()

    peft_cfg = None
    if args.use_peft:
        accelerator.print("Using LoRA")
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=args.use_rslora,
            # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        use_liger_kernel=args.use_liger,
        eval_strategy="steps" if eval_dataset else "no",
        push_to_hub=False,
        optim=args.optim,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        packing=args.packing,
        completion_only_loss=args.completion_only_loss,
        max_length=args.max_length,
        neftune_noise_alpha=args.neftune_noise_alpha,
        activation_offloading=args.activation_offloading,
        # eos_token="<|im_end|>",  # conversational
    )
    accelerator.print(f"SFT config:\n{sft_config}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        # data_collator=collator,
        processing_class=tokenizer,
        peft_config=peft_cfg,
        # formatting_func=lambda examples: [
        #     ex["prompt"] + ex["completion"] for ex in examples
        # ],
    )
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    accelerator.print(f"\nTrainable parameters: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")

    PartialState().wait_for_everyone()

    accelerator.print("Starting training...")
    if args.resume_from_checkpoint:
        accelerator.print(f"Starting training from checkpoint {args.resume_from_checkpoint}...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    # TODO: load from last checkpoint found

    trainer.accelerator.wait_for_everyone()  # save model on main process

    accelerator.print("Training complete. Saving model...")

    trainer.save_model(args.output_dir)
    accelerator.print(f"Model saved to {args.output_dir}")
    if args.push_to_hub:
        accelerator.print("Pushing to hub...")
        trainer.push_to_hub()

    trainer.accelerator.wait_for_everyone()

    if args.use_peft:
        accelerator.print("Merging LoRA adapters into base model...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(
            args.output_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        accelerator.print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
