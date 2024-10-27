import os
import argparse
import json
import wandb
import random
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling, AutoTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer, TransfoXLConfig, get_linear_schedule_with_warmup
from utils import calculate_lcss
from tokenizer import TextDataset
import difflib
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRUST_REMOTE_CODE"] = "True"
wandb.init(project='MobilityGPT-Act')


def calculate_bleu_nltk(reference_text, candidate_text):
    # Tokenize the reference and candidate texts
    reference_tokens = [word_tokenize(ref.lower()) for ref in reference_text]
    candidate_tokens = word_tokenize(candidate_text.lower())
    smoothie = SmoothingFunction().method4

    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)


def parse_arguments():
    parser = argparse.ArgumentParser(description="MobiGPT-Act")
    parser.add_argument('--model_name', default='MobiGPT-Act_large', type=str)
    parser.add_argument('--n_layer', default=2, type=int)
    parser.add_argument('--n_head', default=2, type=int)
    parser.add_argument('--n_emb', default=32, type=int)
    # Need change
    parser.add_argument('--seq_length', default=219, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_decay', default=0.01, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--train_file', default='/home/zhangky/Documents/ZhangKY/TokyoPT'
                                                '/Tokyo2008PTChain_Mode_6_24_Train.txt', type=str)
    parser.add_argument('--val_file', default='/home/zhangky/Documents/ZhangKY/TokyoPT'
                                              '/Tokyo2008PTChain_Mode_6_24_Eval.txt', type=str)
    parser.add_argument('--enhance_file', default='/home/zhangky/Documents/ZhangKY/TokyoPT'
                                                  '/TokyoShopEnhanceWeekday0509.txt', type=str)
    return parser.parse_args()


def load_datasets(tokenizer, train_file, val_file, enhance_file):
    train_dataset = TextDataset(train_file, tokenizer, n_rows=25000)
    eval_dataset = TextDataset(val_file, tokenizer, n_rows=2500)
    # enhance_dataset = TextDataset(enhance_file, tokenizer)
    # combined_train_dataset = torch.utils.data.ConcatDataset([train_dataset, enhance_dataset])

    return train_dataset, eval_dataset


def initialize_model_and_training_args(tokenizer, args):
    # Use the passed tokenizer instead of initializing a new one
    config = TransfoXLConfig.from_pretrained(
        "transfo-xl-wt103",
        vocab_size=len(tokenizer),
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.n_emb,  # Note: d_model is used instead of n_embd for Transformer XL
    )

    model = TransfoXLLMHeadModel(config)
    model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        f"/home/zhangky/PycharmProjects/pythonProject/MobilityGPT-main/Outputs/{args.model_name}",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=args.lr_decay,
        num_train_epochs=args.num_epochs,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        save_total_limit=100,
        save_strategy='epoch',
        save_steps=3,
        report_to=["wandb"],
        logging_steps=100,
        do_eval=True,
        eval_steps=1,
        load_best_model_at_end=True
    )

    return model, training_args


def calculate_token_level_accuracy(generated_seqs, original_seqs, tokenizer):
    gen_tokens = [tokenizer.tokenize(seq) for seq in generated_seqs]
    ori_tokens = [tokenizer.tokenize(seq) for seq in original_seqs]

    correct_tokens = 0
    total_tokens = 0

    for gen, orig in zip(gen_tokens, ori_tokens):
        # Ensure we compare up to the shortest sequence length
        length = min(len(gen), len(orig))
        correct_tokens += sum(1 for i in range(length) if gen[i] == orig[i])
        total_tokens += length

    if total_tokens == 0:
        return 0
    else:
        return correct_tokens / total_tokens


def calculate_token_level_bleu(candidate_text, reference_text):
    candidate_tokens = word_tokenize(candidate_text.lower())
    reference_tokens = [word_tokenize(reference_text.lower())]

    return sentence_bleu(candidate_tokens, reference_tokens)


class GenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, prompt_text="Male 35 Office_Worker", num_return_sequences=5, n_steps=100):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text
        self.num_return_sequences = num_return_sequences
        self.generate_every_n_steps = n_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Ref: Neutral Text Generation with unlikelihood training, 2019
        if state.global_step % self.generate_every_n_steps == 0:
            input_ids = self.tokenizer.encode(self.prompt_text, return_tensors="pt").to(args.device)
            sample_outputs = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=219,
                top_k=25,
                top_p=0.9,
                num_return_sequences=self.num_return_sequences
            )

            generated_texts = [self.tokenizer.decode(sample_output, skip_special_tokens=True)
                               for sample_output in sample_outputs]

            wandb.log({"generated_samples": wandb.Table(data=[[text] for text in generated_texts], columns=["Text"])},
                      step=state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        input_ids = self.tokenizer.encode(self.prompt_text, return_tensors='pt').to(args.device)
        sample_outputs = self.model.generate(
            input_ids,
            do_sample=True,
            # Need change
            max_length=219,
            top_k=25,
            top_p=0.9,
            num_return_sequences=self.num_return_sequences
        )

        generated_sequences = [self.tokenizer.decode(sample_output, skip_special_tokens=True)
                               for sample_output in sample_outputs]

        wandb.log({"generated_samples": wandb.Table(data=[[text] for text in generated_sequences], columns=["Text"])},
                  step=state.global_step)


class CustomTrainer(Trainer):

    def __init__(self, *args, num_tokens_to_generate_from=39, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tokens_to_generate_from = num_tokens_to_generate_from

    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only=None,
            ignore_keys=None,
            metric_key_prefix='eval'
    ):
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )

        accuracies = []
        bleu_scores = []
        lcss_scores = []
        comparison_data = []

        for batch in dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            input_ids = batch['input_ids'][:, :self.num_tokens_to_generate_from]

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=219,
                    top_k=25,
                    top_p=0.9,
                )

            generated_seqs = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            original_seqs = [self.tokenizer.decode(o, skip_special_tokens=True) for o in batch['input_ids']]

            accuracy = calculate_token_level_accuracy(generated_seqs, original_seqs, self.tokenizer)
            accuracies.append(accuracy)

            for gen_seq, orig_seq in zip(generated_seqs, original_seqs):
                comparison_data.append([gen_seq, orig_seq])
                bleu_score = calculate_bleu_nltk([orig_seq], gen_seq)
                bleu_scores.append(bleu_score)
                lcss_scores.append(calculate_lcss(gen_seq, orig_seq))

        # Log the average of the collected metrics
        avg_bleu_score = np.mean(bleu_scores)
        avg_lcss_score = np.mean(lcss_scores)
        avg_accuracy = np.mean(accuracies)

        if wandb.run is not None:
            wandb.log({
                'avg_bleu_score': avg_bleu_score,
                'avg_lcss_score': avg_lcss_score,
                'avg_accuracy': avg_accuracy,
                'comparison_data': wandb.Table(data=comparison_data, columns=["Generated Text", "Original Text"])
            }, step=self.state.global_step)

        return eval_output


def main():
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    train_dataset, eval_dataset = load_datasets(tokenizer, args.train_file, args.val_file, args.enhance_file)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
    model, training_args = initialize_model_and_training_args(tokenizer, args)

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_tokens_to_generate_from=39,
    )

    trainer_state_path = f'{args.model_name}/trainer_state.json'
    if os.path.isfile(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        epoch = trainer_state['epoch']
        args.num_epochs += epoch

    os.environ['WANDB_WATCH'] = 'all'

    train_dataloader = trainer.get_train_dataloader()
    num_train_steps = len(train_dataloader) * args.num_epochs
    trainer.create_optimizer_and_scheduler(num_train_steps)
    trainer.lr_scheduler = get_linear_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    try:
        trainer.train(resume_from_checkpoint=args.model_name if os.path.isfile(trainer_state_path) else None)
    except Exception as e:
        print(f"Error: {e}")
        trainer.train()


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable Tensor Float 32 for matrix multiplications
    torch.backends.cudnn.allow_tf32 = True  # Enable Tensor Float 32 for cuDNN operations

    main()
