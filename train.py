import os
import argparse
import json
import wandb
import random
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from utils import calculate_lcss
from tokenizer import TextDataset
import difflib
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction


torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
nltk.data.load('/home/zhangky/nltk_data/tokenizers/punkt/PY3/english.pickle')


def calculate_bleu_nltk(reference_text, candidate_text):
    # Tokenize the reference and candidate texts
    reference_tokens = [word_tokenize(ref.lower()) for ref in reference_text]
    candidate_tokens = word_tokenize(candidate_text.lower())
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)


def parse_arguments():
    parser = argparse.ArgumentParser(description="MobiGPT-Act")
    parser.add_argument('--model_name', default='MobiGPT-Act_large', type=str)
    parser.add_argument('--n_layer', default=4, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--n_emb', default=64, type=int)
    parser.add_argument('--seq_length', default=723, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--lr_decay', default=0.01, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--tokenizer',
                        default='/home/zhangky/PycharmProjects/pythonProject/MobilityGPT-main'
                                '/PTTokenizerHierarchical',
                        type=str)
    parser.add_argument('--train_file', default='/mnt/free/zhangky/NationwidePT_Hierarchical'
                                                '/Toyama1999PTChain_Hierarchical_Fin_Train.txt', type=str)
    parser.add_argument('--val_file', default='/mnt/free/zhangky/NationwidePT_Hierarchical'
                                              '/Toyama1999PTChain_Hierarchical_Fin_Eval.txt', type=str)
    # parser.add_argument('--enhance_file', default='/home/zhangky/Documents/ZhangKY/TokyoPT'
    #                                               '/TokyoShopEnhanceWeekday0509.txt', type=str)
    return parser.parse_args()


def load_datasets(tokenizer, train_file, val_file):
    train_dataset = TextDataset(train_file, tokenizer, n_rows=30000)
    eval_dataset = TextDataset(val_file, tokenizer, n_rows=3000)
    return train_dataset, eval_dataset


def initialize_model_and_training_args(tokenizer, args):
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_positions=args.seq_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_embd=args.n_emb,
    )

    model = GPT2LMHeadModel(config)
    model.to(args.device)

    training_args = TrainingArguments(
        f"/home/zhangky/PycharmProjects/pythonProject/MobilityGPT-main/Outputs/{args.model_name}",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=args.lr_decay,
        num_train_epochs=args.num_epochs,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        save_total_limit=3,
        save_strategy='epoch',
        save_steps=3,
        logging_steps=500,
        do_eval=True,
        eval_steps=1,
        load_best_model_at_end=True,
        report_to=[]
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


def save_comparison(comparison_data, epoch, output_dir):
    df = pd.DataFrame(comparison_data, columns=["Generated Text", "Original Text"])
    output_path = os.path.join(output_dir, f"comparison_data_epoch_{epoch}.csv")
    df.to_csv(output_path, index=False)


class CustomTrainer(Trainer):

    def __init__(self, *args, output_dir, num_tokens_to_generate_from=243, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tokens_to_generate_from = num_tokens_to_generate_from
        self.output_dir = output_dir

    def extract_and_save_attention(self, input_ids, epoch):
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            attentions = outputs.attentions  # List of tensors, shape: (num_layers, batch_size, num_heads, seq_len, seq_len)

        # Extract attention weights for the last layer
        last_layer_attention = attentions[-1].cpu().numpy()  # Shape: (batch_size, num_heads, seq_len, seq_len)
        average_last_layer_attention = np.mean(last_layer_attention, axis=1)  # Average over heads

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[12].tolist())
        token_labels = [self.tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in input_ids]

        attention_filename = os.path.join(self.output_dir, f"attention_weights_epoch_{epoch}_last_layer.txt")
        with open(attention_filename, 'w') as f:
            f.write("\t" + "\t".join(tokens) + "\n")
            for i, row in enumerate(average_last_layer_attention[12]):
                f.write(token_labels[12][i] + "\t" + "\t".join(map(str, row)) + "\n")

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
                    max_length=723,
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

            # Extract and save attention weights for the last layer
            self.extract_and_save_attention(input_ids, self.state.epoch)

        avg_bleu_score = np.mean(bleu_scores)
        avg_lcss_score = np.mean(lcss_scores)
        avg_accuracy = np.mean(accuracies)

        print(f"Epoch {self.state.epoch}:")
        print(f"  Average BLEU Score: {avg_bleu_score}")
        print(f"  Average LCSS Score: {avg_lcss_score}")
        print(f"  Average Accuracy: {avg_accuracy}")

        save_comparison(comparison_data, self.state.epoch, self.output_dir)

        return eval_output


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_dataset, eval_dataset = load_datasets(tokenizer, args.train_file, args.val_file)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model, training_args = initialize_model_and_training_args(tokenizer, args)

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=training_args.output_dir,
        num_tokens_to_generate_from=243,
    )

    trainer_state_path = f'{args.model_name}/trainer_state.json'
    if os.path.isfile(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        epoch = trainer_state['epoch']
        args.num_epochs += epoch

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
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    main()

# # print hourly distribution of evaluation data.
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# model_path = '/home/ubuntu/PycharmProjects/pythonProject/MobilityGPT/output/Kinki6-24Attr/checkpoint-19550'  # replace the checkpoint here to test different trained results
#
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = GPT2LMHeadModel.from_pretrained(model_path)
# model.to(device)
#
# eval_dataset = TextDataset('/home/ubuntu/PycharmProjects/pythonProject/MobilityGPT/data/Kinki2010PTChainEval.txt',
#                            tokenizer,
#                            n_rows=1000)  # replace the # of rows to generate and compare different amount of generations
#
# input_steps = 15  # replace the input step number from 1~12 or even longer to check the difference
# results = []
#
# for input in eval_dataset:
#     input = input['input_ids'].to(device)
#     input_ids = input[:input_steps].unsqueeze(0)
#     output = model.generate(input_ids, do_sample=True,
#                             max_length=75,
#                             top_k=25,
#                             top_p=0.9, )
#     results.append(output.squeeze(0))
# original_t = {j: defaultdict(int) for j in range(0, 75, 1)}  # step=4 means only count on the hour. 1~4 is OK
# generated_t = {j: defaultdict(int) for j in range(0, 75, 1)}
#
# for i in range(len(eval_dataset)):
#     input_ids = eval_dataset[i]['input_ids']
#     results_i = results[i]
#
#     for j in range(0, 75, 1):
#         original_token = tokenizer.decode(input_ids[j].item())
#         generated_token = tokenizer.decode(results_i[j].item())
#
#         original_t[j][original_token] += 1
#         generated_t[j][generated_token] += 1
#         print("finished")
#
# j_value = 24  # set the time here
#
# original_counts = original_t[j_value]
# generated_counts = generated_t[j_value]
#
# all_tokens = set(original_counts.keys()) | set(generated_counts.keys())
#
# original_frequencies = [original_counts.get(token, 0) for token in all_tokens]
# generated_frequencies = [generated_counts.get(token, 0) for token in all_tokens]
# tokens = list(all_tokens)
#
# sorted_indices = np.argsort(original_frequencies)[::-1]
# original_frequencies = [original_frequencies[i] for i in sorted_indices]
# generated_frequencies = [generated_frequencies[i] for i in sorted_indices]
# tokens = [tokens[i] for i in sorted_indices]
#
# # Directory for saving output data
# output_dir = '/home/ubuntu/PycharmProjects/pythonProject/MobilityGPT/output/Kinki6-24Attr/'
# os.makedirs(output_dir, exist_ok=True)
#
# # Files to save the sequences
# original_sequences_file = os.path.join(output_dir, 'original_sequences.txt')
# generated_sequences_file = os.path.join(output_dir, 'generated_sequences.txt')
#
# # Open files to write sequences
# with open(original_sequences_file, 'w', encoding='utf-8') as orig_file, \
#      open(generated_sequences_file, 'w', encoding='utf-8') as gen_file:
#
#     for i in range(len(eval_dataset)):
#         input_ids = eval_dataset[i]['input_ids']
#         results_i = results[i]
#
#         # Decode and save/write the original and generated sequences
#         original_sequence = tokenizer.decode(input_ids, skip_special_tokens=True)
#         generated_sequence = tokenizer.decode(results_i, skip_special_tokens=True)
#
#         orig_file.write(f"{original_sequence}\n")
#         gen_file.write(f"{generated_sequence}\n")
#
# x = np.arange(len(tokens))
# width = 0.35
#
# plt.figure(figsize=(12, 6))
# plt.bar(x - width / 2, original_frequencies, width, label='Original', alpha=0.7)
# plt.bar(x + width / 2, generated_frequencies, width, label='Generated')
# plt.xlabel('Tokens')
# plt.ylabel('Frequencies')
# plt.title(f'Token Frequencies at {int(6 + j_value / 4)}:00')
# plt.xticks(x, tokens, rotation=45)
# plt.legend()
# plt.tight_layout()
# # plt.savefig('/home/ubuntu/PycharmProjects/pythonProject/MobilityGPT/output/Tokyo6-24NoAttr/plot.png')


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_DISABLED"] = "true"
#
#
# def calculate_bleu_nltk(reference_text, candidate_text):
#     reference_tokens = [word_tokenize(ref.lower()) for ref in reference_text]
#     candidate_tokens = word_tokenize(candidate_text.lower())
#     smoothie = SmoothingFunction().method4
#     return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
#
#
# def parse_arguments():
#     parser = argparse.ArgumentParser(description="MobiGPT-Act")
#     parser.add_argument('--model_name', default='MobiGPT-Act_large', type=str)
#     parser.add_argument('--n_layer', default=12, type=int)
#     parser.add_argument('--n_head', default=12, type=int)
#     parser.add_argument('--n_emb', default=768, type=int)
#     # Need change
#     parser.add_argument('--seq_length', default=219, type=int)
#     parser.add_argument('--batch_size', default=128, type=int)
#     parser.add_argument('--num_epochs', default=50, type=int)
#     parser.add_argument('--lr', default=5e-6, type=float)
#     parser.add_argument('--lr_decay', default=0.01, type=float)
#     parser.add_argument('--device', default='cuda:0', type=str)
#
#     parser.add_argument('--tokenizer',
#                         default='/home/zhangky/PycharmProjects/pythonProject/MobilityGPT-main/PTTokenizerAttrAddressMode',
#                         type=str)
#     parser.add_argument('--val_file',
#                         default='/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Eval.txt', type=str)
#     return parser.parse_args()
#
#
# def load_dataset(tokenizer, val_file):
#     eval_dataset = TextDataset(val_file, tokenizer, n_rows=10000)
#     return eval_dataset
#
#
# def initialize_model_and_tokenizer(args):
#     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
#
#     config = AutoConfig.from_pretrained(
#         "gpt2",
#         vocab_size=len(tokenizer),
#         n_positions=args.seq_length,
#         n_layer=args.n_layer,
#         n_head=args.n_head,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         n_embd=args.n_emb,
#     )
#
#     model = GPT2LMHeadModel(config)
#     model.to(args.device)
#     return model, tokenizer
#
#
# def calculate_token_level_accuracy(generated_seqs, original_seqs, tokenizer):
#     gen_tokens = [tokenizer.tokenize(seq) for seq in generated_seqs]
#     ori_tokens = [tokenizer.tokenize(seq) for seq in original_seqs]
#
#     correct_tokens = 0
#     total_tokens = 0
#
#     for gen, orig in zip(gen_tokens, ori_tokens):
#         length = min(len(gen), len(orig))
#         correct_tokens += sum(1 for i in range(length) if gen[i] == orig[i])
#         total_tokens += length
#
#     if total_tokens == 0:
#         return 0
#     else:
#         return correct_tokens / total_tokens
#
#
# def save_comparison(comparison_data, output_dir):
#     df = pd.DataFrame(comparison_data, columns=["Generated Text", "Original Text"])
#     output_path = os.path.join(output_dir, f"comparison_data_original_model.csv")
#     df.to_csv(output_path, index=False)
#
#
# class CustomTrainer(Trainer):
#
#     def __init__(self, *args, output_dir, num_tokens_to_generate_from=75, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_tokens_to_generate_from = num_tokens_to_generate_from
#         self.output_dir = output_dir
#
#     def evaluation_loop(
#             self,
#             dataloader,
#             description,
#             prediction_loss_only=None,
#             ignore_keys=None,
#             metric_key_prefix='eval'
#     ):
#         eval_output = super().evaluation_loop(
#             dataloader,
#             description,
#             prediction_loss_only,
#             ignore_keys,
#             metric_key_prefix
#         )
#
#         accuracies = []
#         bleu_scores = []
#         lcss_scores = []
#         comparison_data = []
#         rouge = Rouge()
#         dtw_scores = []
#         rouge_scores = []
#
#         for batch in dataloader:
#             batch = {k: v.to(self.args.device) for k, v in batch.items()}
#             input_ids = batch['input_ids'][:, :self.num_tokens_to_generate_from]
#
#             with torch.no_grad():
#                 generated_ids = self.model.generate(
#                     input_ids,
#                     do_sample=True,
#                     max_length=219,
#                     top_k=25,
#                     top_p=0.9,
#                 )
#
#             generated_seqs = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
#             original_seqs = [self.tokenizer.decode(o, skip_special_tokens=True) for o in batch['input_ids']]
#
#             accuracy = calculate_token_level_accuracy(generated_seqs, original_seqs, self.tokenizer)
#             accuracies.append(accuracy)
#
#             for gen_seq, orig_seq in zip(generated_seqs, original_seqs):
#                 comparison_data.append([gen_seq, orig_seq])
#                 bleu_score = calculate_bleu_nltk([orig_seq], gen_seq)
#                 bleu_scores.append(bleu_score)
#                 lcss_scores.append(calculate_lcss(gen_seq, orig_seq))
#                 rouge_result = rouge.get_scores([gen_seq], [orig_seq])[0]
#                 rouge_scores.append(rouge_result['rouge-l']['f'])
#
#                 gen_ids = self.tokenizer.encode(gen_seq)
#                 orig_ids = self.tokenizer.encode(orig_seq)
#                 dtw_distance, _ = fastdtw(gen_ids, orig_ids, dist=2)
#                 dtw_scores.append(dtw_distance)
#
#         avg_bleu_score = np.mean(bleu_scores)
#         avg_lcss_score = np.mean(lcss_scores)
#         avg_accuracy = np.mean(accuracies)
#         avg_rouge_score = np.mean(rouge_scores)
#         avg_dtw_score = np.mean(dtw_scores)
#
#         print(f"Average BLEU Score: {avg_bleu_score}")
#         print(f"Average LCSS Score: {avg_lcss_score}")
#         print(f"Average Accuracy: {avg_accuracy}")
#         print(f"Rouge-L: {avg_rouge_score}")
#         print(f"DTW: {avg_dtw_score}")
#
#         save_comparison(comparison_data, self.output_dir)
#
#         return eval_output
#
#
# def main():
#     args = parse_arguments()
#
#     model, tokenizer = initialize_model_and_tokenizer(args)
#     eval_dataset = load_dataset(tokenizer, args.val_file)
#
#     data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
#
#     training_args = TrainingArguments(
#         output_dir=f"/home/zhangky/PycharmProjects/pythonProject/MobilityGPT-main/Outputs/{args.model_name}",
#         per_device_eval_batch_size=args.batch_size,
#         do_eval=True,
#         evaluation_strategy="epoch",
#         report_to=[],
#         logging_steps=1000
#     )
#
#     trainer = CustomTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         data_collator=data_collator,
#         eval_dataset=eval_dataset,
#         output_dir=training_args.output_dir,
#         num_tokens_to_generate_from=75,
#     )
#
#     trainer.evaluate()
#
#
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#
#     main()
