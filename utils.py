import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LogitsProcessor

import nltk
from nltk.translate.bleu_score import sentence_bleu
import difflib


def generate_activities(model, input_ids, excluded_token_ids, max_length=75, do_sample=True, top_k=10, top_p=0.9,
                        temperature=0.8):
    # input_ids = tokenizer.encode(initial_input, return_tensors='pt')

    generated_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=25,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1,
        logits_processor=[torch.nn.LogSoftmax(dim=-1), CustomLogitsProcessor(excluded_token_ids)]
    )

    return generated_ids


class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, excluded_token_ids):
        self.excluded_token_ids = excluded_token_ids

    def __call__(self, input_ids, scores):
        # Set the logits for excluded tokens to -inf, so their probabilities become 0
        scores[:, self.excluded_token_ids] = float('-inf')
        return scores


def calculate_bleu(reference_texts, candidate_text):
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference_texts]
    candidate_tokens = nltk.word_tokenize(candidate_text.lower())
    return sentence_bleu(reference_tokens, candidate_tokens)


def calculate_lcss(seq1, seq2):
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    match = matcher.find_longest_match(0, len(seq1), 0, len(seq2))
    lcss_length = match.size
    max_length = max(len(seq1), len(seq2))
    return lcss_length / max_length if max_length > 0 else 0
