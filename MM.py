import json
import numpy as np
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
import difflib


def calculate_accuracy(generated, reference):
    gen_tokens = generated.split()
    ref_tokens = reference.split()
    correct = sum(g == r for g, r in zip(gen_tokens, ref_tokens))
    return correct / len(gen_tokens)  # Since gen_tokens and ref_tokens are of the same length


# Define BLEU and LCSS calculation functions
def calculate_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([word_tokenize(reference)], word_tokenize(candidate), smoothing_function=smoothie)


def calculate_lcss(seq1, seq2):
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    match = matcher.find_longest_match(0, len(seq1), 0, len(seq2))
    lcss_length = match.size
    max_length = max(len(seq1), len(seq2))
    return lcss_length / max_length if max_length > 0 else 0


def validate_model(model, tokenizer, dataset, initial_length=75, generate_length=144, top_k=5, top_p=0.95,
                   num_samples=None):
    bleu_scores = []
    lcss_scores = []
    accuracies = []

    # Use the entire dataset if num_samples is None or specify a smaller number for testing
    num_samples = min(num_samples if num_samples is not None else len(dataset), len(dataset))

    for i in range(num_samples):
        initial_tokens = tokenizer.decode(dataset[i][:initial_length])
        ground_truth = tokenizer.decode(dataset[i][initial_length:initial_length + generate_length])
        generated_text = model.generate_text(tokenizer, initial_tokens, generate_length, top_k=top_k, top_p=top_p)

        bleu_score = calculate_bleu(ground_truth, generated_text)
        lcss_score = calculate_lcss(ground_truth, generated_text)
        accuracy = calculate_accuracy(generated_text, ground_truth)
        bleu_scores.append(bleu_score)
        lcss_scores.append(lcss_score)
        accuracies.append(accuracy)

    average_bleu = np.mean(bleu_scores)
    average_lcss = np.mean(lcss_scores)
    average_accuracy = np.mean(accuracies)
    print(
        f"Validation BLEU Score: {average_bleu:.4f}, LCSS Score: {average_lcss:.4f}, Accuracy: {average_accuracy:.4f}")


class JSONTokenizer:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        self.vocab = data['model']['vocab']
        self.token_to_id = {token: int(id) for token, id in self.vocab.items()}
        self.id_to_token = {int(id): token for token, id in self.vocab.items()}

    def encode(self, text):
        return [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in text.split()]

    def decode(self, token_ids):
        return ' '.join(self.id_to_token.get(id, '[UNK]') for id in token_ids)


class SimpleDataset(Dataset):
    def __init__(self, file_path, tokenizer, n_rows=None):
        self.tokenizer = tokenizer
        self.sequences = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if n_rows is not None and i >= n_rows:
                    break
                encoded_line = self.tokenizer.encode(line.strip())
                if len(encoded_line) > 1:
                    self.sequences.append(encoded_line)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class MarkovModel:
    def __init__(self):
        self.transitions = {}

    def train(self, dataset, num_samples=None):
        num_samples = num_samples if num_samples is not None else len(dataset)
        for sequence in dataset[:num_samples]:
            for i in range(3, len(sequence) - 1):
                context = tuple(sequence[i - 3:i + 1])
                next_token = sequence[i + 1]
                if context not in self.transitions:
                    self.transitions[context] = {}
                if next_token not in self.transitions[context]:
                    self.transitions[context][next_token] = 0
                self.transitions[context][next_token] += 1

        # Convert counts to probabilities
        for context in self.transitions:
            total_transitions = sum(self.transitions[context].values())
            for next_token in self.transitions[context]:
                self.transitions[context][next_token] /= total_transitions

    def generate_text(self, tokenizer, initial_tokens, length, top_k=5, top_p=0.95):
        tokens = tokenizer.encode(initial_tokens)
        generated_tokens = []

        for _ in range(length):
            if len(tokens) >= 4:
                context = tuple(tokens[-4:])
            else:
                break  # Break if we don't have enough context
            next_token = self.sample_next(context, top_k=top_k, top_p=top_p)
            generated_tokens.append(next_token)
            tokens.append(next_token)

        generated_text = tokenizer.decode(generated_tokens)
        return generated_text

    def sample_next(self, context, top_k=5, top_p=0.95):
        if context not in self.transitions or not self.transitions[context]:
            return context[-1]  # Fall back to the last token if no transitions are available

        next_tokens, counts = zip(*self.transitions[context].items())
        probabilities = np.array(counts, dtype=np.float32)
        probabilities /= probabilities.sum()

        if top_k is not None:
            indices = np.argsort(probabilities)[::-1][:top_k]
            probabilities = probabilities[indices]
            next_tokens = np.array(next_tokens)[indices]

        if top_p is not None:
            cumulative_probabilities = np.cumsum(probabilities)
            indices = np.where(cumulative_probabilities > top_p)[0]
            if indices.size > 0:
                cutoff_index = indices[0]
                probabilities = probabilities[:cutoff_index + 1]
                next_tokens = next_tokens[:cutoff_index + 1]

        probabilities /= probabilities.sum()
        chosen_index = np.random.choice(len(next_tokens), p=probabilities)
        return next_tokens[chosen_index]


if __name__ == "__main__":
    json_tokenizer_path = '/home/zhangky/Documents/ZhangKY/Tokenizer/trip_chain_tokenizer_Tokyo_attr_loc_mode.json'
    train_file_path = '/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Train.txt'
    val_file_path = '/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Eval.txt'
    tokenizer = JSONTokenizer(json_tokenizer_path)
    train_dataset = SimpleDataset(train_file_path, tokenizer)
    val_dataset = SimpleDataset(val_file_path, tokenizer)

    model = MarkovModel()
    model.train(train_dataset, num_samples=250000)
    validate_model(model, tokenizer, val_dataset, num_samples=25000, generate_length=144)
