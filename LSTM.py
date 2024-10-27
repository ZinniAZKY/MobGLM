import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
import difflib
from rouge import Rouge
from fastdtw import fastdtw

torch.manual_seed(42)


def calculate_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([word_tokenize(reference)], word_tokenize(candidate), smoothing_function=smoothie)


def calculate_lcss(seq1, seq2):
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    match = matcher.find_longest_match(0, len(seq1), 0, len(seq2))
    lcss_length = match.size
    max_length = max(len(seq1), len(seq2))
    return lcss_length / max_length if max_length > 0 else 0


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


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, sequence_length=75, predict_length=144, n_rows=None):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.predict_length = predict_length
        self.pairs = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if n_rows is not None and i >= n_rows:
                    break
                encoded_line = self.tokenizer.encode(line.strip())
                if len(encoded_line) >= sequence_length + predict_length:
                    input_seq = encoded_line[:sequence_length]
                    target_seq = encoded_line[sequence_length:sequence_length + predict_length]
                    self.pairs.append((input_seq, target_seq))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target_seq = self.pairs[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        predictions = [self.fc(last_output.unsqueeze(1))]
        current_token_ids = last_output.argmax(-1)
        current_input = self.embedding(current_token_ids).unsqueeze(1)

        for _ in range(143):  # Generate 179 additional tokens to get a total of 180
            output, hidden = self.lstm(current_input, hidden)
            output = self.fc(output)
            predictions.append(output)

            current_token_ids = output.argmax(-1)
            if current_token_ids.ndim == 1:
                current_input = self.embedding(current_token_ids).unsqueeze(1)
            else:
                current_input = self.embedding(current_token_ids.squeeze()).unsqueeze(1)

        predictions = torch.cat(predictions, dim=1)
        return predictions

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states with zeros
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        return hidden


def train_and_validate(model, train_dataset, val_dataset, batch_size, num_epochs, lr, device):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            _, predicted = outputs.max(2)
            total_train_accuracy += (predicted == targets).float().mean().item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        bleu_scores = []
        lcss_scores = []
        rouge = Rouge()
        dtw_scores = []
        rouge_scores = []

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets)
                total_val_loss += loss.item()
                _, predicted = outputs.max(2)
                total_val_accuracy += (predicted == targets).float().mean().item()

                predicted_texts = [tokenizer.decode(output.argmax(-1).cpu().numpy()) for output in outputs]
                target_texts = [tokenizer.decode(target.cpu().numpy()) for target in targets]

                # Collect token IDs for DTW (if directly available)
                predicted_ids = [output.argmax(-1).cpu().numpy() for output in outputs]
                target_ids = [target.cpu().numpy() for target in targets]

                for pred_text, target_text, pred_id, target_id in zip(predicted_texts, target_texts, predicted_ids,
                                                                      target_ids):
                    bleu_scores.append(calculate_bleu(target_text, pred_text))
                    lcss_scores.append(calculate_lcss(target_text, pred_text))
                    rouge_result = rouge.get_scores(pred_text, target_text)[0]
                    rouge_scores.append(rouge_result['rouge-l']['f'])
                    dtw_distance, _ = fastdtw(pred_id, target_id, dist=2)
                    dtw_scores.append(dtw_distance)

            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_accuracy = total_val_accuracy / len(val_dataloader)
            avg_bleu_score = np.mean(bleu_scores)
            avg_lcss_score = np.mean(lcss_scores)
            avg_rouge_score = np.mean(rouge_scores)
            avg_dtw_score = np.mean(dtw_scores)

            print(
                f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {avg_train_accuracy:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {avg_val_accuracy:.4f}, Avg BLEU = {avg_bleu_score:.4f}, Avg LCSS = {avg_lcss_score:.4f}, Avg ROUGE-L = {avg_rouge_score:.4f}, Avg DTW = {avg_dtw_score:.4f}')


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    json_tokenizer_path = '/home/zhangky/Documents/ZhangKY/Tokenizer/trip_chain_tokenizer_Tokyo_attr_loc_mode.json'
    tokenizer = JSONTokenizer(json_tokenizer_path)
    vocab_size = len(tokenizer.token_to_id)

    train_file_path = '/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Train.txt'
    val_file_path = '/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Eval.txt'
    train_dataset = TextDataset(train_file_path, tokenizer, sequence_length=75, predict_length=144, n_rows=250000)
    val_dataset = TextDataset(val_file_path, tokenizer, sequence_length=75, predict_length=144, n_rows=25000)

    device = get_device()
    model = LSTMModel(vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2).to(device)

    train_and_validate(model, train_dataset, val_dataset, batch_size=128, num_epochs=100, lr=1e-3, device=device)
