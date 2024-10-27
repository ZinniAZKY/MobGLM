import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import json
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)


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
    def __init__(self, file_path, tokenizer, sequence_length=39, predict_length=180, n_rows=None):
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


class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        # Note the addition of 'device' parameter
        return (torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device))


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.fc(output[:, -1, :])
        return torch.sigmoid(logits)


def train_generator(gen, dis, optimizer_g, inputs, hidden):
    optimizer_g.zero_grad()
    logits, _ = gen(inputs, hidden)
    probs = torch.softmax(logits, dim=-1)
    m = Categorical(probs)
    actions = m.sample()
    log_probs = m.log_prob(actions)

    # Detach actions to avoid backprop through discriminator during generator update
    rewards = dis(actions.detach().long())
    rewards = rewards.squeeze()

    # Broadcast rewards to match log_probs shape
    rewards = rewards.unsqueeze(1).expand_as(log_probs)

    loss = -torch.mean(log_probs * rewards)
    loss.backward()
    optimizer_g.step()
    return loss.item()


def train_discriminator(dis, optimizer_d, real_data, fake_data, device):
    optimizer_d.zero_grad()
    real_preds = dis(real_data.to(device)).squeeze()
    real_loss = nn.BCEWithLogitsLoss()(real_preds, torch.ones_like(real_preds))

    fake_preds = dis(fake_data.to(device)).squeeze()
    fake_loss = nn.BCEWithLogitsLoss()(fake_preds, torch.zeros_like(fake_preds))

    total_loss = real_loss + fake_loss
    total_loss.backward()
    optimizer_d.step()
    return total_loss.item()


def validate_model(generator, discriminator, val_loader, device):
    generator.eval()
    discriminator.eval()
    total_gen_loss = 0
    total_disc_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = generator.init_hidden(inputs.size(0), device)

            # Generator's forward pass to get logits and actions
            logits, _ = generator(inputs, hidden)
            probs = torch.softmax(logits, dim=-1)
            m = Categorical(probs)
            actions = m.sample().long()  # Ensure actions are long
            log_probs = m.log_prob(actions)  # Log probabilities of actions  # Log probabilities of actions

            # Get rewards from the discriminator, ensure they're detached to avoid unwanted backprop
            rewards = discriminator(actions.detach()).squeeze()
            rewards = rewards.unsqueeze(1).expand(-1, logits.size(1))  # Expand rewards to match log_probs

            # Calculate generator loss
            gen_loss = -torch.mean(log_probs * rewards)
            total_gen_loss += gen_loss.item()

            # Discriminator losses on real and fake data
            real_preds = discriminator(targets).squeeze()
            real_loss = nn.BCEWithLogitsLoss()(real_preds, torch.ones_like(real_preds))

            fake_data = actions.detach().long()
            fake_preds = discriminator(fake_data).squeeze()
            fake_loss = nn.BCEWithLogitsLoss()(fake_preds, torch.zeros_like(fake_preds))

            disc_loss = real_loss + fake_loss
            total_disc_loss += disc_loss.item()

        avg_gen_loss = total_gen_loss / len(val_loader)
        avg_disc_loss = total_disc_loss / len(val_loader)

    return avg_gen_loss, avg_disc_loss


def train_epochs(generator, discriminator, train_loader, val_loader, optimizer_g, optimizer_d, epochs, device):
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = generator.init_hidden(inputs.size(0), device)

            # Update discriminator
            fake_logits, _ = generator(inputs, hidden)
            fake_probs = torch.softmax(fake_logits, dim=-1)
            m = Categorical(fake_probs)
            fake_data = m.sample()
            fake_data = fake_data.long().to(device)
            d_loss = train_discriminator(discriminator, optimizer_d, targets, fake_data, device)

            # Update generator
            g_loss = train_generator(generator, discriminator, optimizer_g, inputs, hidden)

        avg_gen_loss, avg_disc_loss = validate_model(generator, discriminator, val_loader, device)
        print(f'Epoch {epoch}, Train D loss: {d_loss}, G loss: {g_loss}, Val Gen Loss: {avg_gen_loss}, Val Disc Loss: {avg_disc_loss}')

    return generator, discriminator


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    json_tokenizer_path = '/home/zhangky/Documents/ZhangKY/Tokenizer/trip_chain_tokenizer_Tokyo_attr_loc_mode.json'
    train_file_path = '/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Train.txt'
    val_file_path = '/home/zhangky/Documents/ZhangKY/TokyoPT/Tokyo2008PTChain_Mode_6_24_Eval.txt'
    device = get_device()
    tokenizer = JSONTokenizer(json_tokenizer_path)
    vocab_size = len(tokenizer.token_to_id)
    embedding_dim = 64
    hidden_dim = 128

    generator = Generator(vocab_size, embedding_dim, hidden_dim)
    discriminator = Discriminator(vocab_size, embedding_dim, hidden_dim)

    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)

    train_dataset = TextDataset(train_file_path, tokenizer, n_rows=25000)
    val_dataset = TextDataset(val_file_path, tokenizer, n_rows=2500)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    train_epochs(generator, discriminator, train_loader, val_loader, optimizer_g, optimizer_d, 50, device)
