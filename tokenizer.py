import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from torch.utils.data import Dataset


class TextDataset(Dataset):
    # Block size must be changed to compare the LCSS and BLEU as long as the input dataset is changed
    def __init__(self, file_path, tokenizer, block_size=723, n_rows=None):
        self.tokenizer = tokenizer
        self.examples = []

        with open(file_path, encoding='utf-8') as f:
            i = 0
            for line in tqdm(f, desc="Loading Dataset"):
                if n_rows is not None and i > n_rows:
                    break
                input_ids = tokenizer.encode(line, add_special_tokens=True, truncation=True, max_length=block_size)
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = [1] * len(input_ids)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)

                self.examples.append({'input_ids': input_ids,
                                      'attention_mask': attention_mask,
                                      'labels': input_ids})

                i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


if __name__ == '__main__':
    tokenizer = Tokenizer.from_file(
        "/home/zhangky/Documents/ZhangKY/Tokenizer/trip_chain_tokenizer_hierarchical.json")

    # Wrap with PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]', 'unk_token': '[UNK]', 'mask_token': '[MASK]', 'additional_special_tokens': ['<region>', '<prefecture>', '<municipality>', '<small_zone>']})

    fast_tokenizer.save_pretrained("/home/zhangky/PycharmProjects/pythonProject/MobilityGPT-main/PTTokenizerHierarchical")
