from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer

dataset = load_dataset("text", data_files='data/SumPTChainHalfTrain.txt')

ds = dataset["train"]
df = ds.to_pandas()

# Convert the DataFrame to a Hugging Face dataset
clean_dataset = Dataset.from_pandas(df)
raw_datasets = clean_dataset.train_test_split(test_size=0.1, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained('./PTtokenizer')


def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,  # Removing element longer that context size, no effect in JSB
        max_length=48,
        padding=False
    )
    return {"input_ids": outputs["input_ids"]}


# Create tokenized_dataset. We use map to pass each element of our dataset to tokenize and remove unnecessary columns.
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

print(tokenized_datasets)
