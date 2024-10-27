import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import pandas as pd
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = '/home/ubuntu/Desktop/Documents/2024_train/output_1/checkpoint-16680'  
tokenizer_path = '/home/ubuntu/PycharmProjects/pythonProject/MobilityGPT/PTtokenizer'  
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()


def calculate_accuracy(data_file, tokenizer, model, num_lines=10000):
    correct_predictions = {i: 0 for i in range(1, 48)}
    total_lines = 0

    with open(data_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            if line_number >= num_lines:
                break
            if line_number % 100 == 0:  # 每处理100行打印一次进度
                print(f"Processing line {line_number + 1}/{num_lines}")
            tokens = tokenizer.encode(line.strip(), return_tensors='pt').to(device)
            if len(tokens[0]) < 48:  # 确保每行至少有48个token
                continue
            total_lines += 1
            original_tokens = tokens.tolist()[0]

            for input_length in range(1, 48):
                input_tokens = tokens[:, :input_length]
                with torch.no_grad():
                    outputs = model.generate(input_tokens, max_length=48, pad_token_id=tokenizer.eos_token_id)

                generated_tokens = outputs[0][input_length:].tolist()
                correct_tokens = original_tokens[input_length:48]

                accuracy_count = sum([1 for i, j in zip(generated_tokens, correct_tokens) if i == j])
                correct_predictions[input_length] += accuracy_count / len(correct_tokens)

    
    accuracies = {length: correct / total_lines for length, correct in correct_predictions.items()}
    for input_length, accuracy in accuracies.items():
        print(f"Input Length: {input_length}, Average Accuracy: {accuracy:.4f}")

    return accuracies



accuracies = calculate_accuracy('/home/ubuntu/Desktop/Documents/GISA2023/valseparate/tokyo_no,/eval_tokyo.txt', tokenizer, model)
df = pd.DataFrame(list(accuracies.items()), columns=['Input Length', 'Accuracy'])
df.to_csv('/home/ubuntu/Desktop/Documents/2024_train/length_accuracy/accuracy_by_input_length.csv', index=False)
