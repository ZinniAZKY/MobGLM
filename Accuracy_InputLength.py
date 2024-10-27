import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv

model_path = '/home/ubuntu/Desktop/Documents/2024_train/output_1/checkpoint-16680'
tokenizer_path = '/home/ubuntu/PycharmProjects/pythonProject/MobilityGPT/PTtokenizer'
val_file_path = '/home/ubuntu/Desktop/Documents/GISA2023/valseparate/tokyo_no,/eval_tokyo.txt'
accuracy_file_path = '/home/ubuntu/Desktop/Documents/2024_train/length_accuracy/accuracy0219_inputlenghth.csv'

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open(accuracy_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Input Length', 'Accuracy'])
    file.flush()

    for input_length in range(47, 0, -1):
        correct_predictions = 0
        total_samples = 0

        with open(val_file_path, 'r', encoding='utf-8') as val_file:
            for line in val_file:
                total_samples += 1
                if total_samples > 10000:
                    break
                tokens = tokenizer.encode(line.strip(), return_tensors='pt', max_length=48, padding='max_length',
                                          truncation=True)
                input_ids = tokens[:, :input_length].to(device)

                with torch.no_grad():
                    outputs = model.generate(input_ids, max_length=48, do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)

                predicted_ids = outputs[0, input_length:]
                actual_ids = tokens[0, input_length:].to(device)

                if torch.equal(predicted_ids, actual_ids):
                    correct_predictions += 1

        accuracy = correct_predictions / total_samples if total_samples else 0
        writer.writerow([input_length, accuracy])
        print(f'Input Length: {input_length}, Accuracy: {accuracy:.4f}')
