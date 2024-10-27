import pandas as pd
from sklearn.model_selection import train_test_split

# Read the text file
with open("/home/zhangky/Documents/ZhangKY/NationwidePT_Hierarchical/Toyama1999PTChain_Hierarchical_Fin.txt", 'r') as file:
    lines = file.read().splitlines()

df = pd.DataFrame(lines, columns=['text'])
df = df.sample(frac=1, random_state=42)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_df['text'].to_csv("/home/zhangky/Documents/ZhangKY/NationwidePT_Hierarchical/Toyama1999PTChain_Hierarchical_Fin_Train.txt", index=False, header=False)
val_df['text'].to_csv("/home/zhangky/Documents/ZhangKY/NationwidePT_Hierarchical/Toyama1999PTChain_Hierarchical_Fin_Eval.txt", index=False, header=False)

print(f"Training set length: {len(train_df)}")
print(f"Validation set length: {len(val_df)}")
