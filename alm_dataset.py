import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class ALMDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        # Tokenize the sentence
        encodings = self.tokenizer(sentence, truncation=True, max_length=self.max_length, padding='max_length')
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # Create labels (shifted input_ids)
        labels = input_ids[1:] + [self.tokenizer.eos_token_id]
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

# Example usage
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I went to the store to buy some groceries.",
    "The capital of France is Paris.",
    "She wrote a book about ancient history."
]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
dataset = ALMDataset(sentences, tokenizer)

# Get a sample
sample = dataset[0]
print("Input IDs:", sample['input_ids'])
print("Attention Mask:", sample['attention_mask'])
print("Labels:", sample['labels'])

# Decode the input_ids to see the sentence
sentence = tokenizer.decode(sample['input_ids'])
print("Sentence:", sentence)

# Demonstrate autoregressive property
for i in range(1, len(sample['input_ids'])):
    input_sequence = sample['input_ids'][:i]
    target = sample['labels'][i-1]
    print(f"Input: {tokenizer.decode(input_sequence)}")
    print(f"Target: {tokenizer.decode([target])}\n")