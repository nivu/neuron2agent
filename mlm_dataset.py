import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class MLMDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128, mask_probability=0.15):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        # Tokenize the sentence
        encoding = self.tokenizer(sentence, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        
        # Create mask for tokens that can be masked
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
        mask_candidates = torch.tensor([True if mask == 0 else False for mask in special_tokens_mask])
        
        # Randomly select tokens to mask
        masked_indices = torch.rand(input_ids.shape) < self.mask_probability
        masked_indices = masked_indices & mask_candidates & (attention_mask.bool())
        
        # Set labels for non-masked tokens to -100 (ignore index)
        labels[~masked_indices] = -100
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Example usage
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I went to the store to buy some groceries.",
    "The capital of France is Paris.",
    "She wrote a book about ancient history."
]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = MLMDataset(sentences, tokenizer)

# Get a sample
sample = dataset[0]
print("Input IDs:", sample['input_ids'])
print("Attention Mask:", sample['attention_mask'])
print("Labels:", sample['labels'])

# Decode the input_ids to see the masked sentence
masked_sentence = tokenizer.decode(sample['input_ids'].tolist())
print("Masked sentence:", masked_sentence)

# Decode the non-masked parts of the labels
original_tokens = [token if label != -100 else '[MASK]' for token, label in zip(sample['input_ids'].tolist(), sample['labels'].tolist())]
original_sentence = tokenizer.decode(original_tokens)
print("Original sentence with [MASK]:", original_sentence)