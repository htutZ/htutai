import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from collections import Counter

data_dir = "databases"

def load_data_from_json(category):
    file_path = os.path.join(data_dir, f"{category}.json")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data["queries"], data["responses"]

categories = ["greetings", "reminders", "time"]
all_queries, all_responses = [], []

for category in categories:
    queries, responses = load_data_from_json(category)
    all_queries.extend(queries)
    all_responses.extend(responses)

def tokenize_and_build_vocab(texts):
    tokens_list = [text.split() for text in texts]
    vocab = Counter(token for tokens in tokens_list for token in tokens)
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return tokens_list, vocab

tokenized_queries, vocab = tokenize_and_build_vocab(all_queries)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(vocab.items())}

def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

def pad_sequences(sequences, max_len):
    padded_sequences = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        end = min(len(seq), max_len)
        padded_sequences[i, :end] = seq[:end]
    return padded_sequences

indices_queries = [tokens_to_indices(tokens, vocab) for tokens in tokenized_queries]
max_seq_len = max(len(seq) for seq in indices_queries)
padded_queries = pad_sequences(indices_queries, max_seq_len)

response_to_idx = {response: idx for idx, response in enumerate(set(all_responses))}
idx_to_response = {idx: response for response, idx in response_to_idx.items()}
indices_responses = [response_to_idx[response] for response in all_responses]

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])
        return x

    def expand_output_layer(self, new_output_dim):
        self.fc = nn.Linear(self.hidden_dim, new_output_dim)

embedding_dim = 100
hidden_dim = 128
output_dim = len(response_to_idx)
model = LSTMModel(len(vocab) + 2, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

def interactive_training(user_query, user_response=None, correct=True):
    global output_dim, model, optimizer
    tokenized_query = user_query.split()
    query_indices = [vocab.get(token, vocab['<UNK>']) for token in tokenized_query]
    query_tensor = torch.tensor(pad_sequences([query_indices], max_seq_len), dtype=torch.long)

    if correct:
        model.eval()
        with torch.no_grad():
            output = model(query_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            bot_response = idx_to_response.get(predicted_idx, "I'm not sure what to say.")
            print(f"Bot: {bot_response}")
            feedback = input("Is this response correct? (yes/no): ").strip().lower()
            if feedback == 'no':
                correct_response = input("What should be the correct response? ")
                model.train()
                return interactive_training(user_query, correct_response, False)
        return 
    else:
        model.train() 
        if user_response not in response_to_idx:
            response_to_idx[user_response] = output_dim
            idx_to_response[output_dim] = user_response
            output_dim += 1
            new_output_layer = nn.Linear(hidden_dim, output_dim)
            with torch.no_grad():
                new_output_layer.weight[:model.fc.out_features] = model.fc.weight
                new_output_layer.bias[:model.fc.out_features] = model.fc.bias
            model.fc = new_output_layer
            optimizer = optim.Adam(model.parameters(), lr=0.005)

        response_index = response_to_idx[user_response]
        response_tensor = torch.tensor([response_index], dtype=torch.long)
        optimizer.zero_grad()
        output = model(query_tensor)
        loss = criterion(output, response_tensor)
        loss.backward()
        optimizer.step()
        print(f"Corrected response learned. Loss: {loss.item()}")

while True:
    user_query = input("\nEnter your query (or 'exit' to stop): ").strip()
    if user_query.lower() == 'exit':
        break
    interactive_training(user_query)
