import torch
import torch.nn as nn
import torch.optim as optim
from functions.load_responses import load_responses_from_database
from functions.transliterate import transliterate

queries, responses, _ = load_responses_from_database()
queries = [transliterate(query) for query in queries]

vocab = set(' '.join(queries))
vocab_size = len(vocab)
index_to_char = dict(enumerate(vocab))
char_to_index = {char: idx for idx, char in index_to_char.items()}

def string_to_onehot(string, max_length=50):
    tensor = torch.zeros(max_length, vocab_size)
    for li, letter in enumerate(string):
        if letter in char_to_index:  
            tensor[li][char_to_index[letter]] = 1
    return tensor

X = torch.stack([string_to_onehot(query) for query in queries])
Y = torch.tensor([i for i, _ in enumerate(responses)])

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

split_idx = int(0.8 * len(queries))
train_X = X[:split_idx]
train_Y = Y[:split_idx]
val_X = X[split_idx:]
val_Y = Y[split_idx:]

input_size = 50 * vocab_size
print("Vocabulary Size:", vocab_size)
print("Input Size:", input_size)
hidden_size = 128
output_size = len(responses)
learning_rate = 0.005
epochs = 1000
batch_size = 2

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_data = torch.utils.data.TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
patience = 20
best_val_loss = float('inf')
counter = 0

def get_response(model, query):
    model.eval()
    with torch.no_grad():
        input_tensor = string_to_onehot(transliterate(query))
        prediction = model(input_tensor.view(1, -1))
        response_idx = torch.argmax(prediction).item()
        if response_idx < len(responses):
            return responses[response_idx]
        else:
            return "I'm not sure what you mean."

for epoch in range(epochs):
    model.train()
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X.view(batch_X.size(0), -1))
        loss = criterion(output, batch_Y)
        loss.backward()
        optimizer.step()
    
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_output = model(val_X.view(val_X.size(0), -1))
        val_loss = criterion(val_output, val_Y)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss.item()}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break
    
    # Interaction every epoch
    user_query = input("\nEnter a query to test the model (or 'exit' to continue training): ")
    if user_query.lower() == "exit":
        continue
    else:
        print(f"Model's Response: {get_response(model, user_query)}")
