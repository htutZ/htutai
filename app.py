from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from language_detector import detect_language
from functions.load_responses import load_responses_from_database
from functions.transliterate import transliterate
import torch
import torch.nn as nn

# Dummy data
queries, responses, response_category_mapping = load_responses_from_database()

# Create a vocabulary from the queries
vocab = set(' '.join(queries))
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")
index_to_char = dict(enumerate(vocab))
char_to_index = {char: idx for idx, char in index_to_char.items()}

# Convert queries to one-hot encoded tensors
def string_to_onehot(string, max_length=50):
    tensor = torch.zeros(max_length, vocab_size)
    for li, letter in enumerate(string):
        if letter in char_to_index:
            tensor[li][char_to_index[letter]] = 1
    return tensor

input_size = 50 * vocab_size  # This ensures a consistent input size based on vocab
print(f"Input Size: {input_size}")
hidden_size = 128
output_size = len(responses)

# Define the SimpleNN class again
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

# Load the trained model
model = SimpleNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

class VirtualAssistant:
    def __init__(self, model):
        self.model = model

    def respond_to_query(self, user_query):
        with torch.no_grad():
            user_query = user_query.lower()
            input_tensor = string_to_onehot(user_query)
            prediction = self.model(input_tensor.view(1, -1))
            response_idx = torch.argmax(prediction).item()
            if response_idx < len(responses):
                response = responses[response_idx]
                category = response_category_mapping[response_idx]
                print(f"Response taken from category: {category}")
                return response
            else:
                return "I'm not sure what you mean."

class AssistantApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.user_input = TextInput(hint_text='Type your query here')
        self.response_label = Label(text='', font_name="MyanmarFont.ttf")
        query_button = Button(text='Ask Assistant')
        query_button.bind(on_press=self.handle_query)
        layout.add_widget(self.user_input)
        layout.add_widget(query_button)
        layout.add_widget(self.response_label)
        return layout

    def handle_query(self, instance):
        user_query = transliterate(self.user_input.text)
        user_query = self.user_input.text
        language_used = detect_language(user_query)
        print(f"Detected Language: {language_used}")  # This will print the detected language to console

        assistant = VirtualAssistant(model)  # Pass the trained model
        response = assistant.respond_to_query(user_query)
        self.response_label.text = response


if __name__ == "__main__":
    AssistantApp().run()
