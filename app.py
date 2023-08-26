from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.animation import Animation
from language_detector import detect_language
from functions.load_responses import load_responses_from_database
from functions.transliterate import transliterate
from functions.myanglish_to_burmese import myanglish_to_burmese
from functions.improved_transliterate import improved_transliterate
import torch
import torch.nn as nn

queries, responses, response_category_mapping = load_responses_from_database()

vocab = set(' '.join(queries))
vocab_size = 21
print(f"Vocabulary Size: {vocab_size}")
index_to_char = dict(enumerate(vocab))
char_to_index = {char: idx for idx, char in index_to_char.items()}

def string_to_onehot(string, max_length=50):
    tensor = torch.zeros(max_length, vocab_size)
    for li, letter in enumerate(string):
        if letter in char_to_index:
            tensor[li][char_to_index[letter]] = 1
    return tensor

input_size = 50 * vocab_size  
print(f"Input Size: {input_size}")
hidden_size = 128
output_size = len(responses)

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
        layout = FloatLayout(size=(300, 300))
        
        self.user_input = TextInput(size_hint=(.8, .1), pos_hint={'x':.1, 'y':.8}, hint_text='Type your query here')
        
        query_button = Button(size_hint=(.6, .1), pos_hint={'x':.2, 'y':.6}, text='Ask Assistant', background_color=(0, 0.5, 0.5, 1))
        query_button.bind(on_press=self.handle_query)
        
        self.response_label = Label(size_hint=(.8, .3), pos_hint={'x':.1, 'y':.3}, text='', font_name="fonts/Pyidaungsu.ttf", font_size=24)
        
        layout.add_widget(self.user_input)
        layout.add_widget(query_button)
        layout.add_widget(self.response_label)
        
        return layout

    def handle_query(self, instance):

        anim = Animation(pos_hint={'x': .1, 'y': .59}, t='out_bounce')
        anim += Animation(pos_hint={'x': .2, 'y': .6}, t='out_bounce')
        anim.start(instance)

        original_user_query = self.user_input.text  
        language_used = detect_language(original_user_query)
    
        print(f"User's Original Input: {original_user_query}") 
        print(f"Detected Language: {language_used}") 
  
        user_query = original_user_query

        if language_used == "Myanglish":
             user_query = improved_transliterate(original_user_query)
             burmese_translation = myanglish_to_burmese(user_query)
             print(f"Converted to Burmese: {burmese_translation}") 
             user_query = burmese_translation

        assistant = VirtualAssistant(model)
        response = assistant.respond_to_query(user_query)
        self.response_label.text = response

        self.response_label.opacity = 0
        anim = Animation(opacity=1, duration=2)
        anim.start(self.response_label)

if __name__ == "__main__":
    AssistantApp().run()
