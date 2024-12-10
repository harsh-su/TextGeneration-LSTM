import torch
import pickle
from model import nxtword

# Load model weights
def load_model(vocab_size, model_path='model_state_dict.pth'):
    model = nxtword(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'), weights_only=True))  # Use weights_only=True
    model.eval()  # Set the model to evaluation mode
    return model

# Load vocabulary (word_to_index and index_to_word)
def load_vocabulary(vocab_path='vocabulary.pkl'):
    with open(vocab_path, 'rb') as f:
        word_to_index, index_to_word = pickle.load(f)
    return word_to_index, index_to_word



# Example prediction function
def tokenize_text(text, word_to_index):
    tokens = text.lower().split()  # Tokenize by splitting by spaces
    indices = [word_to_index.get(token, word_to_index.get("<unk>", 0)) for token in tokens]  # Default to 0 if <unk> not found
    return torch.tensor(indices).unsqueeze(0)  # Add batch dimension

def predict_next_word(model, input_text, word_to_index, index_to_word):
    tokenized_input = tokenize_text(input_text, word_to_index)
    
    with torch.no_grad():
        output = model(tokenized_input)
    
    predicted_idx = output.argmax(dim=1).item()  # Get the index of the predicted word
    predicted_word = index_to_word.get(predicted_idx, "<unk>")  # Get the word from index
    
    return predicted_word


# Example terminal-based prediction
if __name__ == "__main__":
    # Load the vocabulary
    
    word_to_index, index_to_word = load_vocabulary('vocabulary.pkl')
    vocab_size = len(word_to_index)  # Based on the vocabulary loaded
    model = load_model(vocab_size)
    
    # Start terminal prediction
    while True:
        input_text = input("Enter a sentence (or 'exit' to quit): ")
        
        if input_text.lower() == 'exit':
            break
        
        next_word = predict_next_word(model, input_text, word_to_index, index_to_word)
        print(f"Predicted next word: {next_word}")