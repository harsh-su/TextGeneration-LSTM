from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import torch
import pickle
from model import nxtword  # Assuming your model is named nxtword

app = FastAPI()

# Load the model
def load_model(vocab_size, model_path='model_state_dict.pth'):
    model = nxtword(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model

# Load vocabulary (word_to_index and index_to_word)
def load_vocabulary(vocab_path='vocabulary.pkl'):
    with open(vocab_path, 'rb') as f:
        word_to_index, index_to_word = pickle.load(f)
    return word_to_index, index_to_word

# Text Generation Function
def generate_text(model, input_text, word_to_index, index_to_word, length=10):
    tokens = input_text.lower().split()
    indices = [word_to_index.get(token, word_to_index.get("<unk>", 0)) for token in tokens]
    input_tensor = torch.tensor(indices).unsqueeze(0)  # Add batch dimension

    generated_text = tokens

    with torch.no_grad():
        for _ in range(length):
            output = model(input_tensor)
            predicted_idx = output.argmax(dim=1).item()
            predicted_word = index_to_word.get(predicted_idx, "<unk>")
            generated_text.append(predicted_word)
            input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_idx]])], dim=1)

    return ' '.join(generated_text)

# Pydantic model for the request
class TextGenerationRequest(BaseModel):
    input_text: str
    length: int

@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("frontend/index.html", "r") as file:
        return file.read()

@app.post("/generate_text/")
async def generate_text_endpoint(request: TextGenerationRequest):
    # Load the vocabulary and model
    word_to_index, index_to_word = load_vocabulary('vocabulary.pkl')
    vocab_size = len(word_to_index)
    model = load_model(vocab_size)

    # Generate text based on the input
    generated_text = generate_text(model, request.input_text, word_to_index, index_to_word, request.length)
    
    return {"generated_text": generated_text}
