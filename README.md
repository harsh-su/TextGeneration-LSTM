# Next-Word Prediction API

A FastAPI-based application for predicting the next `k` words based on user input. The API is powered by a pre-trained LSTM model and comes with an intuitive frontend for easy interaction.

---

## How It Works

1. The backend is built using **FastAPI** to serve the prediction model.
2. The model uses an LSTM architecture to predict the next `k` words based on input context.
3. A frontend interface connects to the API, allowing users to interact with the prediction service.

---

## Project Structure

```plaintext
.
├── app/
│   ├── main.py               # FastAPI app
│   ├── model.py              # LSTM model logic
│   ├── next_word.py          # Prediction API logic
│   ├── frontend/
│   │   └── index.html        # Frontend interface
│   └── model_state_dict.pth  # Trained model weights
├── requirements.txt          # Project dependencies
├── vercel.json               # Vercel deployment configuration
├── README.md                 # Project documentation
