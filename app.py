from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import torch

app = Flask(__name__)

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)

model.load_state_dict(torch.load('sentiment_model.pt', map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.form['text']
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        predicted_class = np.argmax(probabilities)
        confidence = round(max(probabilities) * 100, 2)
        
        sentiment = "positive" if predicted_class == 1 else "negative"
        
        return render_template('index.html', 
                             result={'sentiment': sentiment, 'confidence': confidence},
                             text=text)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)