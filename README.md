# Emotional-Sentiment-Analysis-and-Adaptive-Response-System
main.py

```
import pandas as pd
import numpy as np
import nltk
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)

class EmotionalSentimentAnalysisChatbot:
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize the chatbot with pre-trained models and configurations
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        # Emotion categories
        self.emotions = ['sadness', 'anxiety', 'stress', 'joy', 'neutral']
        
        # Data preprocessing configurations
        self.stop_words = set(stopwords.words('english'))
        
        # Model and tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.emotions)
        
        # Initialize models
        self.sentiment_model = None
    
    def preprocess_text(self, text):
     if not isinstance(text, str):
        return ""
    
    # Keep some meaningful punctuation and emotional indicators
     text = text.lower()
     text = re.sub(r'[^a-zA-Z\s\.,!?]', '', text)
    
    # Remove only select stopwords, keeping emotionally significant words
     text = ' '.join([word for word in text.split() if word not in self.stop_words or word in ['sad', 'happy', 'stress', 'anxiety']])
    
     return text.strip()
    
    def create_sample_dataset(self):
        """
        Create a comprehensive sample dataset for training
        
        Returns:
            tuple: Training and testing datasets
        """
        # Expanded sample data with more diverse emotional contexts
        sample_data = {
            'text': [
                "Hii"
                "hello"
                "I feel so sad and lonely today",
                "Work is causing me so much stress",
                "I'm constantly worried about everything",
                "I'm feeling really happy and excited",
                "Today feels pretty neutral to me",
                "I'm experiencing a lot of anxiety",
                "Everything seems overwhelming right now",
                "I got a promotion and I'm thrilled!",
                "I'm feeling down and can't seem to cheer up",
                "The pressure at work is unbearable",
                "My life feels empty and meaningless",
                "I'm so excited about my upcoming vacation",
                "I can't stop worrying about small things",
                "This day is going perfectly",
                "I feel completely stuck and hopeless"
            ],
            'emotion': [
                'sadness', 'stress', 'anxiety', 'joy', 'neutral', 
                'anxiety', 'stress', 'joy', 'sadness', 'stress',
                'sadness', 'joy', 'anxiety', 'joy', 'sadness'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Clean text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Encode labels
        df['encoded_emotion'] = self.label_encoder.transform(df['emotion'])
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df[['processed_text', 'encoded_emotion']])
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['processed_text'], 
                padding='max_length', 
                truncation=True,
                max_length=128
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split dataset
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.shuffle(seed=42).select(range(train_size))
        test_dataset = tokenized_dataset.shuffle(seed=42).select(range(train_size, len(tokenized_dataset)))
        
        return train_dataset, test_dataset
    
    def train_sentiment_model(self, train_dataset, test_dataset):
        """
        Train sentiment classification model with robust configuration
        
        Args:
            train_dataset (Dataset): Training dataset
            test_dataset (Dataset): Testing dataset
        """
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")
        
        # Load pre-trained model for sequence classification
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.emotions)
        ).to(device)
        
        # Define comprehensive training arguments
        training_args = TrainingArguments(
            output_dir='./sentiment_model_results',
            num_train_epochs=16,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy='epoch',
            learning_rate=5e-5,
            no_cuda=not torch.cuda.is_available()
        )
        
        # Prepare datasets
        def prepare_dataset(dataset):
            dataset = dataset.remove_columns(['__index_level_0__'] if '__index_level_0__' in dataset.column_names else [])
            dataset = dataset.add_column('labels', dataset['encoded_emotion'])
            return dataset
        
        train_dataset = prepare_dataset(train_dataset)
        test_dataset = prepare_dataset(test_dataset)
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.sentiment_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model('./sentiment_model')
        print("Model training completed and saved.")
    
    def predict_emotion(self, text):
        """
        Predict emotion for given text with error handling
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            str: Predicted emotion
        """
        if not self.sentiment_model:
            raise ValueError("Sentiment model not trained. Train the model first.")
        
        # Preprocess and tokenize input
        processed_text = self.preprocess_text(text)
        
        # Handle empty text
        if not processed_text:
            return 'neutral'
        
        inputs = self.tokenizer(processed_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        # Move inputs to the same device as the model
        device = next(self.sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict emotion
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1)
        
        return self.label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
    
    def generate_empathetic_response(self, detected_emotion):
        """
        Generate culturally sensitive empathetic responses
        
        Args:
            detected_emotion (str): Detected emotion
        
        Returns:
            str: Empathetic response
        """
        emotion_responses = {
            'sadness': [
                "I hear that you're going through a difficult time. Your feelings are valid, and it's okay to not be okay.",
                "It sounds like you're experiencing some heavy emotions. Would you like to talk more about what's troubling you?",
                "I'm here to listen and support you during this challenging moment."
            ],
            'anxiety': [
                "It seems like you're feeling overwhelmed. Let's take a deep breath together and break things down step by step.",
                "Your feelings of anxiety are completely understandable. Would you like to explore some coping strategies?",
                "Anxiety can be tough. Remember that you're stronger than your worries, and support is available."
            ],
            'stress': [
                "I understand that you're under a lot of pressure right now. Let's discuss some ways to manage your stress.",
                "Stress can be exhausting. What specific aspects are weighing most heavily on you?",
                "It's important to prioritize your well-being. Would you like to talk about some stress-reduction techniques?"
            ],
            'joy': [
                "That's wonderful! I'm genuinely happy for you and the positive experience you're having.",
                "Congratulations! It's great to hear something positive that's bringing you joy.",
                "Your excitement is contagious! Tell me more about what's making you feel so good."
            ],
            'neutral': [
                "I'm here and ready to listen. How are you feeling right now?",
                "Sometimes it's okay to just be. Is there anything specific on your mind?",
                "I'm present and attentive. Feel free to share whatever you'd like."
            ]
        }
        
        return np.random.choice(emotion_responses.get(detected_emotion, emotion_responses['neutral']))

def main():
    print("Initializing Emotional Sentiment Analysis Chatbot...")
    
    # Initialize chatbot
    chatbot = EmotionalSentimentAnalysisChatbot()
    
    # Create and prepare sample dataset
    train_dataset, test_dataset = chatbot.create_sample_dataset()
    
    # Train sentiment model
    chatbot.train_sentiment_model(train_dataset, test_dataset)
    
    # Interactive conversation loop
    print("\nChatbot is ready. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Detect emotion
        emotion = chatbot.predict_emotion(user_input)
        
        # Generate response
        response = chatbot.generate_empathetic_response(emotion)
        
        print(f"Bot (Detected Emotion - {emotion.upper()}): {response}")

if __name__ == "__main__":
    main()
```
app.py

```

import os
import torch
from flask import Flask, render_template, request, jsonify
from sentiment_chatbot import EmotionalSentimentAnalysisChatbot

app = Flask(__name__)

# Initialize chatbot globally
chatbot = EmotionalSentimentAnalysisChatbot()

# Create and prepare sample dataset
train_dataset, test_dataset = chatbot.create_sample_dataset()

# Train sentiment model
chatbot.train_sentiment_model(train_dataset, test_dataset)

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Process user input and generate chatbot response"""
    user_input = request.json.get('message', '')
    
    try:
        # Detect emotion
        emotion = chatbot.predict_emotion(user_input)
        
        # Generate response (using the existing method with one argument)
        response = chatbot.generate_empathetic_response(emotion)
        
        return jsonify({
            'input': user_input,
            'detected_emotion': emotion,
            'bot_response': response
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
```
