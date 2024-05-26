from flask import Flask, request, jsonify, render_template
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the saved CNN-LSTM model
model = tf.keras.models.load_model('CNN_LSTM_mental_health_model.h5')

# Load the saved Multinomial Naive Bayes classifier
naive_bayes_classifier = joblib.load('naive_bayes_classifier.pkl')

# Load the saved Tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = joblib.load(f)

# Load the saved CountVectorizer
with open('count_vectorizer.pkl', 'rb') as f:
    count_vectorizer = joblib.load(f)

data1 = pd.read_csv('Dataset_Disorders.csv')

# Load mental health-related keywords
mental_health_keywords = [
    'anxious', 'panic', 'worry', 'fear', 'nervousness', 'anxiety', 'tension', 'apprehension', 'unease', 'agitated', 'stress', 'overwhelmed', 'dread', 'races', 'edge', 'hyperventilate', 'overwhelms',
    'manic', 'depressive', 'hypomanic', 'euphoria', 'bipolar', 'mania', 'mood', 'swings', 'cyclothymia', 'instability', 'mixed', 'episodes', 'extreme', 'high-energy', 'crash', 'emotional', 'rollercoaster',
    'sadness', 'hopelessness', 'loss', 'interest', 'disturbances', 'sad', 'hopeless', 'fatigue', 'isolation', 'melancholy', 'despair', 'worthlessness', 'lethargy', 'low mood', 'sadness', 'worthless', 'appetite',
    'anorexia', 'bulimia', 'purging', 'restrictive eating', 'orthorexia', 'EDNOS', 'binge eating', 'body dysmorphia', 'overeating', 'food obsession', 'preoccupation', 'weight', 'eat', 'starving', 'eating', 'secret', 'trapped',
    'insomnia', 'sleeplessness', 'restlessness', 'disturbances', 'difficulty', 'wakefulness', 'deprivation', 'sleepless', 'restless', 'tired', 'sleep', 'rest',
    'apnea', 'snoring', 'interrupted', 'obstructive', 'OSA', 'CPAP', 'sleepiness', 'apnea', 'events', 'tired', 'sleep', 'rest',
    'sleepwalking', 'somnambulism', 'night', 'wandering', 'parasomnia', 'nocturnal', 'wandering', 'arousal',
    'obsession', 'compulsion', 'intrusive', 'thoughts', 'ritualistic', 'behavior', 'repetitive', 'actions', 'checking', 'counting', 'contamination', 'fears', 'symmetry', 'orderliness', 'obsessing', 'thoughts', 'rituals', 'compelled', 'compulsions', 'check',
    'flashbacks', 'nightmares', 'hypervigilance', 'avoidance', 'hyperarousal', 'trauma', 'dissociation', 're-experiencing', 'emotional', 'numbing', 'startle', 'response', 'traumatic', 'event', 'past', 'nightmares', 'reliving', 'haunted',
    'social', 'anxiety', 'fear', 'judgment', 'phobia', 'avoidance', 'performance', 'socialfear', 'excessive', 'self-consciousness', 'embarrassment',
    'hallucinations', 'delusion', 'psychosis', 'disorganized', 'thinking', 'catatonic', 'behavior', 'schizophrenic', 'thought', 'negative', 'symptoms', 'flat', 'affect', 'auditory', 'voices', 'scared', 'real', 'grip', 'reality',
    'defiance', 'defiant', 'behavior', 'irritability', 'argumentative', 'noncompliance', 'hostility', 'temper', 'tantrums', 'oppositional', 'vindictiveness',
    'pressure','unbearable','deadlines','juggling'
    ]


# Function to preprocess input text
def preprocess_input_text(input_text):
    words = word_tokenize(input_text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(words)

# Function to extract keywords from input text
def extract_keywords(input_text):
    words = word_tokenize(input_text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    tagged_words = pos_tag(words)
    return [word for word, tag in tagged_words if tag.startswith('NN') or tag.startswith('JJ')]

# Function to predict mental health class
def predict_mental_health(input_text):
    preprocessed_text = preprocess_input_text(input_text)
    input_sequence = tokenizer.texts_to_sequences([preprocessed_text])
    input_padded = pad_sequences(input_sequence, maxlen=120, padding='post', truncating='post')
    return model.predict(input_padded)[0]


# Function to classify user input and recommend therapeutic techniques
def classify_and_recommend(input_text):
    prediction = predict_mental_health(input_text)
    response = {}
    if prediction > 0.5:
        response['message'] = "The input is related to Mental Health."
        keywords = extract_keywords(input_text)
        new_symptoms = [keyword for keyword in keywords if keyword.lower() in mental_health_keywords]
        response['mental_health_related_symptoms'] = new_symptoms
        if not new_symptoms:
            response['diseases'] = ["No mental health-related symptoms were identified from the input."]
            return response
        new_symptom_matrix = count_vectorizer.transform(new_symptoms)
        predicted_disease = naive_bayes_classifier.predict(new_symptom_matrix)
        unique_diseases = set(predicted_disease)  # Use a set to store unique diseases
        response['diseases'] = list(unique_diseases)  # Convert to list for JSON serialization
        recommendations = []
        for disease in unique_diseases:
            techniques = data1[data1['disease'] == disease]['therapeutic techniques'].values
            if techniques:
                # Format the recommendation as a string
                recommendation_str = f"{disease}: {techniques[0]}"
                recommendations.append(recommendation_str)
        response['recommendations'] = recommendations
    else:
        response['message'] = "The input is not related to Mental Health."
    return response


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    user_input_text = request.form.get('user_input')
    response = classify_and_recommend(user_input_text)
    # Render the result.html template with the classification results
    return render_template('result.html', message=response['message'],
                           diseases=response.get('diseases', []),
                           recommendations=response.get('recommendations', []))

if __name__ == '__main__':
    app.run(debug=True)
