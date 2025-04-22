from flask import Flask, request, jsonify
from flask_cors import CORS
import string
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from datetime import datetime
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Setup
app = Flask(__name__)
CORS(app)

# ===================== NLP Model =====================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english")) - {"not", "no"}
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    return ' '.join(tokens)

try:
    models = joblib.load("text/chatbot_models.pkl")
    medical_classifier = models["medical_classifier"]
    disease_classifier = models["disease_and_specialization_classifier"]
    label_encoder = models["label_encoder"]
except Exception as e:
    raise

# ===================== Image Model =====================
try:
    image_model = tf.keras.models.load_model("image/Brain_Tumors.h5")
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
except Exception as e:
    raise

def preprocess_image(image):
    image = image.convert("RGB")
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


 
# ===================== Routes =====================

@app.route('/')
def home():
    return "✅ Unified Medical Diagnosis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Case 1: Predict from image
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file)
            processed_img = preprocess_image(image)
            predictions = image_model.predict(processed_img)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_labels[tf.argmax(score)]
            if predicted_class == '':
                    # Message when there is no prediction (empty result)
                    message_en = "The result couldn’t be determined. Please consult a specialist for further evaluation."
                    message_ar = "لم يتمكن النظام من تحديد التشخيص. يُفضل استشارة الطبيب المختص لمزيد من التقييم."
            else:
                # Messages for different predictions
                if predicted_class.lower() == 'glioma':
                    message_en = "You have been diagnosed with Glioma. It is recommended to consult a neurologist for further assessment and treatment."
                    message_ar = "تم تشخيصك بورم دبقي. يُفضّل استشارة أطباء الأعصاب لمزيد من التقييم والعلاج المناسب."
                elif predicted_class.lower() == 'meningioma':
                    message_en = "You have been diagnosed with Meningioma. Please visit a specialist for further examination and treatment."
                    message_ar = "تم تشخيصك بورم سحائي. ننصحك بزيارة طبيب مختص لتحديد الخطوات التالية في العلاج."
                elif predicted_class.lower() == 'notumor':
                    message_en = "Good news! There is no tumor detected in the image. Continue taking care of your health."
                    message_ar = "الحمد لله! لا يوجد ورم في الصورة الطبية. استمر في الاهتمام بصحتك."
                elif predicted_class.lower() == 'pituitary':
                    message_en = "You have been diagnosed with Pituitary Tumor. It is important to consult an endocrinologist for treatment."
                    message_ar = "تم تشخيصك بورم في الغدة النخامية. من المهم متابعة حالتك مع أطباء الغدد الصماء للحصول على العلاج اللازم."
                else:
                    message_en = "The result couldn’t be determined. Please consult a specialist."
                    message_ar = "لم يتمكن النظام من تحديد التشخيص. يرجى استشارة الطبيب المتخصص."            

            messageEn = message_en  
            messageAr = message_ar
            return jsonify({
                "code": 200,
                "input_type": "image",
                "status": "success",
                "messageEn": messageEn,
                "messageAr":   messageAr,
                "prediction": predicted_class,
                "time": current_datetime
            }), 200

        # Case 2: Predict from text
        if 'input_text' in request.form:
            user_input = request.form['input_text']
            processed = preprocess_text(user_input)
            is_medical = medical_classifier.predict([processed])[0]

            if is_medical == 1:
                prediction = disease_classifier.predict([processed])[0]
                disease = label_encoder.inverse_transform([prediction])[0]
                return jsonify({
                    "code": 200,
                    "input_type": "text",
                    "status": "success",
                     "messageEn": "Consult a doctor for proper diagnosis",
                "messageAr":  "استشر طبيبًا للتشخيص الصحيح",
                    "prediction": disease,
                    "time": current_datetime
                }), 200
            else:
                return jsonify({
                    "code": 200,
                    "input_type": "text",
                    "status": "success",
                    "messageEn": "Please enter medical symptoms ",
                    "messageAr": "يرجى إدخال أعراض طبية صحيحة",
                    "prediction": "",
                    "time": current_datetime
                }), 200

        return jsonify({
            "code": 400,
            "status": "failure",
            "messageEn": "No valid input provided. Send either 'image' or 'input_text'.",
            "messageAr": "لم يتم توفير إدخال صالح. أرسل إما 'صورة' أو 'نص_إدخال'.",
            "input_type": "none",
            "prediction": "",
            "time": current_datetime
        }), 400

    except Exception as e:
        return jsonify({
            "code": 500,
            "status": "error",
            "message": str(e),
            "input_type": "none",
            "prediction": "",
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500
# ===================== Start =====================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
