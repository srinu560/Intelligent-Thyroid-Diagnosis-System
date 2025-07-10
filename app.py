from flask import Flask, request, jsonify, render_template
import joblib
import sqlite3
import pandas as pd
import tensorflow as tf
import warnings
import random
import smtplib 
from email.message import EmailMessage
from datetime import datetime
# Object detection packages
import os
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS  # Enable CORS

warnings.filterwarnings('ignore')



app = Flask(__name__)

#Throid diagnosis --------start------------------

df_sug=pd.read_csv(r"dataset/thyroid_diseases_detailed_info.csv")

# Load models and preprocessors
scaler = joblib.load("models/Thyroid_symptoms_model/scaler.pkl")
label_encoder = joblib.load("models/Thyroid_symptoms_model/label_encoder.pkl")
encoder = joblib.load("models/Thyroid_symptoms_model/encoder.pkl")
nn_model = tf.keras.models.load_model("models/Thyroid_symptoms_model/thyroid_nn_advanced.keras")
final_model = joblib.load("models/Thyroid_symptoms_model/final_unified_model_advanced.pkl")

# Load hybrid classifiers (ensure these are trained and saved)
voting_clf = joblib.load("models/Thyroid_symptoms_model/voting_clf.pkl")
bagging_clf = joblib.load("models/Thyroid_symptoms_model/bagging_clf.pkl")
stacking_clf = joblib.load("models/Thyroid_symptoms_model/stacking_clf.pkl")

# Categorical features for encoding
categorical_features = ["Weight Change", "Heart Rate", "Temperature Sensitivity", "Digestive Issues"]

#Thyroid diagnosis --------End--------


#Thyroid cancer object detection
#--------------------start-----------------------
CORS(app)  # Allow Cross-Origin requests

# Upload folders
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load YOLOv8 Model
MODEL_PATH = "models/best (1).pt"  # Ensure the correct path
object_detection_model = YOLO(MODEL_PATH)

# Class labels from data.yaml
LABELS = ["TC", "abnormal", "fibrocollagenous tissue", "fibromuscular tissue","inflammatory cell", "lymphoid", "normal", "rbc", "stroma"]

#-------------------------End-------------------------


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about")
def about1():
    return render_template("about.html")


@app.route('/logon')
def logon():
	return render_template('register.html')

@app.route('/login')
def login():
	return render_template('login.html')


@app.route('/thyroid_symptoms')
def home2():
	return render_template('thyroid_symptoms.html')

@app.route('/thyroid_val')
def home1():
	return render_template('thyroid_val.html')

@app.route('/thyroid_cancer')
def thyroid_cancer():
	return render_template('thyroid_cancer.html')

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "manojtruprojects@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("manojtruprojects@gmail.com", "qvhanvuuxyogomze")
    s.send_message(msg)
    s.quit()
    return render_template("OTP.html")

@app.route('/OTP_match', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("login.html")
    return render_template("register.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("login.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("thyroid_val.html")
    else:
        return render_template("login.html")

@app.route("/notebook")
def notebook1():
    return render_template("Notebook.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final4=[np.array(int_features)]
    model = joblib.load('models\model_allfs.sav')
    predict = model.predict(final4)

    if predict == 0:
        output = "NORMAL, PATIENT IS NOT SUFFERING FROM THYROID DISEASE!" 
    elif predict == 1:
        output = "SICK, PATIENT IS SUFFERING FROM THYROID DISEASE!" 
    
    
    return render_template('thyroid_val_result.html', output=output)


@app.route('/symptom_predict',methods=['POST'])
def symptom_predict():
    # Extract input values from the form
    sample_input = {
        "Fatigue": int(request.form.get("fatigue")),
        "Weight Change": request.form.get("weight_change"),
        "Heart Rate": request.form.get("heart_rate"),
        "Temperature Sensitivity": request.form.get("temperature_sensitivity"),
        "Mood Changes": int(request.form.get("mood_changes")),
        "Hair/Nail Changes": int(request.form.get("hair_nail_changes")),
        "Neck Swelling": int(request.form.get("neck_swelling")),
        "Digestive Issues": request.form.get("digestive_issues"),
        "Eye Changes": int(request.form.get("eye_changes")),
        "Goiter Presence": int(request.form.get("goiter_presence")),
        "Menstrual Irregularities": int(request.form.get("menstrual_irregularities")),
        "Hoarseness": int(request.form.get("hoarseness")),
        "Family History": int(request.form.get("family_history")),
        "Previous Thyroid Surgery": int(request.form.get("previous_thyroid_surgery")),
        "Radiation Exposure": int(request.form.get("radiation_exposure")),
        "Slow Reflexes": int(request.form.get("slow_reflexes")),
        "Puffy Face": int(request.form.get("puffy_face")),
        "Joint Pain & Stiffness": int(request.form.get("joint_pain")),
        "Muscle Weakness": int(request.form.get("muscle_weakness")),
        "Memory Problems (Brain Fog)": int(request.form.get("memory_problems")),
        "Dry Skin & Brittle Nails": int(request.form.get("dry_skin")),
        "Sweating Excessively": int(request.form.get("sweating_excessively"))
    }

    sample_df = pd.DataFrame([sample_input])

    # Encode categorical features
    encoded_sample = pd.DataFrame(encoder.transform(sample_df[categorical_features]))
    encoded_sample.columns = encoder.get_feature_names_out(categorical_features)

    # Merge encoded features with the rest
    sample_df = sample_df.drop(columns=categorical_features).reset_index(drop=True)
    sample_df = pd.concat([sample_df, encoded_sample], axis=1)

    # Scale the features
    sample_scaled = scaler.transform(sample_df)

    # Neural Network prediction (feature extraction)
    nn_features = nn_model.predict(sample_scaled)

    # Ensure nn_features is 2D
    if nn_features.ndim == 1:
        nn_features = nn_features.reshape(-1, 1)

    # Get predictions from each hybrid classifier and ensure 2D shape
    voting_pred = voting_clf.predict(nn_features).reshape(-1, 1)
    bagging_pred = bagging_clf.predict(nn_features).reshape(-1, 1)
    stacking_pred = stacking_clf.predict(nn_features).reshape(-1, 1)

    # Combine all features for the final meta-classifier
    combined_features = np.hstack([nn_features, voting_pred, bagging_pred, stacking_pred])

    # Ensure final_model receives correct input shape
    final_prediction = final_model.predict(combined_features)

    # Decode the predicted disease
    predicted_disease = label_encoder.inverse_transform(final_prediction)

    output=predicted_disease[0]

    if output=="Normal":
        return render_template("thyroid_symptoms_result.html", prediction=output)

    dis_pred= df_sug[df_sug["Thyroid Disease"] == output]["Disease Description"].values[0]
    prec_mes= df_sug[df_sug["Thyroid Disease"] == output]["Precautionary Measures"].values[0]
    food_diet= df_sug[df_sug["Thyroid Disease"] == output]["Recommended Food Diet"].values[0]

    return render_template("thyroid_symptoms_result.html", prediction=output,dis_pred=dis_pred,prec_mes=prec_mes,food_diet=food_diet)


#Thyroid Cancer Object detection code and functions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_and_draw(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image file or path.")

    results = object_detection_model(img)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0]) * 100
            cls = int(box.cls[0])
            label = LABELS[cls]

            # Draw bounding box and label
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label}: {conf:.2f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections.append({
                "label": label,
                "confidence": f"{conf:.2f}",
                "box": [x1, y1, x2, y2]
            })

    processed_path = os.path.join(PROCESSED_FOLDER, os.path.basename(img_path))
    cv2.imwrite(processed_path, img)
    return detections, processed_path

@app.route('/thyroid_cancer', methods=['POST'])
def image_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file format"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        detections, processed_image_path = predict_and_draw(filepath)
        return jsonify({"detections": detections, "image_url": processed_image_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == "__main__":
    app.run(debug=False)
