# 🌱 Plant Disease Classification App

A **Streamlit web application** that uses **deep learning models** to detect and classify plant diseases from leaf images.  

Currently supported plants:

- 🥔 **Potato**: Early Blight, Late Blight, Healthy  
- 🍅 **Tomato**: 9 common tomato diseases + Healthy  

This tool helps farmers, researchers, and agricultural specialists **quickly identify plant diseases** for better crop management.

---

## 🚀 Features
- **Upload an image** of a potato or tomato leaf.
- **Real-time prediction** using trained TensorFlow models.
- **Confidence score** for each prediction.
- **Simple sidebar navigation** for plant selection.

---

## 📂 Project Structure
├── app.py # Main Streamlit application
├── Saved_models/
│ ├── model_v1_potato.h5 # Potato disease model
│ ├── Saved_models1_tomato.h5 # Tomato disease model
├── requirements.txt # Python dependencies
├── Potato_Training_data.ipynb # Potato model training notebook
├── Tomato_Training_data.ipynb # Tomato model training notebook
└── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/plant-disease-classification.git
cd plant-disease-classification
2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
📊 Models Used
Potato Model
Input Size: 256x256 RGB

Classes:

Potato__Early_blight

Potato__Late_blight

Potato__Healthy

Tomato Model
Input Size: 256x256 RGB

Classes:

Tomato_Bacterial_spot

Tomato_Early_blight

Tomato_Late_blight

Tomato_Leaf_Mold

Tomato_Septoria_leaf_spot

Tomato_Spider_mites_Two_spotted_spider_mite

Tomato__Target_Spot

Tomato__Tomato_YellowLeaf__Curl_Virus

Tomato__Tomato_mosaic_virus

Tomato_healthy

🛠 How It Works
Image Upload: User uploads a leaf image.

Preprocessing:

Convert to RGB

Resize to 256x256

Normalize pixel values (0–1)

Prediction: Passes through the corresponding trained TensorFlow model.

Output: Displays predicted class & confidence score.

🖼 Example Output
Uploaded Image:

Prediction:

Class: Potato__Early_blight

Confidence: 98.45%

📚 Tech Stack
Python 3

TensorFlow / Keras

Streamlit

PIL

NumPy

📌 Future Improvements
Add support for more plant species.

Deploy to Streamlit Cloud, Heroku, or AWS.

Use data augmentation and transfer learning for improved accuracy.
