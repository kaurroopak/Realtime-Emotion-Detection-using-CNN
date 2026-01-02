# 😊 Real-Time Mood Detection System  
### ELC Mini Project – 5th Semester  
**Course:** Experiential Learning Component (ELC)  
**Domain:** Real-Time Data Analysis & AI  

---

## 📌 Project Overview

This project implements a **Real-Time Mood Detection System** that classifies human facial expressions into three emotional states: **Happy, Neutral, and Sad**. The system uses a **Convolutional Neural Network (CNN)** based on **MobileNetV2** and is deployed as an interactive **Streamlit web application** with real-time inference support.

To ensure efficient real-time performance, the trained model is converted into **TensorFlow Lite (TFLite)** format. The application supports both **image upload** and **live webcam capture**, making it suitable for real-world emotion recognition scenarios.

---

## 🎯 Objectives

- To design and deploy a real-time image-based emotion recognition system  
- To understand CNN-based feature extraction using transfer learning  
- To evaluate model performance on unseen and real-world data  
- To analyze model behavior under unexpected and perturbed inputs  
- To demonstrate real-time deployment using lightweight inference (TFLite)

---

## 📂 Dataset Used

- **Dataset Name:** AffectNet  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/mstjebashazida/affectnet  
- **Data Modality:** Image (Facial Expressions)  
- **Classes Used:** Happy, Neutral, Sad  

The dataset contains labeled facial images collected under varied lighting and pose conditions, making it suitable for emotion recognition tasks.

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (Pretrained on ImageNet)  
- **Approach:** Transfer Learning (CNN)  
- **Input Size:** 224 × 224 RGB images  
- **Output:** 3-class Softmax (Happy / Neutral / Sad)  

The convolutional layers act as automatic feature extractors, learning spatial facial features such as eye contours, mouth curvature, and facial symmetry.

---

## ⚙️ Preprocessing & Training

- Image resizing to 224×224  
- Pixel normalization  
- Dataset filtering for target classes  
- Validation split for model evaluation  
- Early stopping to prevent overfitting  

Training and validation accuracy curves were used to analyze model generalization.

---

## 📊 Model Evaluation

The model was evaluated using:
- Accuracy  
- Precision, Recall, F1-Score  
- Confusion Matrix  
- Training vs Validation Accuracy  

The system demonstrated strong performance for **Happy** and **Neutral** expressions, with some confusion observed between **Sad** and **Neutral**, especially in real-world inputs.

---

## 🚀 Deployment & Real-Time Implementation

- **Deployment Method:** Streamlit Web Application  
- **Model Format:** TensorFlow Lite (TFLite)  
- **Input Modes:**
  - Image Upload  
  - Live Webcam Capture  

### Real-Time Performance
- Inference latency ranges between **120–170 ms**
- Predictions are generated instantly for user inputs
- Confidence scores are visualized using bar charts

---

## 🧪 AI Exploration Experiments (Module 6)

### 1. Unexpected Inputs
Non-face objects (book, wall, laptop, blank image) were tested.  
The model still produced predictions due to the absence of a rejection mechanism, highlighting a limitation of softmax-based classifiers.

### 2. Real-World Inputs
Webcam images captured under natural conditions showed good generalization for happy and neutral expressions, while subtle sad expressions were sometimes misclassified.

### 3. Controlled Perturbation
Gaussian blur was applied to input images to test robustness.  
Blurring reduced confidence and increased misclassification, demonstrating the model’s reliance on fine-grained spatial features.

---

## 🖥️ Application Features

- Clean and interactive UI  
- Real-time mood prediction  
- Confidence visualization  
- Inference time display  
- Robustness testing via image blur (AI exploration)

---

## ▶️ How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Run the Application

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run app.py   `

🧾 Technologies Used
--------------------

*   **Python**
    
*   **TensorFlow & TensorFlow Lite**
    
*   **OpenCV**
    
*   **Streamlit**
    
*   **NumPy**
    
*   **Pandas**
    
*   **PIL (Pillow)**
    

📌 Key Learnings
----------------

*   Practical understanding of real-time AI model deployment
    
*   Importance of dataset quality and the impact of domain shift
    
*   Behavioral analysis of CNN models under unexpected inputs
    
*   Trade-offs between model accuracy and real-world robustness
    

🏁 Conclusion
-------------

This project demonstrates an end-to-end real-time AI system, covering dataset exploration, model training, deployment, and behavioral analysis. It highlights both the strengths and limitations of facial emotion recognition systems in real-world conditions and successfully fulfills all requirements of the **ELC assessment modules**.
