import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = r"D:\BE 24-27\5th sem\SE\SE LAB\Project\BAE\BAE--Bringing-Aesthetics-to-Emotions\models\mood_model\mobilenetv2_mood_3class.h5"
MOOD_LABELS = ['happy', 'neutral', 'sad']

model = tf.keras.models.load_model(MODEL_PATH)

def predict_mood_from_frame(frame):
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = cv2.resize(rgb, (224, 224))
    # Preprocess input for MobileNetV2
    x = np.expand_dims(resized, axis=0).astype(np.float32)
    x = preprocess_input(x)
    preds = model.predict(x)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return MOOD_LABELS[idx], confidence

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mood, conf = predict_mood_from_frame(frame)

        # Display mood and confidence on frame
        label = f"{mood} ({conf*100:.1f}%)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('BAE Mood Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
