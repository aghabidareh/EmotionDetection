import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]

        emotion = result.get('dominant_emotion', 'Unknown').capitalize()
        emotion_scores = result.get('emotion', {})

        height, width, _ = frame.shape
        overlay = frame.copy()
        alpha = 0.6

        color_map = {
            'Happy': (0, 255, 0), 'Sad': (255, 0, 0), 'Angry': (0, 0, 255),
            'Surprise': (0, 255, 255), 'Neutral': (200, 200, 200), 'Fear': (128, 0, 128),
            'Disgust': (0, 128, 128)
        }

        # Background color for the dominant emotion
        color = color_map.get(emotion, (255, 255, 255))
        cv2.rectangle(overlay, (0, 0), (width, 150), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Emotion: {emotion}", (20, 60), font, 2, (0, 0, 0), 3)

    except:
        pass

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
