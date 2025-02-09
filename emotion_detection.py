import cv2
from deepface import DeepFace

# راه‌اندازی دوربین
cap = cv2.VideoCapture(0)

# متغیرهای شمارنده برای احساسات مختلف
sad, happy, angry = 0, 0, 0

while cap.isOpened():
    ret, frame = cap.read()  # خواندن فریم از دوربین
    if not ret:
        break  # اگر فریمی دریافت نشود، حلقه متوقف می‌شود

    try:
        # تحلیل احساسات چهره با استفاده از DeepFace
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
        emotion = analysis[0]['dominant_emotion']  # دریافت احساس غالب

        # نمایش احساس تشخیص داده‌شده روی تصویر
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # به‌روزرسانی شمارنده احساسات
        if emotion == 'sad':
            sad += 1
        elif emotion == 'happy':
            happy += 1  
        elif emotion == 'angry':
            angry += 1  

    except Exception as e:
        print(f"خطا در پردازش فریم: {e}")  # نمایش خطا در صورت بروز مشکل

    # نمایش تصویر در پنجره OpenCV
    cv2.imshow("Emotion Detection", frame)
    
    # خروج از حلقه در صورت فشار دادن کلید 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# آزادسازی دوربین و بستن تمام پنجره‌ها
cap.release()
cv2.destroyAllWindows()

# چاپ تعداد دفعات تشخیص هر احساس
print(f'angry => {angry}')
print(f'happy => {happy}')
print(f'sad => {sad}')
