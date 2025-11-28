from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import logging
import random

# ==========================
# 🔧 CẤU HÌNH CƠ BẢN
# ==========================
# Tắt warning TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Đảm bảo working dir = thư mục chứa app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# ==========================
# 🧱 KIỂM TRA TEMPLATE
# ==========================
template_path = os.path.join(BASE_DIR, 'templates', 'index.html')
if os.path.exists(template_path):
    logger.info(f"✅ Tìm thấy template: {template_path}")
else:
    logger.error(f"❌ KHÔNG tìm thấy template: {template_path}")
    logger.info(f"📁 Current directory: {os.getcwd()}")
    logger.info(f"📁 Files in current directory: {os.listdir('.')}")
    if os.path.exists(os.path.join(BASE_DIR, 'templates')):
        logger.info(f"📁 Files in templates: {os.listdir(os.path.join(BASE_DIR, 'templates'))}")

# ==========================
# 🌿 CLASS MODEL
# ==========================
class PlantDiseaseModel:
    def __init__(self, model_filename="best_b6_model.h5"):
        self.model_path = os.path.join(BASE_DIR, model_filename)
        self.model = None
        self.class_names = [
            'Apple__Apple_scab',
            'Apple__healthy',
            'Corn__Common_rust',
            'Corn__healthy'
        ]
        self.target_size = (224, 224)
        self.load_model()

    def load_model(self):
        logger.info(f"🔍 Đang tìm model tại: {self.model_path}")
        if not os.path.exists(self.model_path):
            logger.error(f"❌ KHÔNG tìm thấy model: {self.model_path}")
            logger.info(f"📁 Files trong thư mục hiện tại: {os.listdir(BASE_DIR)}")
            logger.info("⚠️ Bật chế độ DEMO (không có model thật).")
            self.model = None
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("✅ Model .keras loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Lỗi khi load model: {e}")
            self.model = None

    def preprocess_image(self, image):
        """Tiền xử lý ảnh"""
        try:
            img_array = np.array(image)

            # Convert RGB → BGR (OpenCV format)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Resize + normalize
            img_resized = cv2.resize(img_array, self.target_size)
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            return img_batch

        except Exception as e:
            logger.error(f"❌ Lỗi tiền xử lý ảnh: {e}")
            raise

    def predict(self, image):
        """Dự đoán ảnh"""
        try:
            if self.model is None:
                return self.demo_prediction()

            processed_img = self.preprocess_image(image)
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])

            return {
                'prediction': self.class_names[predicted_class_idx],
                'confidence': float(predictions[0][predicted_class_idx]),
                'all_predictions': {
                    self.class_names[i]: float(pred)
                    for i, pred in enumerate(predictions[0])
                }
            }

        except Exception as e:
            logger.error(f"❌ Lỗi dự đoán: {e}")
            return self.demo_prediction()

    def demo_prediction(self):
        """Trả kết quả ngẫu nhiên khi không có model"""
        predicted_class = random.choice(self.class_names)
        confidence = random.uniform(0.7, 0.95)
        return {
            'prediction': predicted_class,
            'confidence': confidence
        }

# ==========================
# 🚀 KHỞI TẠO MODEL
# ==========================
model = PlantDiseaseModel()

# ==========================
# 🌐 ROUTES
# ==========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được tải lên'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        result = model.predict(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500


# ==========================
# 🏁 MAIN
# ==========================
if __name__ == '__main__':
    logger.info(f"🚀 Flask app running in: {BASE_DIR}")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
