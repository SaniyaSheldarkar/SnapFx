from flask import Flask, render_template, request, redirect, url_for
import os, uuid, base64, cv2
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Output folder
OUTPUT_FOLDER = 'static/output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ====== Utility Functions ======
def overlay_image_alpha(img, overlay, x, y):
    h, w = overlay.shape[:2]
    if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
        return
    for c in range(3):
        img[y:y+h, x:x+w, c] = (
            overlay[:, :, c] * (overlay[:, :, 3]/255.0) +
            img[y:y+h, x:x+w, c] * (1.0 - overlay[:, :, 3]/255.0)
        ).astype(np.uint8)

def save_output(image, style):
    filename = f"{style}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(path, image)
    return filename

def decode_image_from_base64(data_url):
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except:
        return None

# ====== Filters ======
def grey_sketch(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - grey
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(grey, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def watercolor(img): return cv2.stylization(img, sigma_s=60, sigma_r=0.6)
def negative(img): return cv2.bitwise_not(img)

def cartoon(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(grey, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def canny(img): return cv2.Canny(img, 100, 200)

def apply_funny_effect(frame):
    frame = cv2.bitwise_not(frame)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

    h, w = frame.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
    map_x += 10 * np.sin(map_y / 20)
    map_y += 10 * np.cos(map_x / 20)
    frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    glasses = cv2.imread('overlays/glasses.png', cv2.IMREAD_UNCHANGED)
    if glasses is not None and glasses.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(gray, 1.3, 5)
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            if i >= 2: break
            overlay_image_alpha(frame, cv2.resize(glasses, (ew, eh)), ex, ey)
    return frame

def apply_swirl_effect(image):
    h, w = image.shape[:2]
    center, strength = (w/2, h/2), 3.0
    radius = min(center)
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx, dy = x - center[0], y - center[1]
            r = np.sqrt(dx**2 + dy**2)
            if r < radius:
                theta = strength * (radius - r) / radius
                sin_t, cos_t = np.sin(theta), np.cos(theta)
                map_x[y, x] = dx * cos_t - dy * sin_t + center[0]
                map_y[y, x] = dx * sin_t + dy * cos_t + center[1]
            else:
                map_x[y, x], map_y[y, x] = x, y
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

def apply_mirror_effect(image):
    return cv2.flip(image, 1)

# ====== Routes ======
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    outputs = {}
    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_array = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            outputs = {
                "Grey Sketch": save_output(grey_sketch(image), "grey"),
                "Watercolor": save_output(watercolor(image), "watercolor"),
                "Negative": save_output(negative(image), "negative"),
                "Cartoon": save_output(cartoon(image), "cartoon"),
                "Canny": save_output(canny(image), "canny")
            }
        else:
            return render_template('upload.html', error="Invalid file type.")
    return render_template('upload.html', outputs=outputs)

@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    outputs = {}
    if request.method == 'POST':
        image = decode_image_from_base64(request.form.get('webcam_image'))
        if image is not None:
            outputs = {
                "Grey Sketch": save_output(grey_sketch(image), "grey"),
                "Watercolor": save_output(watercolor(image), "watercolor"),
                "Negative": save_output(negative(image), "negative"),
                "Cartoon": save_output(cartoon(image), "cartoon"),
                "Canny": save_output(canny(image), "canny")
            }
    return render_template('webcam.html', outputs=outputs)

@app.route('/funny', methods=['GET', 'POST'])
def funny():
    result_img = None
    if request.method == 'POST':
        image = decode_image_from_base64(request.form.get('webcam_image'))
        if image is not None:
            result_img = save_output(apply_funny_effect(image), "funny")
    return render_template('funny.html', result_img=result_img)

@app.route('/funny_snap', methods=['POST'])
def funny_snap():
    image = decode_image_from_base64(request.form.get('webcam_image'))
    if image is None:
        return redirect(url_for('home'))
    filename = save_output(apply_funny_effect(image), "funny_snap")
    return render_template('funny.html', result_img=filename)

@app.route('/swirl', methods=['POST'])
def swirl():
    image = decode_image_from_base64(request.form.get('webcam_image'))
    result_img = save_output(apply_swirl_effect(image), "swirl") if image is not None else None
    return render_template('funny.html', result_img=result_img)

@app.route('/mirror', methods=['POST'])
def mirror():
    image = decode_image_from_base64(request.form.get('webcam_image'))
    result_img = save_output(apply_mirror_effect(image), "mirror") if image is not None else None
    return render_template('funny.html', result_img=result_img)

# ====== Run Flask App ======
if __name__ == '__main__':
    app.run(debug=True)
