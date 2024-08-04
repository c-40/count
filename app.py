# from flask import Flask, render_template, request, redirect, send_from_directory
# import os
# import cv2
# import numpy as np
# import base64
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
#
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
#
# class ImageSeg:
#     def __init__(self, img, threshold):
#         self.img = img
#         self.threshold = threshold
#
#     def color_filter(self):
#         hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
#         lower_bound = np.array([30, 40, 40])
#         upper_bound = np.array([90, 255, 255])
#         mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
#         filtered_img = cv2.bitwise_and(self.img, self.img, mask=mask)
#         return filtered_img
#
#     def preprocess_img(self, img):
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
#         _, thresh_img = cv2.threshold(blurred_img, self.threshold, 255, cv2.THRESH_BINARY)
#         return thresh_img
#
#     def post_process(self, thresh_img):
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
#         opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
#         return opened_img
#
#     def count_trees(self):
#         filtered_img = self.color_filter()
#         thresh_img = self.preprocess_img(filtered_img)
#         processed_img = self.post_process(thresh_img)
#         num_labels, _, _, _ = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
#         return num_labels - 1
#
#     def mark_trees(self):
#         filtered_img = self.color_filter()
#         thresh_img = self.preprocess_img(filtered_img)
#         processed_img = self.post_process(thresh_img)
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
#         marked_img = np.copy(self.img)
#
#         for i in range(1, num_labels):
#             x, y, w, h, _ = stats[i]
#             cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#         return marked_img
#
# def image_to_base64(image):
#     _, buffer = cv2.imencode('.jpg', image)
#     img_str = base64.b64encode(buffer).decode('utf-8')
#     return img_str
#
# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             try:
#                 file.save(filename)
#                 img = cv2.imread(filename)
#                 final_count = 0
#                 best_threshold = 0
#
#                 for thresh in range(0, 100, 5):
#                     obj = ImageSeg(img, thresh)
#                     count = obj.count_trees()
#                     if count > final_count:
#                         final_count = count
#                         best_threshold = thresh
#
#                 final_obj = ImageSeg(img, best_threshold)
#                 marked_img = final_obj.mark_trees()
#                 marked_img_base64 = image_to_base64(marked_img)
#
#                 return render_template('result.html', count=final_count, image_data=marked_img_base64)
#             finally:
#                 if os.path.exists(filename):
#                     os.remove(filename)
#     return render_template('upload.html')
#
# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class ImageSeg:
    def __init__(self, img, threshold):
        self.img = img
        self.threshold = threshold

    def color_filter(self):
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([30, 40, 40])
        upper_bound = np.array([90, 255, 255])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        filtered_img = cv2.bitwise_and(self.img, self.img, mask=mask)
        return filtered_img

    def preprocess_img(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, thresh_img = cv2.threshold(blurred_img, self.threshold, 255, cv2.THRESH_BINARY)
        return thresh_img

    def post_process(self, thresh_img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
        return opened_img

    def count_trees(self):
        filtered_img = self.color_filter()
        thresh_img = self.preprocess_img(filtered_img)
        processed_img = self.post_process(thresh_img)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        return num_labels - 1

    def mark_trees(self):
        filtered_img = self.color_filter()
        thresh_img = self.preprocess_img(filtered_img)
        processed_img = self.post_process(thresh_img)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        marked_img = np.copy(self.img)

        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return marked_img


def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            try:
                file.save(filename)
                img = cv2.imread(filename)
                final_count = 0
                best_threshold = 0

                for thresh in range(0, 100, 5):
                    obj = ImageSeg(img, thresh)
                    count = obj.count_trees()
                    if count > final_count:
                        final_count = count
                        best_threshold = thresh

                final_obj = ImageSeg(img, best_threshold)
                marked_img = final_obj.mark_trees()

                # Convert image to base64 for rendering in HTML
                marked_img_base64 = image_to_base64(marked_img)

                return render_template('result.html', count=final_count, image_data=marked_img_base64)
            finally:
                if os.path.exists(filename):
                    os.remove(filename)
    return render_template('upload.html')


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
