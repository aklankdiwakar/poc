from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

# List all images in the static/images directory
def get_image_list():
    image_folder = os.path.join('static', 'images')
    images = os.listdir(image_folder)
    return images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load-images/<int:page>', methods=['GET'])
def load_images(page):
    images = get_image_list()
    # Pagination logic - 9 images per page
    per_page = 9
    start = (page - 1) * per_page
    end = start + per_page
    return jsonify(images[start:end])

if __name__ == "__main__":
    app.run(debug=True)
