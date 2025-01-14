from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
from face_recognition import verify_face_from_id

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        address = request.form['address']
        cnp = request.form['cnp']

        # Retrieve and save uploaded files
        id_image = request.files['id_image']
        selfie_image = request.files['selfie_image']

        if id_image and selfie_image:
            id_filename = secure_filename(id_image.filename)
            selfie_filename = secure_filename(selfie_image.filename)
            id_image_path = os.path.join(app.config['UPLOAD_FOLDER'], id_filename)
            selfie_image_path = os.path.join(app.config['UPLOAD_FOLDER'], selfie_filename)
            id_image.save(id_image_path)
            selfie_image.save(selfie_image_path)

            # Proceed to verify the images
            match = verify_face_from_id(id_image_path, selfie_image_path)
            if match:
                return "Verification successful!"
            else:
                return "Verification failed. The images do not match."

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)