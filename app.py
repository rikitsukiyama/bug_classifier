import os
import glob
import numpy as np
from flask import Flask, request, redirect, url_for, flash, session, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from prediction import classifier

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

application = Flask(__name__)
application.secret_key = os.urandom(24)

application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

bug=classifier()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        files = glob.glob('static/images/*')
        for f in files:
            os.remove(f)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            print('return successful')
            return redirect(url_for('predict', filename=filename))
    return render_template('index.html')

@application.route('/predict')
def predict():
    filename = request.args.get('filename')
    print("Filename:", filename)
    path = 'static/images/{}'.format(filename)
    print(path)
    predictions = bug.predict(path)
    data = []
    # loop over the results and add them to the list of
    # returned predictions
    for result in predictions:
        imagenetID, label, prob = result
        data.append((label, np.round(prob*100,2) ))
    print(data)
    return render_template("upload.html", filename = filename, predictions = data)

@application.route('/')
def index():
    return render_template('index.html')

if __name__=='__main__':
    application.run(host='0.0.0.0', port=4000)
    #app.run(debug=True)
