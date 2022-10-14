import numpy as np
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from io import BytesIO
import sys
import os
import gunicorn
import pickle
from audioutils import get_melspectrogram_db, spec_to_image
from io import BytesIO as by

device = torch.device('cpu')

with open('indtocat.pkl','rb') as f:
    indtocat = pickle.load(f)
x=[]
for i in range(5):
    with open(str(i),'rb') as f: x.append(f.read())

x = b''.join(x)

t1 = torch.load(by(x),map_location='cpu')
#resnet_model = torch.load('esc50resnet.pth', map_location=device)
resnet_model = t1
def predict_sound_from_bytes(resnet_model,indtocat, filename):
    spec=get_melspectrogram_db(filename)
    spec_norm = spec_to_image(spec)
    spec_t=torch.tensor(spec_norm).to(device, dtype=torch.float32)
    pr=resnet_model.forward(spec_t.reshape(1,1,*spec_t.shape))[0].cpu().detach().numpy()
    pred = {name:pr[ind] for ind,name in indtocat.items()}
    res = list(pred.items())
    s=0
    for c, val in res:
        s+=np.exp(val)
    for i in range(len(res)):
        res[i]=(res[i][0],np.exp(res[i][1])/s)
    res.sort(key=lambda x:x[1],reverse=True)
    res = res[0][0]
    return res

UPLOAD_FOLDER = '/'
ALLOWED_EXTENSIONS = {'mp3','wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1000 * 1000
app.config['SECRET_KEY'] = os.urandom(30).hex()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(filename)
            rez = predict_sound_from_bytes(resnet_model,indtocat, filename)
            os.remove(filename)
            return vrati(rez)
    return vrati()

def vrati(content=''):
    if content:
        return f'''
<!doctype html>
    <html style='background-image: linear-gradient(45deg,#3c75e0 0,#0956e6);
background-attachment: fixed;color:white;text-align:center'>
    <title>Predikcija</title>
    <button><a style='color:black;text-decoration: none' href='https://zika.ai.hr'>Home</a></button>
    <h1>{content}</h1>
    <form method=post enctype=multipart/form-data>
      <input accept="audio/wav,audio/mp3" type=file name=file>
      <input type=submit value=Upload>
    </form>
    </html>
    '''
    return '''
    <!doctype html>
    <html style='background-image: linear-gradient(45deg,#3c75e0 0,#0956e6);
background-attachment: fixed;color:white;text-align:center'>
    <title>Predikcija</title>
    <button><a style='color:black;text-decoration: none' href='https://zika.ai.hr'>Home</a></button>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input accept="audio/wav,audio/mp3" type=file name=file>
      <input type=submit value=Upload>
    </form>
    </html>
    '''
