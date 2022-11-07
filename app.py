import librosa
import numpy as np
import audioread.ffdec
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import torch
import sys
import gunicorn
from io import BytesIO as by

modeli = os.listdir('models')
modeli = ['models/'+i for i in modeli]
models = []
for i in modeli:
    with open(i,'rb') as f:
        f.seek(0)
        models.append(f.read())
modeli = b''.join(models).split(b'This is model splitter')
device = torch.device('cpu')
def vek_brojac(l):
    x = l.copy()
    uk = len(x)
    x = [i for t in x for i in t]
    zanrovi = [i[0] for i in x]
    predikcije = [i[1] for i in x]
    k = list(dict.fromkeys(zanrovi))
    k = {i:0 for i in k}
    for i,j in x: k[i]+=j
    k = {i:j/uk for i,j in k.items()}
    sve = sorted([[i,j] for j,i in k.items()],reverse=True)
    sve = sve[0][1]
    return sve

filepath = './'
modeli_nazivi = ['res1.pt','res2.pt']
for i,j in zip(modeli_nazivi,modeli):
    with open(i,'wb') as f: f.write(j)
opcije = [modeli_nazivi,[modeli_nazivi[0]],[modeli_nazivi[1]]]
for j in opcije:
    try:
        resnet_model = [torch.load(i, map_location='cpu') for i in j]
    except _pickle.UnpicklingError: continue
assert resnet_model
print(len(resnet_model))

#resnet_model = [torch.load(by(i), map_location='cpu') for i in modeli]

def spec_to_image(specs, eps=1e-6):
    scaled = []
    for spec in specs:
        slika = spec.copy()
        if slika.shape != (300, 600):
            if 300 > slika.shape[1]:
                dorada = np.zeros((300,600-slika.shape[1]))
                slika = np.concatenate((slika,dorada),axis=1)
            else: slika = slika[...,:600]
        assert slika.shape == (300, 600)
    
        
        spec = slika.copy()
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        
        scaled.append(spec_scaled)
    return scaled

def get_melspectrogram_db(file_path, sr=44100, n_fft=2205, hop_length=441, n_mels=300, fmin=0, fmax=20000, top_db=200):
    aro = audioread.ffdec.FFmpegAudioFile(file_path)
    wav,sr = librosa.load(aro,sr=sr,mono=True)
    
    if wav.shape[0]<6*sr:
        wav=np.pad(wav,(0,6*sr-wav.shape[0]),mode='constant',constant_values=(0, 0))
        wavs = [wav]
    else:
        x = (wav.shape[0]-(6*sr))
        verzija = 'sve'
        if verzija =='random':
            cuttings = [randrange(0,x) for _ in range(10)]
            wavs = [wav[i:i+(6*sr)] for i in cuttings]
        else:
            c=0
            koliko = 0
            while c<x:
                c+=6*sr
                koliko+=1
            if koliko > 10:
                cut = (koliko-10)//2
                koliko = range(cut,cut+10)
                wavs = [wav[i*(6*sr):(i+1)*(6*sr)] for i in koliko]
                
            else:
                wavs = [wav[i*(6*sr):(i+1)*(6*sr)] for i in range(koliko)]
        
    specs = []
    for wav in wavs:
        spec=librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=top_db)
        specs.append(spec_db)
    
    return specs

def predict_sound_from_bytes(resnet_model,indtocat, filename):
    specs=get_melspectrogram_db(filename)
    spec_norms = spec_to_image(specs)
    sve = []
    for spec_norm in spec_norms:
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
        
        sve.append(res)
    
    return sve

UPLOAD_FOLDER = 'uploads/' 
ALLOWED_EXTENSIONS = {'wav','mp3'}

app = Flask(__name__,template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['TESTING'] = True
app.config['SECRET_KEY'] = os.urandom(30).hex()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = 'Upload new File'
    
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('drugi.html', data=[data])
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return render_template('drugi.html', data=[data])
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            file.save(filename)
            
            datas = []
            for i in resnet_model:
                data = predict_sound_from_bytes(i,i.labels, filename)
                datas.extend(data)
            
            data = vek_brojac(datas)
            os.remove(filename)
            
            return render_template('drugi.html', data=[data])
    return render_template('index.html')
