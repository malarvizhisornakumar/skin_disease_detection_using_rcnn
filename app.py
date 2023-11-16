from flask import Flask, render_template, request, redirect, flash, url_for, session
from werkzeug.utils import secure_filename
import os
import MySQLdb
import cv2
import numpy as np
from PIL import Image as PILImage
import sys
import glob
import re
from pathlib import Path
from io import BytesIO
import base64
import requests
import subprocess
# Import fast.ai Library
from fastai import *
from fastai.vision import *


#IMAGES_FOLDER = '/RainfallPrediction/test'

app = Flask(_name_)
app.secret_key = "secret key"
#app.config['IMAGES_FOLDER'] = IMAGES_FOLDER
NAME_OF_FILE = 'model_best'  # Name of your exported file
PATH_TO_MODELS_DIR = Path('')  # by default just use /models in root dir
classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']
@app.route('/')
@app.route("/index")
def index():
    return render_template('index.html')

def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
    data = ImageDataBunch.single_from_classes(
        path_to_pth_file, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data, models.densenet169, model_dir='models')
    learn.load(learner_name_to_load, device=torch.device('cpu'))
    return learn


learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)
@app.route("/newuser", methods=["GET","POST"])
def new_user():
    if request.method == "POST":
        db = MySQLdb.connect("localhost", "root", "", "SkinDisease")
        c1 = db.cursor()
        name = request.form["name"]
        mailid = request.form["mailid"]
        mobileno = request.form["mobileno"]
        uid = request.form["uid"]
        pwd = request.form["pwd"]
        c1.execute("INSERT INTO NewUser VALUES ('%s','%s','%s','%s','%s')" % (name, mailid, mobileno, uid, pwd))
        db.commit()
        return render_template("newuser.html", msg="User Details Registered!!!")
    return render_template("newuser.html")

@app.route("/userlogin", methods=["GET","POST"])
def user_login():
    if request.method == "POST":
        db = MySQLdb.connect("localhost", "root", "", "SkinDisease")
        c1 = db.cursor()
        uid=request.form["uid"]
        pwd=request.form["pwd"]
        c1.execute("select * from NewUser where userid='%s' and password='%s'"%(uid,pwd))
        if c1.rowcount>=1:
            row=c1.fetchone()
            session["userid"]=uid
            return render_template("userhome.html")
        else:
            return render_template("userlogin.html", msg="Your Login attempt was not successful. Please try again!!")
    return render_template("userlogin.html")

@app.route("/myprofile")
def my_profile():
    db=MySQLdb.connect("localhost","root","","SkinDisease")
    c1 = db.cursor()
    uid=str(session["userid"])
    c1.execute("select Name,MailId,MobileNo from NewUser where userid='%s'"%uid)
    if c1!=None:
        row=c1.fetchone()
        return render_template("myprofile.html", name=row[0], mailid=row[1], mobileno=row[2])

@app.route("/adminlogin", methods=["GET","POST"])
def admin_login():
    if request.method == "POST":
        uid=request.form["uid"]
        pwd=request.form["pwd"]

        if uid=="Admin" and pwd=="Admin":
            return render_template("adminhome.html")
        else:
            return render_template("adminlogin.html", msg="Your Login attempt was not successful. Please try again!!")
    return render_template("adminlogin.html")

@app.route("/viewusers")
def view_users():
    db = MySQLdb.connect("localhost", "root", "", "MaterialDetection")
    c1 = db.cursor()
    c1.execute("select * from NewUser")
    data = c1.fetchall()
    return render_template("viewusers.html", data=data)

@app.route("/prediction", methods=["GET","POST"])
def prediction():
    subprocess.run("python D:\\skindisease1\\Skin_cancer_Detection.py")

    return render_template("prediction.html")








def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def model_predict(img):
    img = open_image(BytesIO(img))
    pred_class, pred_idx, outputs = learn.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in
                         [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
        zip(learn.data.classes, map(str, formatted_outputs)),
        key=lambda p: p[1],
        reverse=True
    )

    img_data = encode(img)
    result = {"class": pred_class, "probs": pred_probs, "image": img_data}
    return render_template('result.html', result=result)


@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file'].read()
        if img != None:
            # Make prediction
            preds = model_predict(img)
            return preds
    return render_template('predict.html')

@app.route("/classify-url", methods=["POST", "GET"])
def classify_url():
    if request.method == 'POST':
        url = request.form["url"]
        if url != None:
            response = requests.get(url)
            preds = model_predict(response.content)
            return preds
    return 'OK'

@app.route("/signout")
def signout():
    session.clear()
    return redirect(url_for("index"))

if _name_ == "_main_":
    app.run(debug=True)
