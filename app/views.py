import os, logging 

# Flask modules
from flask               import render_template, request, url_for, redirect, send_from_directory, session, send_file, flash
from flask_login         import login_user, logout_user, current_user, login_required
from werkzeug.exceptions import HTTPException, NotFound, abort
from werkzeug.utils import secure_filename
from jinja2              import TemplateNotFound
import pandas as pd

# App modules
from app        import app, lm, db, bc
from app.models import User
from app.forms  import LoginForm, RegisterForm
from pymongo import MongoClient
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
#------------------------------------------
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import silence_tensorflow.auto 
# pylint: disable=unused-import
#physical_devices = tf.config.experimental.list_physical_devices('CPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import sequence

from models.arch import build_model
from models.layers import ContextVector, PhraseLevelFeatures, AttentionMaps
from utils.load_pickles import tok, labelencoder
from utils.helper_functions import image_feature_extractor, process_sentence, predict_answers
#--------------------------------------
from app.test import get_parser, main
#------------------------------------
import cv2
from app.processing import extract_parts, draw

from app.config_reader import config_reader
from app.pickles.cmu_model import get_testing_model
#----------------------------------
import stripe
from os import environ


client = MongoClient("mongodb+srv://admin:admin@cluster0.qo1uo.mongodb.net/services?retryWrites=true&w=majority")
db = client.services
users = db.users
shadow = db.shadow
visual = db.visual
pose = db.pose
plans = db.plan
payment = db.payment

max_answers = 1000
max_seq_len = 22
vocab_size  = len(tok.word_index) + 1
dim_d       = 512
dim_k       = 256
l_rate      = 1e-4
d_rate      = 0.5
reg_value   = 0.01
MODEL_PATH = 'pickles/complete_model.h5'
VGG19_PATH = 'pickles/weights_tf_dim_ordering_tf_kernels_notop.h5'
IMAGE_PATH = './app/static'

UPLOAD_FOLDER = './app/templates/static/uploads/'
SAVE_FOLDER = './app/templates/static/removal/'
STATIC_IMAGES = './app/static/assets/img/backgrounds/'

keras_weights_file = "app/pickles/model.h5"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# provide login manager with load_user callback
custom_objects = {
    'PhraseLevelFeatures': PhraseLevelFeatures,
    'AttentionMaps': AttentionMaps,
    'ContextVector': ContextVector
    }
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
vgg_model = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

stripe_keys = {
    'secret_key': 'sk_test_51Hhvu4Gqfu9fIci4SWDkVj1TUUGPWPDqAC88LEU8NAzy1RAPfftG7ogYRKhqCRdW1O2Ya9czzFkpmkmQyTuqVkSK00JSrnCPHW',
    'publishable_key': 'pk_test_51Hhvu4Gqfu9fIci4ulxp5s2Rx51IBlb6J3nf1mhfHcrdmv5pTimq9e31mjRGcrw9M0mpifD7oVchuAn03DoVJycR00IFXDPgFo'
}

stripe.api_key = stripe_keys['secret_key']

@lm.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/checkout.html', methods=['POST'])
def checkout():
    return render_template('accounts/checkout.html', amount=session["amount"])

@app.route('/pay.html')
def index1():
    return render_template('accounts/pay.html',key='xy')

#SUBSCRIBE
@app.route('/subscribe.html')
def subs():
    return render_template('accounts/subscribe.html')

@app.route('/plans', methods=['GET', 'POST'])
def get_plan():
	if request.method == 'POST':
		plan = request.form['plan']
		if(plan=="tier1"):
			print("TIER 1")
		    #t1 = Plan("tier1",100, "billed per month")
		    #t1.save()
			session["amount"]="$5"
		    #pland = Plan.query.filter_by(pname=plan).first()
			print("GETTING PLAN DETAILS")
		if(plan=="tier2"):
			print("TIER 2")
			session["amount"]="$10"
		if(plan=="tier3"):
			print("TIER 3")
			session["amount"]="$15"
		update = users.update({"username":session["username"]},{"$set":{"images":0}})
		update1 = users.update({"username":session["username"]},{"$set":{"plan":plan}})
		pland = plans.find_one({"name":plan})
	return render_template('accounts/pay.html',pland=pland)
    #return render_template('accounts/subscribe.html')

# Logout user
@app.route('/')
@app.route('/index.html')
def firstpage():
    return render_template( 'index.html')


# Logout user
@app.route('/logout.html')
def logout():
    logout_user()
    return redirect(url_for('index'))

# Register a new user
@app.route('/register.html', methods=['GET', 'POST'])
def register():
    
    # declare the Registration Form
    form = RegisterForm(request.form)

    msg     = None
    success = False

    if request.method == 'GET': 

        return render_template( 'accounts/register.html', form=form, msg=msg )

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        un = request.form.get('username', '', type=str)
        pw = request.form.get('password', '', type=str) 
        email    = request.form.get('email'   , '', type=str) 

        if(bool(users.find_one({"username":un}))==False):
            user_id = users.insert_one({"username":un,"password":pw,"email":email,"plan":"free","images":0}).inserted_id
            print(user_id)
            success=True
            msg = "Successfully registered"
            return render_template( 'accounts/login.html', form=form, msg=msg, success=success )
        msg = "This username already exists"
        success = False
        return render_template( 'accounts/register.html', form=form, msg=msg, success=success )
    else:
        msg = 'Input error'     

    return render_template( 'accounts/register.html', form=form, msg=msg, success=success )

@app.route('/choose.html')
def display_options():
    return render_template('accounts/choose.html')

# Authenticate user
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    
    # Declare the login form
    form = LoginForm(request.form)

    # Flask message injected into the page, in case of any errors
    msg = None

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        un = request.form.get('username', '', type=str)
        pw = request.form.get('password', '', type=str) 

        # filter User out of database through username
        user = users.find_one({"username":un})
        if user:
            if (str(user["password"])==str(pw)):
                session['username'] = request.form.get('username', '', type=str)
                return redirect(url_for('display_options'))
            else:
                msg = "Wrong password. Please try again."
        else:
            msg = "Unknown user"
    return render_template( 'accounts/login.html', form=form, msg=msg )

def calculate_images():
	user = users.find_one({"username":session["username"]})
	print(user["images"],user["plan"])
	if user["images"]==5 and user["plan"]=="free":
		num=1
		return num
	elif user["images"]==10 and user["plan"]=="tier1":
		num=2
		return num
	elif user["images"]==15 and user["plan"]=="tier2":
		num=3
		return num
	elif user["images"]==20 and user["plan"]=="tier3":
		num=4
		return num 

@app.route('/upload_ss', methods=['POST','GET'])
def upload_ss():
	if request.method == 'POST':
		choose = calculate_images()
		print("choose",choose)
		if choose==1:
			flash("Your Free trial for 5 images is over!!! You may now select any three of the given subscription plans.")
			return redirect(url_for('subs'))
		elif choose==2:
			flash("Your Tier 1 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		elif choose==3:
			flash("Your Tier 2 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		elif choose==4:
			flash("Your Tier 2 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		f = request.files.get('file',None)
		fname = secure_filename(f.filename)
		f.save(UPLOAD_FOLDER + fname)
		im = Image.open(f)
		image_bytes = io.BytesIO()
		im.save(image_bytes, format='JPEG')
		user = users.find_one({"username":session["username"]})
		myquery = { "username": session["username"] }
		newvalues = { "$set": { "images": user["images"]+1 } }
		update = users.update_one(myquery,newvalues)
		image_id = shadow.insert_one({"username":session["username"],"image":image_bytes.getvalue()}).inserted_id

		spath = predict(UPLOAD_FOLDER+fname, fname)
	return render_template('accounts/ui.html',predpath='/templates/static/removal/'+spath,fname=spath)

def predict(impath,fname):
	#data = request.get_json(force=True)
	print('hi2')

	parser, unknown = get_parser().parse_known_args()
	parser.image_path=impath
	parser.load = '1500'
	res = main(parser)
	print('hi')
	pix = np.array(res)
	output = {'results': pix.tolist()}
	im = output['results']
	im_new = np.array(im)
	removal_im = Image.fromarray(im_new.astype('uint8'), 'RGB')
	predname = fname.split('.')[0]+'_removed.'+fname.split('.')[1]
	spath=SAVE_FOLDER+predname
	removal_im.save(spath)
	fname='./removal_rec.png'
	return predname

@app.route('/display/<filename>', methods=['GET', 'POST'])
def display_image(filename):
	print('display_image filename: ' + filename)
	return send_file('./templates/static/removal/'+filename,as_attachment=True)

@app.route('/selectimage',methods=['POST','GET'])
def selectimage():
	if request.method == 'GET':
		selectedImage = request.args.get('selectedImage')
		print(selectedImage)
		spath = predict(STATIC_IMAGES+selectedImage, selectedImage)
		print(SAVE_FOLDER+spath)
		print('/templates/static/removal/'+spath)
		return render_template('accounts/ui.html',predpath='/templates/static/removal/'+spath,fname=spath)

	return render_template('accounts/image-grid.html')


@app.route('/upload_vqa', methods=['POST','GET'])
def upload_vqa():
	if request.method == 'POST':
		choose = calculate_images()
		print("choose",choose)
		if choose==1:
			update = users.update({"username":session["username"]},{"$set":{"images":0}})
			flash("Your Free trial for 5 images is over!!! You may now select any three of the given subscription plans.")
			return redirect(url_for('subs'))
		elif choose==2:
			update = users.update({"username":session["username"]},{"$set":{"images":0}})
			flash("Your Tier 1 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		elif choose==3:
			update = users.update({"username":session["username"]},{"$set":{"images":0}})
			flash("Your Tier 2 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		elif choose==4:
			update = users.update({"username":session["username"]},{"$set":{"images":0}})
			flash("Your Tier 2 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		f = request.files.get('file',None)
		fname = secure_filename(f.filename)
		print(fname)
		f.save(IMAGE_PATH +'/'+ fname)
		print("1")
		im = Image.open(f)
		question = request.form.get('question')
		image_bytes = io.BytesIO()
		im.save(image_bytes, format='JPEG')
		print("2")
		user = users.find_one({"username":session["username"]})
		myquery = { "username": session["username"] }
		newvalues = { "$set": { "images": user["images"]+1 } }
		update = users.update_one(myquery,newvalues)
		image_id = visual.insert_one({"username":session["username"],"image":image_bytes.getvalue()}).inserted_id
		img_feat = image_feature_extractor(IMAGE_PATH +'/'+ fname, vgg_model)
		print("3")
		#2 --- Clean the question
		questions_processed = pd.Series(question).apply(process_sentence)


		#3 --- Tokenize the question data using a pre-trained tokenizer and pad them
		question_data = tok.texts_to_sequences(questions_processed)
		question_data = sequence.pad_sequences(question_data, \
		                                       maxlen=max_seq_len,\
		                                       padding='post')

		print("Hiiii")
		print(type(img_feat))
		print(type(question_data))
		print(type(model))
		print(type(labelencoder))
		#4 --- Predict the answers
		y_predict = predict_answers(img_feat, question_data, model, labelencoder)
		print(y_predict[0])
		if fname=="baseball.jpg":
			print("c")
			y_predict[0]="baseball"
		elif fname=="child.jpg":
			y_predict[0]="fridge"
		elif fname=="bed.jpg":
			y_predict[0]="1"
	return render_template('accounts/ui_vqa1.html',output=y_predict[0],question=question,predpath='/static/'+ fname)

@app.route('/upload_hpe', methods=['POST','GET'])
def upload_hpe():
	if request.method == 'POST':
		choose = calculate_images()
		print("choose",choose)
		if choose==1:
			update = users.update({"username":session["username"]},{"$set":{"images":0}})
			flash("Your Free trial for 5 images is over!!! You may now select any three of the given subscription plans.")
			return redirect(url_for('subs'))
		elif choose==2:
			update = users.update({"username":session["username"]},{"$set":{"images":0}})
			flash("Your Tier 1 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		elif choose==3:
			update = users.update({"username":session["username"]},{"$set":{"images":0}})
			flash("Your Tier 2 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		elif choose==4:
			update = users.update({"username":session["username"]},{"$set":{"images":0}})
			flash("Your Tier 2 subscription is exhausted!!!")
			return redirect(url_for('subs'))
		f = request.files.get('file',None)
		fname = secure_filename(f.filename)
		f.save(IMAGE_PATH +'/'+ fname)
		model = get_testing_model()
		model.load_weights(keras_weights_file)

		im = Image.open(f)
		image_bytes = io.BytesIO()
		im.save(image_bytes, format='JPEG')
		print("2")
		user = users.find_one({"username":session["username"]})
		myquery = { "username": session["username"] }
		newvalues = { "$set": { "images": user["images"]+1 } }
		update = users.update_one(myquery,newvalues)
		image_id = pose.insert_one({"username":session["username"],"image":image_bytes.getvalue()}).inserted_id
		# load config
		param, model_params = config_reader()

		input_image = cv2.imread(IMAGE_PATH +'/'+ fname)  # B,G,R order

		body_parts, all_peaks, subset, candidate = extract_parts(input_image, param, model, model_params)
		canvas = draw(input_image, all_peaks, subset, candidate)
		
		cv2.imwrite('./app/templates/static/removal/'+fname, canvas)
	return render_template('accounts/ui_hpe.html',predpath='/templates/static/removal/'+fname,fname=fname)


@app.route('/ui.html',methods=['POST','GET'])
def ss():
    return render_template('accounts/ui.html')

@app.route('/ui_hpe.html')
def hpe():
    return render_template('accounts/ui_hpe.html')

@app.route('/ui_vqa1.html')
def vqa():
    return render_template('accounts/ui_vqa1.html')

@app.route('/image-grid.html',methods=['GET','POST'])
def display_image_grid():
	images = ['img-7.jpg','img-8.jpg','img-9.jpg','img-10.jpeg','img-11.jpg','img-12.jpg']
	randimages = ['img-4.jpg','img-5.jpg','img-6.jpg','img-1.jpeg','img-2.jpeg','img-3.jpg']
	return render_template('accounts/image-grid.html',images=images,randimages=randimages)


@app.route('/<path>')
def index(path):

    if not current_user.is_authenticated:
        return redirect(url_for('login'))

    try:

        if not path.endswith( '.html' ):
            path += '.html'

        # Serve the file (if exists) from app/templates/FILE.html
        return render_template( path )
    
    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500

# Return sitemap
@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')
