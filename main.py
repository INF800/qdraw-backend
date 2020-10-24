# ----------------------------------------
# create fastapi app 
# ----------------------------------------
from fastapi import FastAPI
app = FastAPI()

# ----------------------------------------
# dependency injection
# ----------------------------------------
from fastapi import Depends

def get_db():
	""" returns db session """
	try:
		db = SessionLocal()
		yield db
	finally:
		db.close


# ----------------------------------------
# define structure for requests (Pydantic & more)
# ----------------------------------------
from fastapi import Request # for get
from pydantic import BaseModel # for post
from typing import Optional

class someResponse(BaseModel):
    b64Image: str = None

# -----------------------------------------
# Custom
# -----------------------------------------
import numpy as np
import cv2
import tensorflow as tf
import base64
from classnames import classes

# load model and weights
model = tf.keras.applications.EfficientNetB0(input_shape=(64, 64, 1), weights="effnet.h5", classes=340)

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
   img = np.where((img == 1) | (img == 2), 255, 0) # !makes sure ctx stroke is `rgb(1,1,1)`
   return img

def preprocess_input(img):
    img = cv2.resize(img.astype('uint8'), (64,64))
    #cv2.imwrite('save.png', img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    return img # (1,w,h,c)

def expected(img):
    print(img.shape, model.input_shape)
    N=3
    preds = model.predict(img)
    # return {
    #     'preds': [
    #         ['Q', 0.9],
    #         ['B', 0.9],
    #         ['S', 0.9],
    #         ['M', 0.9],
    #         ['Z', 0.9],
    #     ]
    # }
    context = []
    for _id in preds.argsort()[0][::-1][:N]: # topN ids
        print('id:', _id)
        print('preds:', classes[_id], preds[0][_id])
        context.append([classes[_id], int(preds[0][_id]*1000)])
    return {'preds': context}

# -----------------------------------------
# CORS: List of servers to respond to...
# -----------------------------------------
from fastapi.middleware.cors import CORSMiddleware
origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    "https://rakesh4real.github.io",
    "http://rakesh4real.github.io",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:3003",
    "http://localhost:3004",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================================================
# =============================================================================================================
# routes and related funcs
# =============================================================================================================
# =============================================================================================================
@app.get("/wakeup")
def get_initial_conditions(request: Request):
	"""
    revive backend if not up
	"""
	return {'status': 'up'}



@app.post("/predict")
def predict(resp: someResponse):
    img = readb64(resp.b64Image)
    img = preprocess_input(img)
    preds = expected(img)
    return preds
