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
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
#from tf.keras.applications.efficientnet import preprocess_input

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   print('BEF:',nparr.shape)
   img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
   print('AF:',img.shape)
   img = np.where((img == 1) | (img == 2), 255, 0) # !makes sure ctx stroke is `rgb(1,1,1)`
   return img

def preprocess_input(img):
    img = cv2.resize(img, (64,64))
    # todo: img = preprocess_input(img) (using built-in)

def expected(img):
    # todo: convert to (1,w,h,1)
    # todo: get top 5 scores
    return {
        1: ['Q', 0.9],
        2: ['B', 0.9],
        3: ['S', 0.9],
        4: ['M', 0.9],
        5: ['Z', 0.9],
    }

# -----------------------------------------
# CORS: List of servers to respond to...
# -----------------------------------------
from fastapi.middleware.cors import CORSMiddleware
origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
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
    # img = preprocess_input(img)
    preds = expected(img)
    return preds
