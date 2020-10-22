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

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   print(nparr.shape)
   img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
   print(img.shape, type(img))
   cv2.imwrite('decodedcv.png', img)
   # plt.imshow(img)
   # plt.colorbar()
   # plt.savefig('colorbar.png')
   # plt.close()
   #return img

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
# routes and related funcs
# =============================================================================================================

@app.get("/wakeup")
def get_initial_conditions(request: Request):
	"""
    revive backend if not up
	"""
	return {'status': 'up'}


@app.post("/predict")
def predict(resp: someResponse):
    readb64(resp.b64Image)
    return {"label": 'rat'}
