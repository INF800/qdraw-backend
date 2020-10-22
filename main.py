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
    b64Image: str

# -----------------------------------------
# Custom
# -----------------------------------------


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
def update_player_move(resp: someResponse):
    print(someResponse.b64Image)
