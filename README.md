## Backend for Doodle Recognition

Run `python3 -m uvicorn main:app --port 8008 --reload`. And then go to [frontend code](https://github.com/rakesh4real/qdraw) directory and run `npm start`


> Cannot host in Heroku as it takes maximum slug size of `500 MB`. This project generates `525 MB` (Model is small and efficient but installing packages like TF and opencv increase slug size)

![](https://rakesh4real.github.io/whoami/assets/projs/qdraw.gif)
