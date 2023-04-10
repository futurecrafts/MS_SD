from dotenv import load_dotenv
import os
import requests
import io
from PIL import Image

from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import fastapi as _fapi

load_dotenv()
HuggingFace_API_KEY = os.getenv("HP_API_KEY")

print(HuggingFace_API_KEY)
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
headers = {"Authorization": "Bearer "+ HuggingFace_API_KEY}

app = FastAPI()
origins = [
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Data(BaseModel):
    prompt: str
    numInfSteps: int
    guidanceScale: int
    seed: int

@app.api_route("/", methods=['GET', 'POST'])
async def index():
  return "Hello from AI!"

@app.api_route("/api/texttoimage", methods=['GET', 'POST'])
async def textToImage(data: Data):
  print('prompt:' + data.prompt)
  print('numInfSteps:' +  str(data.numInfSteps))
  print('guidanceScale:' + str(data.guidanceScale))
  print('seed:' + str(data.seed))

  prompt_JSON = {
      "inputs": data.prompt,
      "parameters": {
          "width": 512,
          "height": 512,
          "guidance_scale": 9,
          "num_inference_steps": 100,
          # "negative_prompt": 'low quality',
          # "num_images_per_prompt": 2
      }
  }

  response = requests.post(API_URL, headers=headers, json=prompt_JSON)
  # print(response.content)
  # if response.error:
  #   print(response.error)
  if response:
      print('got response!')
      image = Image.open(io.BytesIO(response.content))
      image.save("generation.png")
      memory_stream = io.BytesIO()
      image.save(memory_stream, format="PNG")
      memory_stream.seek(0)
      return StreamingResponse(memory_stream, media_type="image/png")

  #return {"success" : "Imagine completed!"}

# if __name__ == "__main__":
#   app.run(debug=True)
