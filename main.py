"""
    TODO:
    - Handle all the possible model outputs
    - Add GPU support
    - Add session RunOptions support
"""
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
import pickle
import base64
from typing import Union, Dict, List
import json

class DecodingException(Exception):

    def __init__(self, key: str, *args: object) -> None:
        super().__init__(*args)
        self.key = key

    def __str__(self) -> str:
        return f"Decoding error: unable to decode key {self.key}"

class RequestBody(BaseModel):
    input: Union[Dict, None] = None
    outputs: Union[List[str], None] = None

app = FastAPI()

MODEL_NAME = os.environ.get("MODEL_NAME", "model.onnx")
print(f"Using MODEL_NAME = {MODEL_NAME}")
MODEL_DIR = os.environ.get("MODEL_DIR", ".")
print(f"Using MODEL_DIR = {MODEL_DIR}")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
print(f"Complete MODEL_PATH = {MODEL_PATH}")

print("Loading Onnx Runtime session...", end=" ")
try:
    session = ort.InferenceSession(MODEL_PATH)
except Exception as ex:
    print("Unable to load session.")
    print(f"Exception details: {ex}")
    print("Exit.")
    exit(-1)
print("Done.")

session_inputs = session.get_inputs()
if session_inputs == None or len(session_inputs) <= 0:
    print("WARNING: Session has no inputs")
else:
    print(f"Session Input: {list(map(lambda i: i.name, session_inputs))}")

session_outputs = session.get_outputs()
if session_outputs == None or len(session_outputs) <= 0:
    print("WARNING: Session has no outputs")
print(f"Session Output: {list(map(lambda o: o.name, session_outputs))}")

def decode_input_string(b64_encoded: str) -> np.ndarray:
    """
    >>> a = np.array(...)
        s = pickle.dumps(a)
        b64_encoded = json.dumps(base64.b64encode(s).decode())
    """
    return pickle.loads(base64.b64decode(b64_encoded))

def decode_input_dict(b64_encoded_dict: dict) -> dict:
    decoded = {}
    for key in b64_encoded_dict.keys():
        try:
            decoded[key] = decode_input_string(b64_encoded_dict[key])
        except:
            print(f"ERROR: unable to decode input {key}.")
            raise DecodingException(key = key)
    return decoded

def encode_output_array(array: np.ndarray) -> str:
    s = pickle.dumps(array)
    b64_encoded = json.dumps(base64.b64encode(s).decode())
    return b64_encoded

def encode_output_list(output_list: list) -> list:
    encoded = list(map(lambda array: encode_output_array(array), output_list))
    return encoded

@app.get("/")
def health():
    return "OK", 200

@app.get("/session/inputs")
def get_session_inputs():
    return list(map(lambda i: {"name": i.name, "shape": i.shape, "type": i.type},
                    session_inputs))

@app.get("/session/outputs")
def get_session_outputs():
    return list(map(lambda o: {"name": o.name, "shape": o.shape, "type": o.type}, 
                    session_outputs))

@app.post("/predict")
def predict(requestBody: RequestBody):
    # Get encoded input dictionary
    input_dict = requestBody.input
    if input_dict == None or len(input_dict.keys()) <= 0:
        decoded_input_dict = {}
    else:
        # Decode input dictionary
        try:
            decoded_input_dict = decode_input_dict(input_dict)
        except DecodingException as ex:
            return HTTPException(status_code = 500, detail = str(ex))
    # Get output list
    output_list = requestBody.outputs
    if output_list != None and len(output_list) <= 0:
        output_list = None
    # Run Onnx Runtime session
    try:
        result = session.run(output_names = output_list, 
                             input_feed = decoded_input_dict)
    except Exception as ex:
        return HTTPException(status_code = 500, detail = f"Session run error. Exception {ex}")
    # Handle session result
    if result == None or len(result) <= 0:
        return []
    try:
        encoded = encode_output_list(result)
    except Exception as ex:
        return HTTPException(status_code = 500, detail = f"Result encoding error. Exception {ex}")
    return encoded
