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
from ray import serve
from datetime import datetime

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

default_app_name = f"onnx_serve_{datetime.fromtimestamp(datetime.now().timestamp())}" \
                    .replace("-", "_") \
                    .replace(" ", "_") \
                    .replace(":", "_")
APP_NAME = os.environ.get("APP_NAME", default_app_name)
print(f"Using App name = {APP_NAME}")

@serve.deployment(name = APP_NAME,
                  route_prefix="/",
                  user_config={
                      "model_name": "model.onnx",
                      "model_path": ".",
                  })
@serve.ingress(app=app)
class ONNX:

    MODEL_NAME = "model.onnx"
    MODEL_DIR = "."

    def __init__(self) -> None:
        self.model_name = self.MODEL_NAME
        self.model_dir = self.MODEL_DIR
        self.model_path = os.path.join(self.model_dir, self.model_name)
        
        print(f"Using model : {self.model_name}")
        print(f"Using model directory : {self.model_dir}")
        print(f"Complete model path = {self.model_path}")

        print("Loading Onnx Runtime session...", end=" ")
        try:
            self.session = ort.InferenceSession(self.model_path)
        except Exception as ex:
            print("Unable to load session.")
            print(f"Exception details: {ex}")
            print("Exit.")
            exit(-1)
        print("Done.")

        self.session_inputs = self.session.get_inputs()
        if self.session_inputs == None or len(self.session_inputs) <= 0:
            print("WARNING: Session has no inputs")
        else:
            print(f"Session Input: {list(map(lambda i: i.name, self.session_inputs))}")

        self.session_outputs = self.session.get_outputs()
        if self.session_outputs == None or len(self.session_outputs) <= 0:
            print("WARNING: Session has no outputs")
        print(f"Session Output: {list(map(lambda o: o.name, self.session_outputs))}")

    def reconfigure(self, config: Dict):
        self.model_name = config.get("model_name", self.MODEL_NAME)
        self.model_dir = config.get("model_dir", self.MODEL_DIR)
        self.model_path = os.path.join(self.model_dir, self.model_name)

    def decode_input_string(self, b64_encoded: str) -> np.ndarray:
        """
        >>> a = np.array(...)
            s = pickle.dumps(a)
            b64_encoded = json.dumps(base64.b64encode(s).decode())
        """
        return pickle.loads(base64.b64decode(b64_encoded))

    def decode_input_dict(self, b64_encoded_dict: dict) -> dict:
        decoded = {}
        for key in b64_encoded_dict.keys():
            try:
                decoded[key] = self.decode_input_string(b64_encoded_dict[key])
            except:
                print(f"ERROR: unable to decode input {key}.")
                raise DecodingException(key = key)
        return decoded

    def encode_output_array(self, array: np.ndarray) -> str:
        s = pickle.dumps(array)
        b64_encoded = json.dumps(base64.b64encode(s).decode())
        return b64_encoded

    def encode_output_list(self, output_list: list) -> list:
        encoded = list(map(lambda array: self.encode_output_array(array), output_list))
        return encoded

    @app.get("/")
    def health(self):
        return "OK", 200

    @app.get("/session/inputs")
    def get_session_inputs(self):
        return list(map(lambda i: {"name": i.name, "shape": i.shape, "type": i.type},
                        self.session_inputs))

    @app.get("/session/outputs")
    def get_session_outputs(self):
        return list(map(lambda o: {"name": o.name, "shape": o.shape, "type": o.type}, 
                        self.session_outputs))

    @app.post("/predict")
    def predict(self, requestBody: RequestBody):
        # Get encoded input dictionary
        input_dict = requestBody.input
        if input_dict == None or len(input_dict.keys()) <= 0:
            decoded_input_dict = {}
        else:
            # Decode input dictionary
            try:
                decoded_input_dict = self.decode_input_dict(input_dict)
            except DecodingException as ex:
                return HTTPException(status_code = 500, detail = str(ex))
        # Get output list
        output_list = requestBody.outputs
        if output_list != None and len(output_list) <= 0:
            output_list = None
        # Run Onnx Runtime session
        try:
            result = self.session.run(output_names = output_list, 
                                      input_feed = decoded_input_dict)
        except Exception as ex:
            return HTTPException(status_code = 500, detail = f"Session run error. Exception {ex}")
        # Handle session result
        if result == None or len(result) <= 0:
            return []
        try:
            encoded = self.encode_output_list(result)
        except Exception as ex:
            return HTTPException(status_code = 500, detail = f"Result encoding error. Exception {ex}")
        return encoded

onnx_deployment = ONNX.bind()
