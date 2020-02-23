#Import necessary libraries
from VideoDetection import *
import json
import os
import sys
from predict import *
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2
sys.path.append('../../')
from drowsiness_detect import *
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/umanggupta/Downloads/My First Project-22ad1122a76b.json"
from google.cloud import automl
from config import *

project_id = PROJECT_ID
model_id = MODEL_ID
file_path = "/Users/umanggupta/Downloads/sleep.png"


def func():

    prediction_client = automl.PredictionServiceClient()

    # Get the full path of the model.
    model_full_id = prediction_client.model_path(
        project_id, "us-central1", model_id
    )

    # Read the file.
    with open(file_path, "rb") as content_file:
        content = content_file.read()

    image = automl.types.Image(image_bytes=content)
    payload = automl.types.ExamplePayload(image=image)

    return get_prediction(content, project_id, model_id)


def call_drowsy():
    # First detect the drowsiness
    drowsy()