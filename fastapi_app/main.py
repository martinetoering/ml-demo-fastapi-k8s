import json
from typing import List

import numpy as np
import numpy.typing as npt
import onnxruntime
import torch
import torchvision
from fastapi import FastAPI, File, Response
from pydantic import BaseModel
from transforms.videotransforms import VideoTransform

app = FastAPI(title="Video Classification FastAPI app")

session = onnxruntime.InferenceSession("model_onnx/R2+1D_18_K400Tiny_v0.1.onnx")

with open("model_onnx/id_by_label.json", "r") as file:
    id_by_label = json.load(file)
label_by_id = {v: k for k, v in id_by_label.items()}


class Prediction(BaseModel):
    """The model prediction returned by the app.

    Args:
        BaseModel (pydantic.BaseModel): Simple class for data validation.
    """

    label: str
    score: str


def process_video(video_file: bytes = File(...)) -> npt.NDArray[np.float32]:
    """Takes a video file in bytes and converts to numpy ndarray.

    Args:
        video_file (bytes): The video file.

    Returns:
        npt.NDArray[np.float32]: The video as numpy array.
    """
    transform = VideoTransform(8, "test", 128, 112)
    org_vid = torchvision.io.VideoReader(src=video_file)
    frames = []
    for frame in org_vid:
        frames.append(frame["data"])
    org_vid_tensor = torch.stack(frames, 0)
    vid_tensor = transform(org_vid_tensor).unsqueeze(0)
    np_vid = vid_tensor.numpy()
    return np_vid.astype(np.float32)


def get_prediction(data: npt.NDArray[np.float32]) -> List[npt.NDArray[np.float32]]:
    """Returns prediction from model in onnx format based on input.

    Args:
        data (npt.NDArray[np.float32]): The input to the model.

    Returns:
        List[npt.NDArray[np.float32]]: Prediction returned from the model.
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: data})


@app.get("/")
async def root() -> Response:
    """Welcome message for the API.

    Returns:
        Response: Fastapi response.
    """
    return Response("Video Classification FastAPI app")


@app.post("/predict/")
async def predict_video(video_file: bytes = File(...)) -> Prediction:
    """Return model prediction on provided video file.

    Args:
        video_file (bytes): The video file.

    Returns:
        Prediction: The label and score prediction.
    """
    data = process_video(video_file)
    result = get_prediction(data)

    predictions = np.squeeze(result)
    pred_id = int(predictions.argmax())
    pred_name = str(label_by_id.get(pred_id))

    return Prediction(
        label=pred_name,
        score=f"{predictions.max():.3f}",
    )
