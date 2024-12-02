from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd

app = FastAPI()

with open("models.pkl", "rb") as file:
    data = pickle.load(file)
    model = data["model"]
    preprocessor = data["preprocessor"]


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    max_torque: float
    torque: float
    seats: str


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    preprocessed_data = preprocessor.transform(df)
    return model.predict(preprocessed_data)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame([item.dict() for item in items])
    preprocessed_data = preprocessor.transform(df)
    return model.predict(preprocessed_data)