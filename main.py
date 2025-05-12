from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Coordinate(BaseModel):
    tops: list[int]
    bottoms: list[int]


@app.get("/")
def read_root(coordinate: Coordinate):
    return {"Result": 80}
