from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.optimize import minimize
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Coordinate(BaseModel):
    tops: list[int]
    bottoms: list[int]

def rgb_to_lab(rgb):
    #RGB値をリニアRGB値に変換する関数
    def gamma_correct(c):
        return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4

    r_top = gamma_correct(rgb[0])
    g_top = gamma_correct(rgb[1])
    b_top = gamma_correct(rgb[2])
    
    
    #リニアRGB値をXYZ値に変換する
    X_t = 0.4124*r_top + 0.3576*g_top + 0.1805*b_top
    Y_t = 0.2126*r_top + 0.7152*g_top + 0.0722*b_top
    Z_t = 0.0193*r_top + 0.1192*g_top + 0.9505*b_top

    #XYZ値をLab値に変換する
    #定数の定義
    Xn = 0.9505
    Yn = 1.0000
    Zn = 1.0890

    delta = 6/29
    delta3 = delta**3
    inv_3delta2 = 1/(3*delta**2)
    #XYZ値をLab値に変換する関数の定義
    def f(t):
        if t > delta3:
            return t**(1/3)
        else:
            return t * inv_3delta2 + 4 / 29
    
    fx = f(X_t/Xn)
    fy = f(Y_t/Yn)
    fz = f(Z_t/Zn)

    #topのLabを算出
    L_top = 116 * fy - 16
    a_top = 500 * (fx-fy)
    b_top = 200 * (fy-fz)
    return np.array([L_top, a_top, b_top])
#topとbottomのLabからdelta_e（色差）を算出
def delta_e(lab1,lab2):
    return np.linalg.norm(lab1-lab2)

#topから適したbottomを計算
def find_bottom_rgb(top_rgb,target_delta_e=25):
        top_lab = rgb_to_lab(np.array(top_rgb))
        #色差が２５に近くなるようなbottomを出す関数
        def objective(bottom_rgb):
            bottom_rgb = np.clip(bottom_rgb,0,255)
            bottom_lab = rgb_to_lab(bottom_rgb)
            return abs(delta_e(top_lab,bottom_lab)-target_delta_e)

        res = minimize(objective, x0=[128, 128, 128], bounds=[(0, 255)]*3)
        return np.clip(res.x.round(), 0, 255).astype(int).tolist()

@app.post("/")
def get_bottom_with_delta(coordinate:Coordinate):
    top_rgb = coordinate.tops
    bottom_rgb = coordinate.bottoms
    recommend_bottom_rgb = find_bottom_rgb(top_rgb,target_delta_e = 25)
    actual_delta = delta_e(rgb_to_lab(np.array(top_rgb)),rgb_to_lab(bottom_rgb))
    return {"result":recommend_bottom_rgb}
    # return {
    #     "top_rgb":top_rgb,
    #     "bottom_rgb":bottom_rgb,
    #     "recommend_bottom_rgb":recommend_bottom_rgb,
    #     "delta_E":round(actual_delta,3),
    #     "difference_from_25":round(abs(actual_delta-25),3)
    # }

    
   
