from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.optimize import minimize
import numpy as np
import colorsys
import math

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
    r_normalized = rgb[0] / 255.0
    g_normalized = rgb[1] / 255.0
    b_normalized = rgb[2] / 255.0
    
    #RGB値をリニアRGB値に変換する関数
    def gamma_correct(c):
        return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4

    r_top = gamma_correct(r_normalized)
    g_top = gamma_correct(g_normalized)
    b_top = gamma_correct(b_normalized)
    
    
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
    delta_E = np.linalg.norm(lab1-lab2) 
    return delta_E

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
#rgbからhslに変換
def rgb_to_hsl(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360, s, l)  # 色相を0-360度に変換

#色相の関係から「組み合わせタイプ」を分類する

def get_hue_difference(h1, h2):
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)

def determine_harmony_type(h1, h2):
    diff = get_hue_difference(h1, h2)
    if diff < 15:
        #落ち着いた印象
        return "monochrome"
    elif diff <= 30:
        #自然な一体感がある
        return "analogous"
    elif 150 <= diff <= 210:
        #対比が強くてインパクト大
        return "complementary"
    elif abs(h1 - h2) % 120 < 15:
        #多色でもバランスよい
        return "triadic"
    else:
        return "neutral"

        #調和タイプ別スコア
harmony_scores = {
    "monochrome": 85,
    "analogous": 90,
    "complementary": 75,
    "triadic": 80,
    "neutral": 60
}
# 彩度・明度のバランス評価（差が大きいと減点）
def brightness_contrast_penalty(l1, l2, s1, s2):
    brightness_diff = abs(l1 - l2)
    saturation_diff = abs(s1 - s2)
    penalty = (brightness_diff + saturation_diff) * 30  # 最大30点引き
    return penalty
def evaluate_color_pair(rgb1, rgb2):
    # ΔEスコア（おしゃれ評価型）
    delta = delta_e(rgb1, rgb2)
    delta_score = delta_e_fashion_score(delta, ideal=20, width=10)  # 正規分布型

    # HSL調和スコア
    h1, s1, l1 = rgb_to_hsl(rgb1)
    h2, s2, l2 = rgb_to_hsl(rgb2)
    harmony_type = determine_harmony_type(h1, h2)
    base_harmony = harmony_scores[harmony_type]
    penalty = brightness_contrast_penalty(l1, l2, s1, s2)
    harmony_score = max(0, base_harmony - penalty)

    final_score = round(0.4 * delta_score + 0.6 * harmony_score, 1)
    return final_score

def delta_e_fashion_score(delta_e, ideal=20, width=10):
    """
    delta_e: 実測値
    ideal: 最も調和が取れると考えるΔEの値（15〜25が目安）
    width: 幅が広いほど、許容される色差の範囲が広くなる（標準偏差に相当）
    """
    score = math.exp(-((delta_e - ideal) ** 2) / (2 * width ** 2)) * 100
    return round(score, 1)





@app.post("/")
def get_bottom_with_delta(coordinate:Coordinate):
    top_rgb = coordinate.tops
    bottom_rgb = coordinate.bottoms
    recommend_bottom_rgb = find_bottom_rgb(top_rgb,target_delta_e = 25)
    actual_delta = delta_e(rgb_to_lab(np.array(top_rgb)),rgb_to_lab(bottom_rgb))
    degree_of_harmony = evaluate_color_pair(rgb_to_lab(np.array(top_rgb)),rgb_to_lab(bottom_rgb))
    return {"result": round(actual_delta,2),
            "harmony":degree_of_harmony
    }
    # return {
    #     "top_rgb":top_rgb,
    #     "bottom_rgb":bottom_rgb,
    #     "recommend_bottom_rgb":recommend_bottom_rgb,
    #     "delta_E":round(actual_delta,3),
    #     "difference_from_25":round(abs(actual_delta-25),3)
    # }

    
   