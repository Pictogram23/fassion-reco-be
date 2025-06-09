from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.optimize import minimize
from enum import Enum
import numpy as np
import colorsys
import math
from typing import Optional


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "https://153.121.51.21",
    "https://iroca.ddns.net"
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
    season: Optional[str] = None



class CommentModel(str, Enum):
    SUBDUED = "控えめな印象です"
    BALANCED = "調和のとれたコーデです"
    FANCY = "ポップで派手な印象です"
    UNBALANCED = "アンバランスな印象です"


def rgb_to_lab(rgb):
    
    if isinstance(rgb, (float, int)):
        rgb = [rgb, rgb, rgb]

    rgb = list(rgb[:3]) 

    r_normalized = rgb[0] / 255.0
    g_normalized = rgb[1] / 255.0
    b_normalized = rgb[2] / 255.0

    # RGB値をリニアRGB値に変換する関数
    def gamma_correct(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_top = gamma_correct(r_normalized)
    g_top = gamma_correct(g_normalized)
    b_top = gamma_correct(b_normalized)

    # リニアRGB値をXYZ値に変換する
    X_t = 0.4124 * r_top + 0.3576 * g_top + 0.1805 * b_top
    Y_t = 0.2126 * r_top + 0.7152 * g_top + 0.0722 * b_top
    Z_t = 0.0193 * r_top + 0.1192 * g_top + 0.9505 * b_top

    # XYZ値をLab値に変換する
    # 定数の定義
    Xn = 0.9505
    Yn = 1.0000
    Zn = 1.0890

    delta = 6 / 29
    delta3 = delta**3
    inv_3delta2 = 1 / (3 * delta**2)

    # XYZ値をLab値に変換する関数の定義
    def f(t):
        if t > delta3:
            return t ** (1 / 3)
        else:
            return t * inv_3delta2 + 4 / 29

    fx = f(X_t / Xn)
    fy = f(Y_t / Yn)
    fz = f(Z_t / Zn)

    # topのLabを算出
    L_top = 116 * fy - 16
    a_top = 500 * (fx - fy)
    b_top = 200 * (fy - fz)
    return np.array([L_top, a_top, b_top])
#topとbottomのLabからdelta_e（色差）を算出
def delta_e(lab1, lab2):
    lab1 = np.array(lab1)  # 💡ここが大事！
    lab2 = np.array(lab2)
    return np.linalg.norm(lab1 - lab2)



#rgbからhslに変換
def rgb_to_hsl(rgb):
    r, g, b = [x / 255.0 for x in rgb[:3]] 
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360, s, l)  # 色相を0-360度に変換

def get_hue_difference(h1, h2):
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)



# --- 色相スコア（補色でも評価） ---
def get_hue_similarity_score(hue_diff):
    if hue_diff < 30:
        return 100  # アナログ配色
    elif 150 <= hue_diff <= 210:
        return 90   # 補色配色も評価
    elif abs(hue_diff - 120) < 20:
        return 75   # 三分割（トライアド）
    else:
        return max(0, 100 - (hue_diff / 180) * 100)


#明度の評価。 標準偏差と範囲のバランスでスコア化。緩やかな変化を高評価
def lightness_gradient_score(rgb1, rgb2):
    # Lab色空間に変換
    l1 = rgb_to_lab(rgb1)[0] * 100  # L*値に換算（正規化されていないと仮定）
    l2 = rgb_to_lab(rgb2)[0] * 100
    std = np.std([l1, l2])
    rng = abs(l1 - l2)
    # 標準偏差スコアは中心値30±10に設定（±20だと評価が緩すぎる）
    std_score = math.exp(-((std - 20) ** 2) / (2 * 12 ** 2))  # L*差ベースで20が中心
    # 明度差の評価：rangeが20〜50なら高評価（中心35）
    range_score = math.exp(-((rng - 50) ** 2) / (2 * 15 ** 2))  # 中心50, σ=15
    return 50 * std_score + 50 * range_score


def delta_e_fashion_score(delta_e, ideal=25, width=15):
    """
    delta_e: 実測値
    ideal: 最も調和が取れると考えるΔEの値（15〜25が目安）
    width: 幅が広いほど、許容される色差の範囲が広くなる（標準偏差に相当）
    """
    score = math.exp(-((delta_e - ideal) ** 2) / (2 * width ** 2)) * 100
    return round(score, 1)

#季節ボーナス
def seasonal_bonus(rgb, season):
    h, s, l = rgb_to_hsl(rgb)
    if season == "spring":
        if 30 <= h <= 120 and s >= 0.4 and l >= 0.5:
            return 10
    elif season == "summer":
        if 180 <= h <= 300 and s <= 0.6 and l >= 0.5:
            return 10
    elif season == "autumn":
        if 20 <= h <= 60 and 0.3 <= s <= 0.7 and 0.3 <= l <= 0.7:
            return 10
    elif season == "winter":
        if (h >= 240 or h <= 30) and s >= 0.6 and l <= 0.6:
            return 10
    return 0

    

def evaluate_color_pair(rgb1, rgb2, season=None):
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    delta = delta_e(lab1, lab2)
    delta_score = delta_e_fashion_score(delta, ideal=25, width=20)

    h1, s1, l1 = rgb_to_hsl(rgb1)
    h2, s2, l2 = rgb_to_hsl(rgb2)
    hue_diff = get_hue_difference(h1, h2)
    hue_score = get_hue_similarity_score(hue_diff)
    light_score = lightness_gradient_score(rgb1, rgb2)

    base_score = round(
        0.4 * delta_score + 0.4 * hue_score + 0.2 * light_score,
        1
    )

    bonus = 0
    if season:
        bonus += seasonal_bonus(rgb1, season)
        bonus += seasonal_bonus(rgb2, season)

    total_score = round(base_score + bonus, 1)

    return {
        "delta": delta,
        "delta_score": delta_score,
        "hue_diff": hue_diff,
        "hue_score": hue_score,
        "light_score": light_score,
        "base_score": base_score,
        "season_bonus": bonus,
        "total_score": total_score,
        "bonus":bonus
    }


def recommend_best_rgb(suggest_rgb):
    best_score = -float("inf")
    best_rgb = None

    for r in range(0, 256, 32):
        for g in range(0, 256, 32):
            for b in range(0, 256, 32):
                score_data = evaluate_color_pair(suggest_rgb, [r, g, b])
                score = score_data["total_score"]
                if score > best_score:
                    best_score = score
                    best_rgb = [r, g, b]
    return best_rgb

@app.get("/api/")
def read_root():
    return {"message": "Hello, this is Color Recommendation API"}

@app.post("/api/")
def get_bottom_with_delta(coordinate: Coordinate):
    top_rgb = coordinate.tops[:3]
    bottom_rgb = coordinate.bottoms[:3]
    season = coordinate.season

    recommend_top_rgb = recommend_best_rgb(top_rgb)
    recommend_bottom_rgb = recommend_best_rgb(bottom_rgb)
    score_data = evaluate_color_pair(np.array(top_rgb), np.array(bottom_rgb), season)



    if score_data["delta"] < 10:
        comment = CommentModel.SUBDUED
    elif score_data["delta"] < 25:
        comment = CommentModel.BALANCED
    elif score_data["delta"] < 50:
        comment = CommentModel.FANCY
    else:
        comment = CommentModel.UNBALANCED
    return {
        "recommend_top": recommend_top_rgb,
        "recommend_bottom": recommend_bottom_rgb,
        "ΔE": round(score_data["delta"], 2),
        "ΔEスコア": score_data["delta_score"],
        "色相差": round(score_data["hue_diff"], 1),
        "色相スコア": round(score_data["hue_score"], 1),
        "明度スコア": round(score_data["light_score"], 1),
        "最終スコア": score_data["base_score"],
        "season_bonus": score_data["season_bonus"],
        "total_score": score_data["total_score"],
        "comment": comment,
        "season_bonus":score_data["bonus"]
    }


   
