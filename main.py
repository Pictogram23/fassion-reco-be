from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Coordinate(BaseModel):
    tops: list[int]
    bottoms: list[int]


@app.get("/")
def read_root(coordinate: Coordinate):

    r_top = coordinate.tops[0]
    g_top = coordinate.tops[1]
    b_top = coordinate.tops[2]

    #RGB値をリニアRGB値に変換する式
    if r_top/255 <= 0.04045:
        r_top_linear = (r_top/255)/12.92
    else:
        r_top_linear = ((r_top/255+0.055)/1.055)**2.4

    if g_top/255 <= 0.04045:
        g_top_linear = (r_top/255)/12.92
    else:
        g_top_linear = ((r_top/255+0.055)/1.055)**2.4
    
    if b_top/255 <= 0.04045:
        b_top_linear = (r_top/255)/12.92
    else:
        b_top_linear = ((r_top/255+0.055)/1.055)**2.4
    
    #リニアRGB値をXYZ値に変換する式
    X_t = 0.4124*r_top_linear + 0.3576*g_top_linear + 0.1805*b_top_linear
    Y_t = 0.2126*r_top_linear + 0.7152*g_top_linear + 0.0722*b_top_linear
    Z_t = 0.0193*r_top_linear + 0.1192*g_top_linear + 0.9505*b_top_linear

    #XYZ値をxyz値に変換する式
    x = X_t/(X_t+Y_t+Z_t)
    y = Y_t/(X_t+Y_t+Z_t)
    z = Z_t/(X_t+Y_t+Z_t)

    #xyz値をLab値に変換する式
    Xn = 0.9505
    Yn = 1.0000
    Zn = 1.0890

    delta = 6/29
    delta3 = delta**3
    inv_3delta2 = 1/(3*delta**2)

    def f(t):
        if t > delta3:
            return t**(1/3)
        else:
            return t*inv_3delta2+4/29
    
    fx = f(X_t/Xn)
    fy = f(Y_t/Yn)
    fz = f(Z_t/Zn)

    L_top = 116 * fy - 16
    a_top = 500 * (fx-fy)
    b_top = 200 * (fy-fz)


    #--------bottomsについて----------
    r_bottom = coordinate.bottoms[0]
    g_bottom = coordinate.bottoms[1]
    b_bottom = coordinate.bottoms[2]

    #RGB値をリニアRGB値に変換する式
    if r_bottom/255 <= 0.04045:
        r_bottom_linear = (r_bottom/255)/12.92
    else:
        r_bottom_linear = ((r_bottom/255+0.055)/1.055)**2.4

    if g_bottom/255 <= 0.04045:
        g_bottom_linear = (r_bottom/255)/12.92
    else:
        g_bottom_linear = ((r_bottom/255+0.055)/1.055)**2.4
    
    if b_bottom/255 <= 0.04045:
        b_bottom_linear = (r_bottom/255)/12.92
    else:
        b_bottom_linear = ((r_bottom/255+0.055)/1.055)**2.4
    
    #リニアRGB値をXYZ値に変換する式
    X_b = 0.4124*r_bottom_linear + 0.3576*g_bottom_linear + 0.1805*b_bottom_linear
    Y_b = 0.2126*r_bottom_linear + 0.7152*g_bottom_linear + 0.0722*b_bottom_linear
    Z_b = 0.0193*r_bottom_linear + 0.1192*g_bottom_linear + 0.9505*b_bottom_linear

    #XYZ値をxyz値に変換する式
    x_b = X_b/(X_b+Y_b+Z_b)
    y_b = Y_b/(X_b+Y_b+Z_b)
    z_b = Z_b/(X_b+Y_b+Z_b)

    #xyz値をLab値に変換する式
   
    
    fx = f(X_b/Xn)
    fy = f(Y_b/Yn)
    fz = f(Z_b/Zn)

    L_bottom = 116 * fy - 16
    a_bottom = 500 * (fx-fy)
    b_bottom = 200 * (fy-fz)

    #色差の計算
    delta_L = L_top - L_bottom
    delta_a = a_top - a_bottom
    delta_b = b_top - b_bottom

    delta_Eab = ((delta_L)**2 + (delta_a)**2 + (delta_b)**2)**(1/2) 

    #判定
    if delta_Eab < 10:
        print("result=60点（同系色・控え目)")
    elif delta_Eab < 25:
        print("result=100点（調和の取れたコーデ)")
    else :
        print("result=50点（ポップな印象)")

   
