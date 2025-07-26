

"""
都市の気温をシミュレーションするプログラム


定めた条件
-シミュレーションを行う7日間の最高気温と最低気温はすべて同じとする。
- 夏：最高気温 T_x = 31℃、最低気温 T_n = 21℃ を用いる
- 冬：最高気温 T_x = 10℃、最低気温 T_n = 2℃ を用いる
- 入力データは24時間分のため、1日分を繰り返し使用する。
・地盤熱伝導率 λ= 1.1 W/m^2K
・地盤比熱 C_GAMMA = 1600000 J/m3K

-地中にある下端の境界条件
地下の気温情報は10cm間隔で与え、1 mが下端とする
下端の気温は26℃で固定し、ほかの地中気温は、下端の気温からの温度差を考慮する。
地下の気温は列が時間、行が深度のデータフレームで定義する。
地表の温度T_sはindex=0 のデータとなる。
地中の温度変化は差分法により求める。


- 温度 T における飽和水蒸気圧
Tetensの式を用いる
水面の場合: a = 7.5 , b = 237.3とする


- 地表面の顕熱フラックス条件
z_M = 0.1  # 運動量に関する粗度 (m)
z_H = 0.2 * z_M  # 顕熱に関する粗度 (m)
・零面変位 d=0
・人工排熱量  0
・日射吸収率を0.8、長波放射率は0.9
・地表面の表層対流熱伝達率 a_s= 18 W/(m2K)
コンダクタンスで表記すると、表層熱コンダクタンス g_Has =a_s/c_p = 0.614 mol/(m2s)

- 潜熱フラックス条件
・蒸発潜熱 L = 2.5 * 10^6 J/kg 
・地表面の湿気伝達率 a_w = 8.5 * 10^-8 kg/(m2sPa)
・地表面近傍大気の水蒸気圧 E_a = 2750 Pa (夏) , E_a = 660 Pa (冬)


入力データ
0.5時間刻みで24時間分のデータ
本シミュレーションでは、7日間のシミュレーションを行うので、すべての日で同じ値を用いる。
最初の値は0.5時間の値であることに注意すること。

入力データのフォーマット
時間経過(h),全天日射量(W/m2),大気放射量(W/m2)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# 入力データのパス 相対パス
SUMMER_DATA_PATH = r'data\夏の全天日射量と大気放射量.csv'
WINTER_DATA_PATH = r'data\冬の全天日射量と大気放射量.csv'
###
## Parameters
###
# シナリオ
SEASON = "冬"  # "夏" or "冬"
BETA = 0 # 蒸発効率　（0~1の範囲で指定）
Q_H = 0  # 人工排熱量 (W/m2)
U_100 = 5  # z=100 m における風速 (m/s)
INPUT_DATA_PATH = SUMMER_DATA_PATH if SEASON == "夏" else WINTER_DATA_PATH

#物理定数
K = 0.4  # カルマン定数
SIGMA = 5.67e-8  # ステファン・ボルツマン定数 (W/(m2*K^4))
G = 9.81  # 重力加速度 (m/s2)
R = 8.314  # 気体定数 (J/(mol*K))
#顕熱フラックス条件
Z_M =  0.1 # 運動量に関する粗度 (m)
Z_H = 0.2 * Z_M  # 顕熱に関する粗度 (m)
D = 0  # 零面変位 (m)
EPSILON = 0.9  # 長波放射率
SOLAR_ABSORPTION = 0.8  # 日射吸収率
G_HAS = 0.614  # 表層熱コンダクタンス (mol/(m2s))
#潜熱フラックス条件
LATENT_HEAT = 2.5e6  # 蒸発潜熱 (J/kg) 
ALPHA_W = 8.5e-8 #地表面の湿気伝達率 (kg/(m2sPa))
E_A = 2750 if SEASON == "夏" else 660  # 地表面近傍大気の水蒸気圧 (Pa)
#その他
HEIGHT=100  # 高さ (m) (大気温度を求める高さ)
# 夏とwinterでの最高気温と最低気温
T_X = 31 if SEASON== "夏" else 21  # 最高気温　(℃)
T_N = 21 if SEASON== "夏" else 2  # 最低気温 (℃)


### 時間設定 ###
DELTA_T = 0.5 # 時刻の刻み幅 (h)　ここは0.5で固定（読み込みデータが0.5時間刻みのため）
SIMULATION_DAYS = 7  # シミュレーション日数
T_START = 0  # シミュレーション開始時刻 (h)
T_END = SIMULATION_DAYS * 24  # シミュレーション終了時刻 (h)
# シミュレーションの時間刻み
TIME_STEPS = np.arange(T_START, T_END + DELTA_T, DELTA_T)
# 時間刻みの数
NUM_TIME_STEPS = len(TIME_STEPS)
### 時間設定 ###

#初期値
U_STER_INITIAL = 0.3 # 摩擦速度の初期値 (m/s)
H_0 = 0.0  # 顕熱フラックスの初期値 (W/m2)

##
# 自身で設定した条件
##
# 地中の情報
DELTA_X = 0.1 # 地下の空間刻み(m)
UNDERGROUND_MAX_DEPTH = 1.0  # 地下の最大深さ (m)
T_UNDERGROUND_INITIAL = 26.0 if SEASON == "夏" else 10.0  # 地下の初期温度 (℃)
ALPHA_S = 18  # 表層対流熱伝達率 (W/(m2*K))
C_P = ALPHA_S / G_HAS  # 空気の比熱 (J/(mol*K))
P= 101325  # 大気圧 (Pa)
LAMBDA = 1.1  # 地盤熱伝導率 (W/(m^2*K))
C_GAMMA = 1600000  # 地盤比熱 (J/(m^3*K))
ALPHA = LAMBDA*3600*DELTA_T /( C_GAMMA * DELTA_X**2)  # 地中の熱拡散率 (m^2/h)
C_1 = C_GAMMA * DELTA_X / 2  # 地表の熱容量 (J/(m2*K))



#地表面近傍大気の水上気圧

###
## end parameters
###

###
## Functions
###
# 大気温度を計算する関数 
def calculate_air_temperature(t):
    """
    t: 時刻 (h)
    T_X: 夏の最高気温 (℃)
    T_N: 夏の最低気温 (℃)
    """
    omega = 2 * np.pi / 24  # 1時間あたりの角速度
    gamma_t = 0.44 - 0.46 * np.sin(omega * t + 0.9) + 0.11 * np.sin(2 * omega * t + 0.9)
    T_a = T_X * gamma_t + T_N * (1 - gamma_t)
    return T_a

# 大気安定度と非断熱補正係数を求める関数
def calculate_stability_and_correction(heat_flux, T_a, u_star):
    """
    heat_flux: 顕熱フラックス (W/m2)
    T_a: 大気温度 (℃)
    u_star: 摩擦速度 (m/s)
    zeta: 大気安定度
    psi_M: 非断熱補正係数
    psi_H: 非断熱補正係数
    """
    rho_hat = P / (R * (T_a + 273.15))  # 大気の密度 (mol/m3)
    zeta = -K * G * HEIGHT * heat_flux / (rho_hat * C_P * (T_a + 273.15) * u_star**3) 
    if zeta > 0: # 安定な場合
        psi_M = psi_H = 6 * np.log(1 + zeta)
    elif zeta == 0: # 中立な場合
        psi_M = 0
        psi_H = 0
    else: # 不安定な場合
        psi_H = -2 * np.log((1 + (1 - 16 * zeta)**0.5) / 2)
        psi_M = 0.6* psi_H
    return zeta,psi_M, psi_H

# 摩擦速度を計算する関数
def calculate_friction_velocity(psi_M):
    """
    psi_M: 非断熱補正係数 (摩擦速度に関する)
    u_star: 摩擦速度 (m/s)
    """
    u_star = K * U_100 / (np.log((HEIGHT - D) / Z_M) + psi_M)
    return u_star

# 境界層コンダクタンスを計算する関数
def calculate_boundary_layer_conductance(T_a, psi_M, psi_H):
    """
    T_a: 大気温度 (℃)
    psi_M: 非断熱補正係数 (摩擦速度に関する)
    psi_H: 非断熱補正係数 (顕熱に関する)
    """
    rho_hat = P / (R * (T_a + 273.15))  # 大気の密度 (mol/m3)
    g_Ha = K**2 * rho_hat * U_100 / ((np.log((HEIGHT - D) / Z_M) + psi_M) * (np.log((HEIGHT - D) / Z_H) + psi_H))
    return g_Ha

# 潜熱フラックス　lEを計算する関数
def calculate_latent_heat_flux(T_s):
    """
    T_s: 地表面温度 (℃)
    e_s: 飽和水蒸気圧 (hPa)
    lE: 潜熱フラックス (W/m2)
    e_sは Tetensの式を用いて計算する
    LE算出時にはe_sをPaに変換する必要があり、100を掛ける。
    """
    e_s = 6.11 * 10**(7.5 * T_s / (T_s + 237.3))  # 飽和水蒸気圧 (hPa) 
    lE = LATENT_HEAT* BETA * ALPHA_W * (e_s*100 - E_A)  # 潜熱フラックス (W/m2)
    return lE

# 地表面の熱収支から地表面温度 T_s を求める関数
def calculate_surface_temperature(S_r, L_a, T_s_prev,T_a,T_underground_prev,lE_prev):
    """
    S_r: 全天日射量 (W/m2)
    L_a: 大気放射量 (W/m2)
    T_s_prev: 前の時刻の地表面温度 (℃)
    T_a: 前の時刻の大気温度 (℃)
    T_underground_prev: 前の時刻の第1層目（地表面が0）の地下温度 (℃)
    lE_prev: 前の時刻の潜熱フラックス (W/m2)
    """
    R_abs = SOLAR_ABSORPTION * S_r + EPSILON * L_a
    R_n = R_abs - EPSILON * SIGMA * (T_s_prev + 273.15)**4
    T_s_new = T_s_prev + (DELTA_T * 3600 / C_1) * (
        ALPHA_S * (T_a - T_s_prev) 
        - LAMBDA * (T_s_prev - T_underground_prev) 
        + R_n 
        - lE_prev
        + Q_H)
    return T_s_new

# 顕熱フラックスを計算する関数
def calculate_fluxes(T_s, T_a, alpha_u):
    """ 
    T_s: 地表面温度 (℃)
    T_a: 大気温度 (℃)
    alpha_u: 都市境界層対流熱伝達率 (W/(m2*K))
    """
    R_s = 1 / (ALPHA_S)  # 表面熱伝達抵抗
    R_u = 1 / (alpha_u)   # 都市境界層熱抵抗
    heat_flux = (T_s - T_a) / (R_s + R_u)  # 顕熱フラックス
    return heat_flux

# 都市気温 T_uを計算する関数
def calculate_urban_temperature(T_s, T_a, alpha_u):
    """
    T_s: 地表面温度 (℃)
    T_a: 大気温度 (℃)
    alpha_u: 都市境界層対流熱伝達率 (W/(m2*K))
    """
    T_u = (alpha_u* T_a + ALPHA_S * T_s) / (alpha_u + ALPHA_S)
    return T_u

###
## end functions
###

input_data = pd.read_csv(INPUT_DATA_PATH)

T_underground = pd.DataFrame(
    T_UNDERGROUND_INITIAL,
    index=np.arange(T_START, T_END + DELTA_T, DELTA_T),
    columns=np.arange(0, UNDERGROUND_MAX_DEPTH + DELTA_X, DELTA_X)
)

# 時刻0時の大気温度を計算
T_a_0 = calculate_air_temperature(0)  #
# 時刻0時における大気安定度を求める(H_0=0, U_STER_INITIAL=0.3,T_a_0を用いる)
zeta_0,psi_M_0, psi_H_0 = calculate_stability_and_correction(H_0,T_a_0, U_STER_INITIAL)
# 時刻0時における g_Ha を求める
g_Ha_0 = calculate_boundary_layer_conductance(T_a_0, psi_M_0, psi_H_0)
alpha_u_0  = g_Ha_0 * C_P
# 時刻0時における摩擦速度を求める
u_star_0 = calculate_friction_velocity(psi_M_0)
# 時刻0時の顕熱フラックスを計算
heat_flux_0 = calculate_fluxes(T_underground.loc[0,0], T_a_0, alpha_u_0)
# 地表面温度T_s_0を初期値として取得
T_s_0 = T_underground.iloc[0, 0]
# 都市気温T_u_0を計算
T_u_0 = calculate_urban_temperature(T_s_0, T_a_0, alpha_u_0)


# 時刻ごとに計算を行う
T_u = np.zeros(NUM_TIME_STEPS)
heat_flux = np.zeros(NUM_TIME_STEPS)
T_a = np.zeros(NUM_TIME_STEPS)
u_star = np.zeros(NUM_TIME_STEPS)
zeta = np.zeros(NUM_TIME_STEPS)
psi_M = np.zeros(NUM_TIME_STEPS)
psi_H = np.zeros(NUM_TIME_STEPS)
g_Ha = np.zeros(NUM_TIME_STEPS)
alpha_u = np.zeros(NUM_TIME_STEPS)
S_r = np.zeros(NUM_TIME_STEPS)
L_a = np.zeros(NUM_TIME_STEPS)
lE = np.zeros(NUM_TIME_STEPS)

# 全天日射量と大気放射量は24時間分しかないので、1日分を繰り返す
for i in range(NUM_TIME_STEPS):
    if i <= 24/DELTA_T-1:  # 最初の24時間分はそのまま
        S_r[i+1] = input_data["全天日射量(W/m2)"].iloc[i]
        L_a[i+1] = input_data["大気放射量(W/m2)"].iloc[i]
    elif i % (24/DELTA_T) == 0 and i > 0:  # 24時間目は初期値を設定
        S_r[i] = input_data["全天日射量(W/m2)"].iloc[47]
        L_a[i] = input_data["大気放射量(W/m2)"].iloc[47]
    else:  # 24時間以降は繰り返す
        S_r[i] = S_r[i % int(24/DELTA_T)]
        L_a[i] = L_a[i % int(24/DELTA_T)]

T_u[0] = T_u_0
heat_flux[0] = heat_flux_0
T_a[0] = T_a_0
u_star[0] = u_star_0
zeta[0] = zeta_0
psi_M[0] = psi_M_0
psi_H[0] = psi_H_0
g_Ha[0] = g_Ha_0
alpha_u[0] = alpha_u_0


for t in range(1, NUM_TIME_STEPS):
    T_a[t] = calculate_air_temperature(TIME_STEPS[t])
    zeta[t], psi_M[t], psi_H[t] = calculate_stability_and_correction(heat_flux[t-1], T_a[t], u_star[t-1])
    
    u_star[t] = calculate_friction_velocity(psi_M[t])
    g_Ha[t] = calculate_boundary_layer_conductance(T_a[t], psi_M[t], psi_H[t])
    alpha_u[t] = C_P * g_Ha[t] if g_Ha[t] is not None else 0.0
    T_s_prev = T_underground.loc[TIME_STEPS[t-1], 0.0]  # 前の時刻の地表面温度を取得
    T_s_new = calculate_surface_temperature(S_r[t-1], L_a[t-1], T_s_prev, T_a[t-1], T_underground.iloc[t-1, 1], lE[t-1])
    lE[t] = calculate_latent_heat_flux(T_s_new)  # 潜熱フラックスを計算
    T_underground.iloc[t, 0] = T_s_new 
    heat_flux[t] = calculate_fluxes(T_s_new, T_a[t], alpha_u[t])
    T_u[t] = calculate_urban_temperature(T_s_new, T_a[t], alpha_u[t])

    # 地下1層目以降を差分法で更新（地表面は差分法では求めない）
    for i in range(1, T_underground.shape[1]):
        if i+1 < T_underground.shape[1]:
            # i==1 のときは t時刻の地表面温度を使う
            left = T_underground.loc[TIME_STEPS[t], 0.0] if i == 1 else T_underground.iloc[t-1, i-1]
            right = T_underground.iloc[t-1, i+1]
            center = T_underground.iloc[t-1, i]
            T_underground.iloc[t, i] = center + ALPHA * (left + right - 2 * center)
        else:
            # 下端は温度固定
            T_underground.iloc[t, i] = T_UNDERGROUND_INITIAL

    #データフレームにまとめる
df = pd.DataFrame({
    'time': TIME_STEPS,
    'S_r': S_r,
    'L_a': L_a,
    'T_a': T_a,
    'T_u': T_u,
    'T_s': T_underground.iloc[:, 0].values,  # 地
    'T_underground': T_underground.values.tolist(),  # 地下の温度をリスト形式で保存
    'u_star': u_star,
    'heat_flux': heat_flux,
    'zeta': zeta,
    'psi_M': psi_M,
    'psi_H': psi_H,
    'g_Ha': g_Ha,
    'alpha_u': alpha_u,
    'lE': lE
})
# CSVファイルに保存
df.to_csv('urban_temperature_simulation.csv', index=False)

# データの分析
"""
都市気温について、最終24時間の最高気温、最低気温、平均気温を計算
"""
T_u_last_24 = T_u[-48:]  # 最後の24時間の都市温度
max_T_u = np.max(T_u_last_24)
min_T_u = np.min(T_u_last_24)
mean_T_u = np.mean(T_u_last_24)
print(f"計算条件： 季節：{SEASON}  , 蒸発効率：{BETA} , 人工排熱量：{Q_H} W/m2")
print(f"最後の24時間の都市温度: 最高 {max_T_u:.2f} ℃, 最低 {min_T_u:.2f} ℃, 平均 {mean_T_u:.2f} ℃")

#都市気温を時間ごとにプロット
plt.figure(figsize=(12, 6))
plt.plot(TIME_STEPS, T_u, label='都市気温 T_u (℃)', color='red')
plt.plot(TIME_STEPS, T_a, label='上空気温 T_a (℃)', color='blue')
plt.plot(TIME_STEPS, T_underground.iloc[:, 0], label='地表面温度 T_s (℃)', color='green')
plt.title(f'都市気温のシミュレーション (季節: {SEASON}, 蒸発効率: {BETA}, 人工排熱量: {Q_H} W/m2)')
plt.xlabel('時間 (h)')
plt.ylabel('温度 (℃)')
plt.legend()
plt.grid()
plt.xticks(np.arange(T_START, T_END + 1, 24))  # x軸の目盛りを1日ごとに設定
plt.xlim(T_START, T_END)
plt.tight_layout()
plt.savefig('urban_temperature_simulation.png')
plt.show()

# 最後の24時間の都市気温をプロット
plt.figure(figsize=(12, 6))
plt.plot(TIME_STEPS[-48:], T_u[-48:], label='都市気温 T_u (℃)', color='red')
plt.plot(TIME_STEPS[-48:], T_a[-48:], label='上空気温 T_a (℃)', color='blue')
plt.plot(TIME_STEPS[-48:], T_underground.iloc[-48:, 0], label='地表面温度 T_s (℃)', color='green')
plt.title(f'最後の24時間の都市気温のシミュレーション (季節: {SEASON}, 蒸発効率: {BETA}, 人工排熱量: {Q_H} W/m2)')
plt.xlabel('時間 (h)')
plt.ylabel('温度 (℃)')
plt.legend()
plt.grid()
plt.xticks(np.arange(T_END - 24, T_END + 1, 1))  # x軸の目盛りを1時間ごとに設定
plt.xlim(T_END - 24, T_END)
plt.tight_layout()
plt.savefig('urban_temperature_simulation_last_24_hours.png')
plt.show()
