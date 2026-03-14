# ==============================================================================
# MEM QUANTITATIVE ANALYSIS ENGINE
# Multiplicative Error Model — Volatility & Volume Forecasting
# ==============================================================================
# Tác giả : Quantitative Developer
# Cấu trúc: Modular Programming (Lập trình hàm)
# Mục đích : Phân tích Kỹ thuật & Định lượng dựa trên MEM(1,1)
#   1. Data Pipeline        — Tải dữ liệu & tính Log Return
#   2. Realized Volatility   — Đo lường biến động thực tế
#   3. MEM Core Engine       — Tối ưu hóa bộ hệ số (ω, α, β)
#   4. Volume Forecast       — Dự báo khối lượng giao dịch
#   5. Liquidity Assessment  — Đánh giá chất lượng thanh khoản
#   6. Report Generator      — Xuất báo cáo tổng hợp
# ==============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# MODULE 1: DATA PIPELINE — Nhập dữ liệu động & Tiền xử lý
# ==============================================================================

def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Tải dữ liệu thị trường từ Yahoo Finance và tính toán các cột phái sinh.
    
    Parameters
    ----------
    ticker     : str — Mã cổ phiếu hoặc chỉ số (VD: 'AAPL', '^GSPC', 'NVDA')
    start_date : str — Ngày bắt đầu (format: 'YYYY-MM-DD')
    end_date   : str — Ngày kết thúc (format: 'YYYY-MM-DD')
    
    Returns
    -------
    pd.DataFrame chứa các cột: Close, Volume, Log_Return
    
    Raises
    ------
    ValueError nếu không tải được dữ liệu hoặc dữ liệu rỗng.
    """
    try:
        print(f"  📡 Đang tải dữ liệu [{ticker}] từ {start_date} đến {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # --- Kiểm tra dữ liệu trả về ---
        if df is None or df.empty:
            raise ValueError(
                f"Không tải được dữ liệu cho '{ticker}'. "
                "Kiểm tra lại mã ticker hoặc khoảng thời gian."
            )
        
        # --- Xử lý MultiIndex (yfinance đôi khi trả về MultiIndex columns) ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # --- Tính Lợi suất Logarit (Log Return) ---
        # Công thức: r_t = ln(P_t / P_{t-1})
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Loại bỏ hàng NaN đầu tiên (do shift)
        df = df.dropna()
        
        print(f"  ✅ Tải thành công: {len(df)} phiên giao dịch.")
        return df
        
    except Exception as e:
        raise ValueError(f"❌ Lỗi Data Pipeline: {e}")


# ==============================================================================
# MODULE 2: REALIZED VOLATILITY — Biến động thực tế
# ==============================================================================

def compute_realized_volatility(log_returns: np.ndarray, annualize: bool = True) -> dict:
    """
    Tính Realized Volatility dựa trên chuỗi Log Return bình phương.
    
    Phương pháp:
        RV = sqrt( sum(r_t^2) )                    (tổng phiên)
        RV_annualized = RV_daily * sqrt(252)        (quy năm)
    
    Parameters
    ----------
    log_returns : np.ndarray — Chuỗi lợi suất logarit
    annualize   : bool       — Có quy đổi ra năm hay không (mặc định: True)
    
    Returns
    -------
    dict chứa: rv_daily, rv_annualized, variance_series
    """
    # Chuỗi phương sai tức thời: x_t = r_t^2
    variance_series = log_returns ** 2
    
    # Realized Volatility trung bình ngày
    rv_daily = np.sqrt(np.mean(variance_series))
    
    # Quy năm (giả định 252 phiên giao dịch/năm)
    rv_annualized = rv_daily * np.sqrt(252) if annualize else rv_daily
    
    return {
        "rv_daily": rv_daily,
        "rv_annualized": rv_annualized,
        "variance_series": variance_series  # Dùng làm input cho MEM Volatility
    }


# ==============================================================================
# MODULE 3: MEM CORE ENGINE — Lõi Toán học Multiplicative Error Model
# ==============================================================================
# Công thức nền tảng MEM(1,1):
#   μ_t = ω + α · x_{t-1} + β · μ_{t-1}
#
# Trong đó:
#   ω (omega) : Hằng số nền (base level)
#   α (alpha) : Trọng số phản ứng với cú sốc quá khứ (shock reaction)
#   β (beta)  : Trọng số quán tính / bộ nhớ (persistence / memory)
#   x_{t-1}   : Giá trị quan sát thực tế tại thời điểm t-1
#   μ_{t-1}   : Giá trị kỳ vọng (dự báo) tại thời điểm t-1
# ==============================================================================

def mem_filter(params: tuple, data: np.ndarray) -> np.ndarray:
    """
    Bộ lọc MEM(1,1): Tính toán chuỗi kỳ vọng có điều kiện μ_t.
    
    Đây là HÀM LÕI (core function) được tái sử dụng cho cả
    Volatility Forecasting và Volume Forecasting.
    
    Parameters
    ----------
    params : tuple(ω, α, β) — Bộ 3 hệ số MEM
    data   : np.ndarray      — Chuỗi dữ liệu đầu vào (dương, non-negative)
    
    Returns
    -------
    np.ndarray — Chuỗi μ_t (conditional mean)
    """
    omega, alpha, beta = params
    n = len(data)
    mu = np.zeros(n)
    
    # Khởi tạo: μ_0 = giá trị trung bình mẫu (unconditional mean)
    mu[0] = np.mean(data)
    
    # Đệ quy tiến: μ_t = ω + α·x_{t-1} + β·μ_{t-1}
    for t in range(1, n):
        mu[t] = omega + alpha * data[t - 1] + beta * mu[t - 1]
        
        # Đảm bảo μ_t luôn dương (tránh log(0) trong NLL)
        if mu[t] <= 0:
            mu[t] = 1e-8
    
    return mu


def mem_negative_log_likelihood(params: tuple, data: np.ndarray) -> float:
    """
    Hàm Loss: Negative Log-Likelihood dưới giả định phân phối Exponential.
    
    Giả định: ε_t = x_t / μ_t ~ Exponential(1)
    ⟹ NLL = Σ [ ln(μ_t) + x_t / μ_t ]
    
    Parameters
    ----------
    params : tuple(ω, α, β) — Bộ 3 hệ số cần tối ưu
    data   : np.ndarray      — Chuỗi quan sát thực tế (dương)
    
    Returns
    -------
    float — Giá trị Negative Log-Likelihood (càng nhỏ càng tốt)
    """
    omega, alpha, beta = params
    
    # ── Ràng buộc Toán học (Stationarity Constraints) ──
    # 1. ω > 0          : Đảm bảo mức nền dương
    # 2. α ≥ 0, β ≥ 0   : Hệ số không âm
    # 3. α + β < 1      : Điều kiện dừng (stationarity) — tránh phát tán vô hạn
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
        return np.inf  # Phạt vô cực nếu vi phạm ràng buộc
    
    # Tính chuỗi kỳ vọng μ_t
    mu = mem_filter(params, data)
    
    # Negative Log-Likelihood (Exponential distribution)
    nll = np.sum(np.log(mu) + data / mu)
    
    return nll


def fit_mem_model(data: np.ndarray, label: str = "MEM") -> dict:
    """
    Tối ưu hóa bộ hệ số MEM bằng scipy.optimize.minimize (L-BFGS-B).
    
    Parameters
    ----------
    data  : np.ndarray — Chuỗi dữ liệu đầu vào (dương)
    label : str        — Nhãn mô hình (để phân biệt Volatility vs Volume)
    
    Returns
    -------
    dict chứa: omega, alpha, beta, nll_score, mu_fitted, success
    """
    # --- Giá trị khởi tạo (initial guess) ---
    # α + β = 0.9 < 1 → thỏa điều kiện dừng
    x0 = [0.01, 0.10, 0.80]
    
    # --- Biên của từng tham số ---
    bounds = (
        (1e-8, None),    # ω > 0
        (1e-8, 0.999),   # 0 < α < 1
        (1e-8, 0.999),   # 0 < β < 1
    )
    
    # --- Chạy bộ tối ưu hóa L-BFGS-B ---
    result = minimize(
        fun=mem_negative_log_likelihood,
        x0=x0,
        args=(data,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12}
    )
    
    if not result.success:
        print(f"  ⚠️  [{label}] Cảnh báo: Optimizer không hội tụ hoàn toàn.")
    
    omega_opt, alpha_opt, beta_opt = result.x
    mu_fitted = mem_filter(result.x, data)
    
    return {
        "label": label,
        "omega": omega_opt,
        "alpha": alpha_opt,
        "beta": beta_opt,
        "persistence": alpha_opt + beta_opt,   # Mức độ quán tính tổng
        "nll_score": result.fun,
        "mu_fitted": mu_fitted,
        "success": result.success,
    }


def mem_forecast_next(mem_result: dict, last_observation: float) -> float:
    """
    Dự báo giá trị tiếp theo (t+1) dựa trên bộ hệ số MEM đã fit.
    
    Công thức: μ_{t+1} = ω + α · x_t + β · μ_t
    
    Parameters
    ----------
    mem_result       : dict  — Kết quả từ fit_mem_model()
    last_observation : float — Giá trị quan sát cuối cùng (x_t)
    
    Returns
    -------
    float — Giá trị dự báo μ_{t+1}
    """
    omega = mem_result["omega"]
    alpha = mem_result["alpha"]
    beta  = mem_result["beta"]
    mu_last = mem_result["mu_fitted"][-1]  # μ_t (giá trị fitted cuối cùng)
    
    forecast = omega + alpha * last_observation + beta * mu_last
    return forecast


# ==============================================================================
# MODULE 4: VOLUME FORECAST — Dự báo Khối lượng Giao dịch
# ==============================================================================

def forecast_volume(df: pd.DataFrame) -> dict:
    """
    Tái sử dụng lõi MEM để dự báo Khối lượng Giao dịch ngày tiếp theo.
    
    Thay vì đầu vào là Volatility (r_t^2), ta đưa Volume vào MEM:
        μ_t^{vol} = ω + α · Volume_{t-1} + β · μ_{t-1}^{vol}
    
    Parameters
    ----------
    df : pd.DataFrame — DataFrame chứa cột 'Volume'
    
    Returns
    -------
    dict chứa: mem_result (bộ hệ số), forecast_tomorrow (dự báo ngày mai),
               last_volume, avg_volume
    """
    volume_data = df['Volume'].values.astype(float)
    
    # Loại bỏ các phiên có Volume = 0 (nếu có)
    volume_data = volume_data[volume_data > 0]
    
    if len(volume_data) < 10:
        raise ValueError("Không đủ dữ liệu Volume để fit MEM (cần ≥ 10 phiên).")
    
    # --- Fit MEM trên chuỗi Volume ---
    print("  📊 Đang fit MEM trên chuỗi Volume...")
    mem_vol = fit_mem_model(volume_data, label="MEM-Volume")
    
    # --- Dự báo Volume ngày mai ---
    last_vol = volume_data[-1]
    forecast_tomorrow = mem_forecast_next(mem_vol, last_vol)
    
    return {
        "mem_result": mem_vol,
        "forecast_tomorrow": forecast_tomorrow,
        "last_volume": last_vol,
        "avg_volume": np.mean(volume_data),
    }


# ==============================================================================
# MODULE 5: LIQUIDITY ASSESSMENT — Đánh giá Thanh khoản
# ==============================================================================

def assess_liquidity(rv_annualized: float, forecast_vol: float, avg_volume: float) -> dict:
    """
    Kết hợp Realized Volatility và Forecasted Volume để đánh giá thanh khoản.
    
    Logic phân loại:
    ┌──────────────────────────────┬──────────────────────────────────────────┐
    │ Điều kiện                    │ Kết luận                                 │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Volume ↑ cao  + Vol ↓ thấp  │ 🟢 XUẤT SẮC — An toàn đi lệnh lớn      │
    │ Volume ↑ cao  + Vol ↑ cao   │ 🟡 TRUNG BÌNH — Cẩn thận với slippage   │
    │ Volume ↓ thấp + Vol ↓ thấp  │ 🟠 YẾU — Thị trường trầm lắng          │
    │ Volume ↓ thấp + Vol ↑ cao   │ 🔴 NGUY HIỂM — Rủi ro trượt giá cao    │
    └──────────────────────────────┴──────────────────────────────────────────┘
    
    Parameters
    ----------
    rv_annualized : float — Realized Volatility (annualized)
    forecast_vol  : float — Volume dự báo ngày mai
    avg_volume    : float — Volume trung bình lịch sử
    
    Returns
    -------
    dict chứa: grade, description, volume_signal, volatility_signal
    """
    # --- Phân loại Volume ---
    volume_ratio = forecast_vol / avg_volume
    volume_high = volume_ratio >= 1.0  # Volume dự báo ≥ trung bình → Cao
    
    # --- Phân loại Volatility ---
    # Ngưỡng: RV annualized > 30% → Cao (tương đương biến động mạnh)
    volatility_high = rv_annualized > 0.30
    
    volume_signal = "CAO" if volume_high else "THẤP"
    volatility_signal = "CAO" if volatility_high else "THẤP"
    
    # --- Ma trận quyết định ---
    if volume_high and not volatility_high:
        grade = "🟢 XUẤT SẮC"
        description = (
            "Thanh khoản dồi dào, biến động ổn định. "
            "An toàn để thực hiện lệnh lớn với slippage thấp."
        )
    elif volume_high and volatility_high:
        grade = "🟡 TRUNG BÌNH"
        description = (
            "Thanh khoản có nhưng biến động mạnh. "
            "Có thể giao dịch nhưng cần cẩn thận với trượt giá (slippage)."
        )
    elif not volume_high and not volatility_high:
        grade = "🟠 YẾU"
        description = (
            "Thị trường trầm lắng, thanh khoản mỏng. "
            "Nên chia nhỏ lệnh hoặc chờ phiên sôi động hơn."
        )
    else:  # not volume_high and volatility_high
        grade = "🔴 NGUY HIỂM"
        description = (
            "Thanh khoản kém + biến động mạnh = Rủi ro trượt giá RẤT CAO. "
            "KHÔNG NÊN đi lệnh lớn. Đợi thanh khoản cải thiện."
        )
    
    return {
        "grade": grade,
        "description": description,
        "volume_signal": volume_signal,
        "volatility_signal": volatility_signal,
        "volume_ratio": volume_ratio,
    }


# ==============================================================================
# MODULE 6: REPORT GENERATOR — Xuất Báo cáo Tổng hợp
# ==============================================================================

def print_report(ticker: str, period: str,
                 rv_result: dict, 
                 mem_vol_result: dict, mem_volatility_result: dict,
                 vol_forecast_info: dict, 
                 liquidity: dict):
    """
    In báo cáo phân tích tổng hợp ra màn hình.
    """
    line = "═" * 68
    thin = "─" * 68
    
    print(f"\n{line}")
    print(f"  📋  BÁO CÁO PHÂN TÍCH ĐỊNH LƯỢNG — MEM ENGINE")
    print(f"{line}")
    print(f"  Ticker  : {ticker}")
    print(f"  Giai đoạn: {period}")
    print(f"{thin}")
    
    # ── PHẦN 1: Realized Volatility ──
    print(f"\n  📐 REALIZED VOLATILITY")
    print(f"  {thin}")
    print(f"  • RV Daily (trung bình)     : {rv_result['rv_daily']:.6f}")
    print(f"  • RV Annualized (quy năm)   : {rv_result['rv_annualized']:.4f}  "
          f"({rv_result['rv_annualized']*100:.2f}%)")
    
    # ── PHẦN 2: MEM — Volatility ──
    print(f"\n  ⚙️  MEM(1,1) — VOLATILITY MODEL")
    print(f"  {thin}")
    print(f"  • Omega  (ω) : {mem_volatility_result['omega']:.8f}")
    print(f"  • Alpha  (α) : {mem_volatility_result['alpha']:.6f}   "
          f"← Phản ứng cú sốc")
    print(f"  • Beta   (β) : {mem_volatility_result['beta']:.6f}   "
          f"← Quán tính / Bộ nhớ")
    print(f"  • α + β      : {mem_volatility_result['persistence']:.6f}   "
          f"← Persistence")
    print(f"  • NLL Score  : {mem_volatility_result['nll_score']:.4f}")
    print(f"  • Converged  : {'✅ Có' if mem_volatility_result['success'] else '❌ Không'}")
    
    # ── PHẦN 3: MEM — Volume ──
    mem_v = vol_forecast_info['mem_result']
    print(f"\n  📊 MEM(1,1) — VOLUME FORECAST MODEL")
    print(f"  {thin}")
    print(f"  • Omega  (ω) : {mem_v['omega']:.4f}")
    print(f"  • Alpha  (α) : {mem_v['alpha']:.6f}")
    print(f"  • Beta   (β) : {mem_v['beta']:.6f}")
    print(f"  • α + β      : {mem_v['persistence']:.6f}")
    print(f"  • NLL Score  : {mem_v['nll_score']:.2f}")
    print(f"  • Converged  : {'✅ Có' if mem_v['success'] else '❌ Không'}")
    
    # ── PHẦN 4: Dự báo Volume ──
    print(f"\n  🔮 DỰ BÁO VOLUME NGÀY MAI")
    print(f"  {thin}")
    print(f"  • Volume phiên cuối    : {vol_forecast_info['last_volume']:>15,.0f}")
    print(f"  • Volume trung bình    : {vol_forecast_info['avg_volume']:>15,.0f}")
    print(f"  • Volume dự báo (t+1)  : {vol_forecast_info['forecast_tomorrow']:>15,.0f}")
    print(f"  • Tỷ lệ vs Trung bình : {liquidity['volume_ratio']:.2f}x")
    
    # ── PHẦN 5: Đánh giá Thanh khoản ──
    print(f"\n  💧 ĐÁNH GIÁ THANH KHOẢN")
    print(f"  {thin}")
    print(f"  • Tín hiệu Volume     : {liquidity['volume_signal']}")
    print(f"  • Tín hiệu Volatility : {liquidity['volatility_signal']}")
    print(f"  • Xếp hạng            : {liquidity['grade']}")
    print(f"  • Nhận định            : {liquidity['description']}")
    
    print(f"\n{line}")
    print(f"  🏁  KẾT THÚC BÁO CÁO")
    print(f"{line}\n")


# ==============================================================================
# MAIN EXECUTION — Điều phối toàn bộ quy trình
# ==============================================================================

def main():
    """
    Hàm điều phối chính:
    Thu thập input → Chạy pipeline → Xuất báo cáo.
    """
    print("=" * 68)
    print("  MEM QUANTITATIVE ANALYSIS ENGINE v1.0")
    print("  Multiplicative Error Model — Volatility & Volume Forecasting")
    print("=" * 68)
    
    # ── Bước 0: Thu thập tham số từ người dùng ──
    try:
        ticker     = input("\n  📌 Nhập mã Ticker (VD: AAPL, NVDA, ^GSPC): ").strip().upper()
        start_date = input("  📅 Ngày bắt đầu (YYYY-MM-DD): ").strip()
        end_date   = input("  📅 Ngày kết thúc (YYYY-MM-DD): ").strip()
        
        if not ticker or not start_date or not end_date:
            raise ValueError("Tất cả 3 tham số (ticker, start_date, end_date) đều bắt buộc.")
            
    except KeyboardInterrupt:
        print("\n\n  ⛔ Người dùng hủy chương trình.")
        return
    except ValueError as ve:
        print(f"\n  ❌ Lỗi đầu vào: {ve}")
        return
    
    period_str = f"{start_date} → {end_date}"
    print(f"\n{'─' * 68}")
    print(f"  🚀 Bắt đầu phân tích [{ticker}] | {period_str}")
    print(f"{'─' * 68}\n")
    
    # ══════════════════════════════════════════════════════════════════════
    # Bước 1: DATA PIPELINE
    # ══════════════════════════════════════════════════════════════════════
    try:
        df = fetch_market_data(ticker, start_date, end_date)
    except ValueError as e:
        print(f"\n  {e}")
        return
    
    log_returns = df['Log_Return'].values
    
    # ══════════════════════════════════════════════════════════════════════
    # Bước 2: REALIZED VOLATILITY
    # ══════════════════════════════════════════════════════════════════════
    print("\n  📐 Đang tính Realized Volatility...")
    rv_result = compute_realized_volatility(log_returns)
    print(f"  ✅ RV Annualized = {rv_result['rv_annualized']*100:.2f}%")
    
    # ══════════════════════════════════════════════════════════════════════
    # Bước 3: MEM — VOLATILITY (dùng chuỗi r_t^2 đã scale)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  ⚙️  Đang fit MEM(1,1) trên Volatility Proxy (r²×10000)...")
    
    # Scale x10000 để tránh underflow số quá nhỏ
    volatility_data = rv_result['variance_series'] * 10000
    mem_volatility_result = fit_mem_model(volatility_data, label="MEM-Volatility")
    print(f"  ✅ Hội tụ: {'Có' if mem_volatility_result['success'] else 'Không'} | "
          f"Persistence (α+β) = {mem_volatility_result['persistence']:.4f}")
    
    # ══════════════════════════════════════════════════════════════════════
    # Bước 4: MEM — VOLUME FORECAST
    # ══════════════════════════════════════════════════════════════════════
    print("\n  📊 Đang chạy module Volume Forecast...")
    try:
        vol_forecast_info = forecast_volume(df)
        print(f"  ✅ Dự báo Volume ngày mai: "
              f"{vol_forecast_info['forecast_tomorrow']:,.0f} shares")
    except ValueError as e:
        print(f"  ❌ {e}")
        return
    
    # ══════════════════════════════════════════════════════════════════════
    # Bước 5: LIQUIDITY ASSESSMENT
    # ══════════════════════════════════════════════════════════════════════
    print("\n  💧 Đang đánh giá Thanh khoản...")
    liquidity = assess_liquidity(
        rv_annualized=rv_result['rv_annualized'],
        forecast_vol=vol_forecast_info['forecast_tomorrow'],
        avg_volume=vol_forecast_info['avg_volume'],
    )
    print(f"  ✅ Xếp hạng: {liquidity['grade']}")
    
    # ══════════════════════════════════════════════════════════════════════
    # Bước 6: XUẤT BÁO CÁO
    # ══════════════════════════════════════════════════════════════════════
    print_report(
        ticker=ticker,
        period=period_str,
        rv_result=rv_result,
        mem_vol_result=vol_forecast_info['mem_result'],
        mem_volatility_result=mem_volatility_result,
        vol_forecast_info=vol_forecast_info,
        liquidity=liquidity,
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
