# model/src/utils.py
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, scaler_y, loader, device):
    """
    주어진 DataLoader의 배치들을 순회하며 모델을 평가하고,
    역스케일링된 예측값과 실제값으로 MAE, MSE를 계산해 반환합니다.

    Args:
        model       (torch.nn.Module): 평가할 Seq2Seq 모델
        scaler_y    (sklearn.preprocessing.MinMaxScaler): 타깃 역스케일러
        loader      (torch.utils.data.DataLoader): 평가용 데이터 로더
        device      (torch.device): 연산할 디바이스 (cpu / cuda)

    Returns:
        mae (float), mse (float)
    """
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for xb, yb, y0b in loader:
            xb, yb, y0b = xb.to(device), yb.to(device), y0b.to(device)
            # [B, out_len, 1]
            pred_scaled = model(xb, y0b).squeeze(-1).cpu().numpy()
            true_scaled = yb.squeeze(-1).cpu().numpy()

            # 역스케일링
            B = pred_scaled.shape[0]
            preds = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)) \
                                .reshape(B, -1)
            trues = scaler_y.inverse_transform(true_scaled.reshape(-1,1)) \
                                .reshape(B, -1)

            all_preds.append(preds.flatten())
            all_trues.append(trues.flatten())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)
    return mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred)


def forecast_next_hour(model, scaler_X, scaler_y, df, feature_cols,
                       in_len=1440, device=None):
    """
    최신 in_len 길이의 시퀀스를 사용해 다음 4스텝(out_len=4)에 대한
    종가를 예측한 뒤, 역스케일링하여 리스트로 반환합니다.

    Args:
        model        (torch.nn.Module): 학습된 Seq2Seq 모델
        scaler_X     (MinMaxScaler): 입력 피처 역스케일러
        scaler_y     (MinMaxScaler): 타깃 역스케일러
        df           (pd.DataFrame): 전체 캔들 + 지표 데이터프레임
        feature_cols (List[str]): 모델 입력에 사용할 컬럼 리스트
        in_len       (int): 입력 시퀀스 길이 (기본 96)
        device       (torch.device, optional): 연산 디바이스

    Returns:
        List[float]: 예측된 4스텝 종가 (실제 스케일)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # 가장 최근 in_len 개의 피처
    recent = df[feature_cols].values[-in_len:]
    X_scaled = scaler_X.transform(recent)

    # 마지막 실제 종가를 decoder 입력으로 변환
    last_close = df["Close"].values[-1]
    y0_scaled = scaler_y.transform([[last_close]])[0,0]

    # 텐서로 변환
    x_t = torch.tensor(X_scaled, dtype=torch.float32) \
               .unsqueeze(0).to(device)    # [1, in_len, feat_dim]
    y0_t = torch.tensor([y0_scaled], dtype=torch.float32) \
               .to(device)                # [1]

    with torch.no_grad():
        pred_scaled = model(x_t, y0_t)     # [1, out_len, 1]
        pred_scaled = pred_scaled.squeeze(0).squeeze(-1).cpu().numpy()

    # 역스케일링 후 반환
    return scaler_y.inverse_transform(pred_scaled.reshape(-1,1)) \
                   .flatten().tolist()
