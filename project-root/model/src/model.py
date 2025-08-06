# model.py
# import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# ============================================
# 1) Model Definitions: BahdanauAttention, Encoder, Decoder, Seq2Seq
# ============================================
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, dropout_rate=0.0):
        super().__init__()
        self.W_a = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.U_a = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.v_a = nn.Linear(dec_hid_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, dec_hidden, enc_outputs):
        # enc_outputs: [B, T, enc_hid_dim]
        T = enc_outputs.size(1)
        dec_exp = dec_hidden.unsqueeze(1).repeat(1, T, 1)  # [B, T, dec_hid_dim]
        energy = torch.tanh(self.W_a(enc_outputs) + self.U_a(dec_exp))
        scores = self.v_a(energy).squeeze(-1)               # [B, T]
        weights = torch.softmax(scores, dim=1)             # [B, T]
        weights = self.dropout(weights.unsqueeze(-1)).squeeze(-1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # [B, enc_hid_dim]
        return context, weights

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hid_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers>1 else 0.0
        )
    def forward(self, x):
        # x: [B, T, input_dim]
        return self.lstm(x)  # outputs, (h, c)

class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, out_len, n_layers=1, dropout=0.0):
        super().__init__()
        self.out_len = out_len
        self.attn    = BahdanauAttention(hid_dim, hid_dim, dropout_rate=dropout)
        self.lstm    = nn.LSTM(
            input_dim + hid_dim, hid_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers>1 else 0.0
        )
        self.fc      = nn.Linear(hid_dim, 1)

    def forward(self, enc_outputs, states, y0, y_target=None, teacher_forcing_ratio=0.5):
        """
        enc_outputs: [B, T, hid_dim]
        states: (h, c)
        y0: [B]                 ← 마지막 실제값
        y_target: [B, out_len]  ← 전체 실제 시퀀스 (teacher forcing)
        """
        h, c = states
        B = enc_outputs.size(0)
        device = enc_outputs.device

        inp = y0.unsqueeze(1).unsqueeze(2)  # [B,1,1]
        outputs = torch.zeros(B, self.out_len, 1, device=device)

        for t in range(self.out_len):
            # 1) Attention
            context, _ = self.attn(h[-1], enc_outputs)  # [B, hid_dim]
            # 2) LSTM step
            lstm_in = torch.cat([inp, context.unsqueeze(1)], dim=2)  # [B,1,1+hid_dim]
            out, (h, c) = self.lstm(lstm_in, (h, c))                # out: [B,1,hid_dim]
            pred = self.fc(out.squeeze(1)).unsqueeze(1)             # [B,1,1]
            outputs[:, t:t+1, :] = pred

            # 3) Teacher forcing?
            if (y_target is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                # use ground-truth
                inp = y_target[:, t].unsqueeze(1).unsqueeze(2)
            else:
                # use own prediction
                inp = pred

        return outputs  # [B, out_len, 1]

class Seq2Seq(nn.Module):
    def __init__(self, enc_input_dim, dec_input_dim, hid_dim,
                 out_len, n_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(enc_input_dim, hid_dim, n_layers, dropout)
        self.decoder = Decoder(dec_input_dim, hid_dim, out_len, n_layers, dropout)

    def forward(self, x, y0, y_target=None, teacher_forcing_ratio=0.5):
        enc_out, (h, c) = self.encoder(x)
        return self.decoder(
            enc_out, (h, c),
            y0,
            y_target=y_target,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

# ============================================
# 2) Training Function: train_model with Teacher Forcing
# ============================================
def train_model(
    X_train, Y_train,
    X_val,   Y_val,
    X_test,  Y_test,
    best_params: dict,
    device: torch.device
):
    """
    Trains Seq2Seq with Bahdanau Attention + teacher forcing.
    Returns: model, train_losses, val_losses, test_loss, test_preds (np.ndarray)
    """
    # Unpack hyperparameters
    lstm_layers  = int(best_params['lstm_layers'])
    units        = int(best_params['units'])
    lr           = float(best_params['learning_rate'])
    tfr          = float(best_params.get('teacher_forcing_ratio', 0.5))
    dropout_rate = float(best_params.get('lstm_dropout_rate', 0.0))
    batch_size   = int(best_params['batch_size'])
    out_len      = Y_train.shape[-1]

    # Instantiate model
    model = Seq2Seq(
        enc_input_dim=X_train.shape[-1],
        dec_input_dim=1,
        hid_dim=units,
        out_len=out_len,
        n_layers=lstm_layers,
        dropout=dropout_rate
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Wrap data
    Xtr = torch.from_numpy(X_train).float().to(device)
    Ytr = torch.from_numpy(Y_train).float().to(device)  # [N_tr, out_len]
    Xva = torch.from_numpy(X_val).float().to(device)
    Yva = torch.from_numpy(Y_val).float().to(device)

    tr_loader = DataLoader(
        TensorDataset(Xtr, Ytr, Ytr[:,0]), batch_size=batch_size, shuffle=True
    )
    va_loader = DataLoader(
        TensorDataset(Xva, Yva, Yva[:,0]), batch_size=batch_size
    )

    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(1, 151):
        # --- Train ---
        model.train()
        total_tr = 0.0
        for xb, yb, y0b in tr_loader:
            optimizer.zero_grad()
            # yb: [B, out_len,1] → y_target: [B, out_len]
            y_target = yb.squeeze(-1)
            preds = model(
                xb, y0b,
                y_target=y_target,
                teacher_forcing_ratio=tfr
            ).squeeze(-1)  # [B, out_len]
            loss = criterion(preds, y_target)
            loss.backward()
            optimizer.step()
            total_tr += loss.item()
        train_losses.append(total_tr / len(tr_loader))

        # --- Validate ---
        model.eval()
        total_va = 0.0
        with torch.no_grad():
            for xb, yb, y0b in va_loader:
                y_target = yb.squeeze(-1)
                preds = model(xb, y0b,
                              y_target=y_target,
                              teacher_forcing_ratio=0.0) \
                        .squeeze(-1)
                total_va += criterion(preds, y_target).item()
        val_loss = total_va / len(va_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # logging
        if epoch == 1 or epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"[{epoch}/150] Train={train_losses[-1]:.4f}, Val={val_loss:.4f}, LR={lr_now:.2e}")

        # simple early stop
        if val_loss > best_val and epoch > 20:
            break

    # --- Test ---
    Xte = torch.from_numpy(X_test).float().to(device)
    Yte = torch.from_numpy(Y_test).float().to(device)
    model.eval()
    with torch.no_grad():
        preds_te = model(Xte, Yte[:,0],
                         y_target=None,
                         teacher_forcing_ratio=0.0) \
                   .squeeze(-1)  # [N_te, out_len]
        test_loss = criterion(preds_te, Yte.squeeze(-1)).item()

    test_preds = preds_te.cpu().numpy()
    print(f"Test Loss: {test_loss:.4f}")

    return model, train_losses, val_losses, test_loss, test_preds


# ============================================
# 3) Device Definition
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
