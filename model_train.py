import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RULDataset


class BiLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden_dim * 2, output_dim)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # (batch, output_dim)


def train(model, train_loader, val_loader, optimizer,
          num_epochs=100, patience=15, save_path='best_model.pth'):
    """
    Training standar: MSE loss, next-step prediction.
    Early stopping berdasarkan val loss.
    """
    device    = next(model.parameters()).device
    criterion = nn.MSELoss()

    best_val_loss    = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)                  # (batch, 1)
            loss = criterion(pred, y)        # y shape: (batch, 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validasi ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y    = x.to(device), y.to(device)
                pred    = model(x)
                val_loss += criterion(pred, y).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Best val MSE: {best_val_loss:.6f} — model saved to '{save_path}'")
    return best_val_loss
