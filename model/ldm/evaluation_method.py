def calculate_rmse(y_true, y_pred, mask):
    mask = (mask == 0).unsqueeze(-1).unsqueeze(-1)
    mse = torch.mean(((y_true - y_pred) * mask) ** 2)
    return torch.sqrt(mse)

def calculate_mape(y_true, y_pred, mask, epsilon=1e-8):
    mask = (mask == 0).unsqueeze(-1).unsqueeze(-1)
    y_true, y_pred = y_true * mask, y_pred * mask
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
