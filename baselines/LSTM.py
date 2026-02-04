import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.n_vars=args.n_vars
        self.hidden_size = args.hidden_dim
        self.num_layers = args.num_layers
        self.lstm = nn.LSTM(self.n_vars,self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, args.num_class)
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取序列中最后一个时间步的输出
        return out