import torch.nn as nn

class CNN_RNN_Model(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super(CNN_RNN_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels=3 for RGB
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
        )

        img_width = 256
        self.rnn_input_size = (img_width // 8) * 128
        self.lstm = nn.LSTM(self.rnn_input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.cnn(x)

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, height, -1)

        x, _ = self.lstm(x)

        x = self.fc(x)

        return x
