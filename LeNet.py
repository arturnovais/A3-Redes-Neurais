import torch
from torch import nn
class LeNet(nn.Module):
    '''
    Primeira Camada Convolucional:
        - 6 filtros 1x5
        - Stride de 1
        - Padding de 2
        - Função de ativação: Tanh
        - Saída: 6 mapas de características de tamanho 40 (mantendo o tamanho da entrada)

    Primeira Camada de Pooling (Subsampling):
        - Pooling 1x2 (Average pooling)
        - Stride de 2
        - Saída: 6 mapas de características de tamanho 20

    Segunda Camada Convolucional:
        - 16 filtros 1x5
        - Stride de 1
        - Sem padding
        - Função de ativação: Tanh
        - Saída: 16 mapas de características de tamanho 16

    Segunda Camada de Pooling (Subsampling):
        - Pooling 1x2 (Average pooling)
        - Stride de 2
        - Saída: 16 mapas de características de tamanho 8

    Primeira Camada Totalmente Conectada:
        - 120 neurônios
        - Função de ativação: Tanh

    Segunda Camada Totalmente Conectada:
        - 84 neurônios
        - Função de ativação: Tanh

    Terceira Camada Totalmente Conectada (Saída):
        - 10 neurônios (um para cada classe)
        - Função de ativação: Softmax (implícita na loss)
    '''

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(x.size(0), 1, 40)  # arruma para (bath_size, 1, 40)
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Achata o tensor para as lineares
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x) #não coloquei a softmax pois está implícita na CrossEntropyLoss, no paper original foi usado MSE como loss e por isso era requerido o uso da softmax
        return x
