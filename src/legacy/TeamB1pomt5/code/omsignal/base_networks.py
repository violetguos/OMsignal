import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def output_size(input_size, kernel_size, stride, padding):
    '''
    Helper function to determine output size of a convolution/pooling.
    '''
    output = int((input_size - kernel_size + 2*(padding)) / stride) + 1
    return output

class CNNRegression(nn.Module):
    '''
    To be used as a submodule in more complex models --
    just a regression using a CNN.
    '''
    def __init__(self, input_size, conv1_num_filters=2, conv2_num_filters=1, \
        conv_ksize=10, conv_stride=1, conv_padding=4, \
        pool_ksize=5, pool_stride=8, pool_padding=1, \
        num_linear=100,
        p=0.5):
        super(CNNRegression, self).__init__()

        # Set hyperparameters needed in forward
        self.conv1_num_filters, self.conv2_num_filters = conv1_num_filters, conv2_num_filters

        # Define layers
        #   Conv 1
        self.conv1 = nn.Conv1d(1, self.conv1_num_filters, \
                               kernel_size=conv_ksize, stride=conv_stride, padding=conv_padding)
        out_size = output_size(input_size, conv_ksize, conv_stride, conv_padding)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm1d(self.conv1_num_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_ksize, stride=pool_stride, padding=pool_padding)
        out_size = output_size(out_size, pool_ksize, pool_stride, pool_padding)

        #   Conv2
        self.conv2 = nn.Conv1d(self.conv1_num_filters, self.conv2_num_filters, \
                               kernel_size=conv_ksize, stride=conv_stride, padding=conv_padding)
        out_size = output_size(out_size, conv_ksize, conv_stride, conv_padding)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(self.conv2_num_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_ksize, stride=pool_stride, padding=pool_padding)
        #self.linear_input = output_size(out_size, pool_ksize, pool_stride, pool_padding)
        out_size = output_size(out_size, pool_ksize, pool_stride, pool_padding)

        # Conv3
        self.conv3 = nn.Conv1d(self.conv2_num_filters, 2, \
                               kernel_size=conv_ksize, stride=conv_stride, padding=conv_padding)
        out_size = output_size(out_size, conv_ksize, stride=conv_stride, padding=conv_padding)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm1d(2)
        self.pool3 = nn.MaxPool1d(kernel_size=pool_ksize, stride=pool_stride, padding=pool_padding)
        self.linear_input = output_size(out_size, pool_ksize, pool_stride, pool_padding)

        #   Linear output
        #self.fc1 = nn.Linear(self.linear_input*self.conv2_num_filters, num_linear)
        self.fc1 = nn.Linear(self.linear_input*2, num_linear)
        self.fc1_bn = nn.BatchNorm1d(num_linear)

        self.fc_regress = nn.Linear(num_linear, 1)
        self.drop_layer = nn.Dropout(p=p)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.conv2_bn(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool3(x)

        #x = x.view(-1, self.linear_input*self.conv2_num_filters)
        x = x.view(-1, self.linear_input*2)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = self.fc_regress(x)
        x = torch.squeeze(x,dim=1)
        return x

class CNNClassification(nn.Module):
    '''
    To be used as a submodule in more complex models --
    just a classification using a CNN.
    '''
    def __init__(self, input_size, num_classes, conv1_num_filters=2, conv2_num_filters=2, \
        conv_ksize=32, conv_stride=1, conv_padding=1, \
        pool_ksize=5, pool_stride=2, pool_padding=0, \
        num_linear=128,
        p=0.5):
        super(CNNClassification, self).__init__()

        # Set hyperparameters needed in forward
        self.conv1_num_filters, self.conv2_num_filters = conv1_num_filters, conv2_num_filters

        # Define layers
        #   Conv 1
        self.conv1 = nn.Conv1d(1, self.conv1_num_filters, \
                               kernel_size=conv_ksize, stride=conv_stride, padding=conv_padding)
        out_size = output_size(input_size, conv_ksize, conv_stride, conv_padding)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm1d(self.conv1_num_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_ksize, stride=pool_stride, padding=pool_padding)
        out_size = output_size(out_size, pool_ksize, pool_stride, pool_padding)

        #   Conv2
        self.conv2 = nn.Conv1d(self.conv1_num_filters, self.conv2_num_filters, \
                               kernel_size=conv_ksize, stride=conv_stride, padding=conv_padding)
        out_size = output_size(out_size, conv_ksize, conv_stride, conv_padding)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(self.conv2_num_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_ksize, stride=pool_stride, padding=pool_padding)
        self.linear_input = output_size(out_size, pool_ksize, pool_stride, pool_padding)

        #   Linear output
        self.fc1 = nn.Linear(self.linear_input*self.conv2_num_filters, num_linear)
        self.fc1_bn = nn.BatchNorm1d(num_linear)

        self.fc_classify = nn.Linear(num_linear, num_classes)
        self.drop_layer = nn.Dropout(p=p)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.conv2_bn(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, self.linear_input*self.conv2_num_filters)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = self.fc_classify(x)
        x = F.log_softmax(x, dim=0)
        return x

    def Predict_class(self, X, device):
        self.eval()
        copy_arr = np.copy(X)
        outputs = self.forward(torch.Tensor(np.fft.rfft(copy_arr, axis = 2).astype(np.float32)).to(device))
        _ , predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        return predicted


"""
old Network modules, we don't use these classes but might be useful to next teams

class Encoder_RNN(nn.Module):
    '''
    CNN base class
    '''
    def __init__(self, out_channels =10, out_len = 50):
        super(Encoder_RNN, self).__init__()
        #prototype to reduce the sequence length from 3750 to 50, this encoder isn't useful in its current state

        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = out_channels, dilation =1, padding = 2, stride =1 , kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=int(3750/out_len), stride=int(3750/out_len), padding=0)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.pool1(x)

class RNN(nn.Module):
    '''
    RNN base class (less complicated implementation than LSTM)
    '''
    def __init__(self, input_size=10, hidden_size=50, num_classes =32 ):
        super(RNN, self).__init__()

        #structure of the basic network
        self.hidden_size = hidden_size
        self.to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.to_output = nn.Linear(input_size + hidden_size, num_classes)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.to_hidden(combined)
        output = self.to_output(combined)
        return output, hidden

    def init_hidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size)
class MLP(nn.Module):
    '''
    Vanilla MLP for testing
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = torch.squeeze(x,dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MLPRegression(nn.Module):
    '''
    To be used as a submodule in more complex models --
    just a regression using an MLP.
    '''
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 20)
        self.bn_fc2 = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, 20)
        self.bn_fc3 = nn.BatchNorm1d(20)
        self.fc_regress = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.squeeze(x,dim=1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc_regress(x)
        return x

class MLPClassification(nn.Module):
    '''
    To be used as a submodule in more complex models --
    just a classification using an MLP.
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassification, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_classify = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.squeeze(x,dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc_classify(x)
        x = F.log_softmax(x, dim=0)
        return x

class CNN(nn.Module):
    '''
    CNN base class
    '''
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        # These are filler hyperparameters for now - numbers should
        # flow, but are arbitrary
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(4, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(937*128, 175)
        self.fc2 = nn.Linear(175, num_classes)

    def forward(self, x):
        #print('Input:', x.size())
        x = F.relu(self.conv1(x))
        #print('After relu conv1:', x.size())
        x = self.pool1(x)
        #print('After pool conv1:', x.size())
        x = F.relu(self.conv2(x))
        #print('After relu conv2:', x.size())
        x = self.pool2(x)
        #print('After pool conv2:', x.size())
        x = x.view(-1, 937*128)
        #print('After view:', x.size())
        x = F.relu(self.fc1(x))
        #print('After relu fc1:', x.size())
        x = self.fc2(x)
        #print('After fc2:', x.size())
        return x
"""