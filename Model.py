import torch.nn as nn

class Traffic_model(nn.Module):
    def __init__(self, args):
        super(Traffic_model, self).__init__()
        self.input_base_dim = args.input_base_dim

        from Framework import ESTNet
        self.predictor = ESTNet(args, args.device, self.input_base_dim)

    def forward(self, source, label, select_dataset):
        x_predic = self.predictor(source, label, select_dataset)
        return x_predic
