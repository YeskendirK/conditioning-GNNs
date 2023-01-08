import torch


class ConditionalLinear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 cond_features,
                 method='weak'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cond_features = cond_features
        self.method = method

        # Using torch nn Linear and BiLinear
        # The weights that parametrize the conditional linear layer
        if method == 'weak':
            self.linear_weak = torch.nn.Linear(self.in_features + self.cond_features, self.out_features)
        elif method == 'strong':
            self.linear = torch.nn.Linear(self.in_features, self.out_features)
            self.linear_embedding = torch.nn.Linear(self.cond_features, self.out_features, bias=False)
        elif method == 'pure':
            self.bilinear = torch.nn.Bilinear(self.in_features, self.cond_features, self.out_features)
        else:
            raise ValueError('Unknown method, should be \'weak\', \'strong\', or \'pure\'.')

    def forward(self, f_in, cond_vec):
        if self.method == 'weak':
            f_out = self.linear_weak(torch.cat((f_in, cond_vec), dim=-1))
        elif self.method == 'strong':
            f_out = self.linear(f_in) * self.linear_embedding(cond_vec)
        elif self.method == 'pure':
            f_out = self.bilinear(f_in, cond_vec)
        return f_out
