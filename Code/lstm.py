import torch
import torch.nn.functional as F

def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None): 
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cy_tilde, outgate = gates.chunk(4, 1) #dim modified from 1 to 2

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cy_tilde = F.tanh(cy_tilde)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cy_tilde)
    hy = outgate * F.tanh(cy)

    return (hy, cy), {'in': ingate, 'forget': forgetgate, 'out': outgate, 'c_tilde': cy_tilde}


def apply_mask(hidden_l, mask):
    if type(hidden_l) == torch.autograd.Variable:
        return hidden_l * mask
    else:
        return tuple(h * mask for h in hidden_l)

def forward(self, input, hidden, mask=None):
    num_layers = self.num_layers
    weight = self.all_weights
    dropout = self.dropout
    # saves the gate values into the rnn object
    self.last_gates = []
    self.last_hidden =[]

    next_hidden = []

    hidden = list(zip(*hidden))

    for l in range(num_layers):
        hidden_l = hidden[l]
        if mask and l in mask:
            hidden_l = apply_mask(hidden_l, mask[l])
        # we assume there is just one token in the input
        hy, gates = LSTMCell(input[0], hidden_l, *weight[l])
        if mask and l in mask:
            hy = apply_mask(hy, mask[l])

        self.last_gates.append(gates)
        self.last_hidden.append(hy)
        next_hidden.append(hy)

        input = hy[0]

        if dropout != 0 and l < num_layers - 1:
            input = F.dropout(input, p=dropout, training=False, inplace=False)

    next_h, next_c = zip(*next_hidden)
    next_hidden = (
        torch.cat(next_h, 0).view(num_layers, *next_h[0].size()),
        torch.cat(next_c, 0).view(num_layers, *next_c[0].size())
    )


    # we restore the right dimensionality
    input = input.unsqueeze(0)

    return input, next_hidden

