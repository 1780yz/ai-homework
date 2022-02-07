from __future__ import print_function
import torch

def forward(x_input,
            weight_input, bias_input,
            weight_hidden, bias_hidden):
    # The input layer
    x = x_input
    w = weight_input
    b = bias_input
    y = torch.sigmoid(x.mm(w) + b)

    # The hidden layer
    x = y
    w = weight_hidden
    b = bias_hidden
    y = torch.sigmoid(x.mm(w) + b)

    return y

def main():
    # The inputs
    x_input = torch.tensor([[.05, .10]])

    weight_input = torch.tensor([[.15, .25], [.20, .30]]).t()
    weight_input.requires_grad = True
    bias_input = torch.tensor(0.35, requires_grad=True)

    weight_hidden = torch.tensor([[.40, .50], [.45, .55]]).t()
    weight_hidden.requires_grad = True
    bias_hidden = torch.tensor(.60, requires_grad=True)

    # The expected outputs
    y_target = torch.tensor([[.01, .99]])

    # The learning parameters
    learning_rate = 20.0
    porches = 3000
    display_interval = 500
    print('learning_rate: ', learning_rate)

    for t in range(porches + 1):
        y_pred = forward(x_input,
                    weight_input, bias_input,
                    weight_hidden, bias_hidden)
        loss = (y_pred - y_target).pow(2).sum()
        if t % display_interval == 0:
            print('t:', t, ', loss(mse): ', loss)

        loss.backward()
        with torch.no_grad():
            weight_hidden -= learning_rate * weight_hidden.grad
            bias_hidden -= learning_rate * bias_hidden.grad
            weight_input -= learning_rate * weight_input.grad
            bias_input -= learning_rate * bias_input.grad

            weight_hidden.grad.zero_()
            bias_hidden.grad.zero_()
            weight_input.grad.zero_()
            bias_input.grad.zero_()

    print('Finish')
    y_pred = forward(x_input,
            weight_input, bias_input,
            weight_hidden, bias_hidden)
    print('y_pred: ', y_pred)

if __name__ == '__main__':
    main()