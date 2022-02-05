# Reference to the slide below:
# https://docs.google.com/presentation/d/1wn6UONkzJTTA3XxSgzS1hmkFONoVt2X8TO-VFU6Uo-E/edit#slide=id.g9f8093ad43_0_429
#
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def auto_grad_1(x, y, w1, w2, times):
    learning_rate = 1e-6
    for t in range(times):
        # Forward
        # y_pred = x.mm(w1).clamp(min=0).mm(w2)
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        loss = (y_pred - y).pow(2).sum()

        # Backward
        # loss.backward()
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # with torch.no_grad():
            # w1 -= learning_rate * w1.grad
            # w2 -= learning_rate * w2.grad
            # w1.grad.zero_()
            # w2.grad.zero_()
        w1 = w1 - learning_rate * grad_w1
        w2 = w2 - learning_rate * grad_w2

    print('\nThe loss: {:.4f}\n'.format(loss))


def auto_grad_2(x, y, w1, w2, times):
    learning_rate = 1e-6
    for t in range(times):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum()

        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()

    print('\nThe loss: {:.4f}\n'.format(loss))

def main():
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    w1 = torch.randn(D_in, H, requires_grad=True)
    w2 = torch.randn(H, D_out, requires_grad=True)

    # Auto grad
    auto_grad_1(x, y, w1, w2, 500)
    auto_grad_2(x, y, w1, w2, 500)

if __name__ == '__main__':
    main()