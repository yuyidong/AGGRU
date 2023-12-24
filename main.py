import torch
import argparse
import baselines
from torch import nn
from model import AGRNN
from engine import Engine
from dataset import DataOperator

# torch.manual_seed(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/test.xlsx', help='data path')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--timestep', type=int, default=24)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=3)
    args = parser.parse_args()

    data = DataOperator(args.path, args.timestep, args.batch_size, args.num_nodes)
    # model = AGRNN(args.timestep, args.num_nodes, args.hidden_size, args.dropout, args.num_blocks).to(args.device)
    # model = AGRNN(args.timestep, args.num_nodes, args.hidden_size, args.dropout, args.num_blocks, data.adj).to(args.device)
    model = baselines.GRU(args.num_nodes, args.hidden_size).to(args.device)
    # model = baselines.GCN(args.timestep, args.hidden_size, data.adj).to(args.device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    engine = Engine(data, model, loss_func, optimizer, scheduler, args.num_epochs)

    engine.train()
    engine.test()


if __name__ == '__main__':
    main()
