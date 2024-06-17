from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks.small import SMALL

import torch
import torch.nn as nn
from torch.optim import SGD

import os

def main():
    parent_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    current_path = os.path.dirname(__file__)

    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 100

    def compute_acc(argmax, y):
        count = 0
        for i in range(len(argmax)):
            if argmax[i] == y[i]:
                count += 1
        return count / len(argmax)

    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHES = 5

    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))  # First we create an encoder of board size.

    processor = GoDataProcessor(encoder=encoder.name())  # Then we initialize a Go Data processor with it.

    generator = processor.load_go_data('train', num_games, use_generator=True)  # From the processor we create two data generators, for training and testing.
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    # input_shape = (encoder.num_planes, go_board_rows, go_board_cols)

    model = SMALL(go_board_rows, encoder.num_planes).cuda()
    print(model)

    optimizer = SGD(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    model.train()


    for epoch in range(NUM_EPOCHES):
        tot_loss = 0.0

        for x, y in generator.generate(BATCH_SIZE, num_classes):
            optimizer.zero_grad()
            x = x.cuda()
            y_ = model(x)
            loss = loss_fn(y_, y.cuda()) 
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()

        print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss))
        if epoch % 2 == 1:
            x, y = next(iter(test_generator))
            x = x.cuda()
            y_ = model(x)
            _, argmax = torch.max(y_, dim=-1)
            test_acc = compute_acc(argmax, y)

            print("Acc(val) : {}".format(test_acc))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, current_path + "\\checkpoints\\small_model_epoch_{epoch}.pt")

if __name__ == '__main__':
    main()