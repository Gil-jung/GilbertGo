from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.networks.large import Large

import torch
import torch.nn as nn
from torch.optim import Adadelta

import os

def main():
    parent_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    current_path = os.path.dirname(__file__)

    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 100000

    def compute_acc(argmax, y):
        count = 0
        for i in range(len(argmax)):
            if argmax[i] == y[i]:
                count += 1
        return count / len(argmax)
    
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHES = 100

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))  # First we create an encoder of board size.

    processor = GoDataProcessor(encoder=encoder.name())  # Then we initialize a Go Data processor with it.

    generator = processor.load_go_data('train', num_games, use_generator=True)  # From the processor we create two data generators, for training and testing.
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    # input_shape = (encoder.num_planes, go_board_rows, go_board_cols)

    model = Large(go_board_rows, encoder.num_planes).cuda()
    model.apply(initialize_weights                                                                                                         )
    print(model)

    optimizer = Adadelta(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    total_steps = generator.get_num_samples() // 1000
    print(generator.num_samples)

    for epoch in range(NUM_EPOCHES):
        model.train()
        tot_loss = 0.0
        steps = 0

        for x, y in generator.generate(BATCH_SIZE, num_classes):
            steps += 1
            optimizer.zero_grad()
            x = x.cuda()
            y_ = model(x)
            loss = loss_fn(y_, y.cuda()) 
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()

            if steps >= total_steps:
                break

        print('='*100)
        print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss / steps))
        _, argmax = torch.max(y_, dim=1)
        train_acc = compute_acc(argmax, y)
        print("Epoch {}, Acc(train) : {}".format(epoch+1, train_acc))

        model.eval()
        x, y = next(iter(test_generator.generate(BATCH_SIZE, num_classes)))
        x = x.cuda()
        y_ = model(x)
        loss = loss_fn(y_, y.cuda()) 
        print("Epoch {}, Loss(val) : {}".format(epoch+1, loss.item()))
        _, argmax = torch.max(y_, dim=1)
        test_acc = compute_acc(argmax, y)
        print("Epoch {}, Acc(val) : {}".format(epoch+1, test_acc))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'train_loss': tot_loss / steps,
            'val_loss': loss.item(),
        }, current_path + f"\\checkpoints\\large_model_epoch_{epoch+1}.pt")

if __name__ == '__main__':
    main()