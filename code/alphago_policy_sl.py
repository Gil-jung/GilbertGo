from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.networks.alphago import AlphaGoPolicyResNet

import torch
import torch.nn as nn
from torch.optim import SGD

import os

def main():
    current_path = os.path.dirname(__file__)

    rows, cols = 19, 19
    num_classes = rows * cols
    num_games = 1000

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
    pre_trained = True
    re_train_epoch = 32
    version = 1

    encoder = AlphaGoEncoder(use_player_plane=False)
    processor = GoDataProcessor(encoder=encoder.name())
    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    alphago_sl_policy = AlphaGoPolicyResNet().cuda()
    if not pre_trained:
        alphago_sl_policy.apply(initialize_weights)
        print("initializing...")
    else:
        pt_flie = torch.load(current_path + f"\\checkpoints\\alphago_sl_policy_epoch_{re_train_epoch}_v{version}.pt")
        alphago_sl_policy.load_state_dict(pt_flie['model_state_dict'])
        print("model loading...")

    print(alphago_sl_policy)

    optimizer = SGD(alphago_sl_policy.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    total_steps = generator.get_num_samples() // BATCH_SIZE
    print(generator.num_samples)

    for epoch in range(NUM_EPOCHES):
        alphago_sl_policy.train()
        tot_loss = 0.0
        steps = 0

        for x, y in generator.generate(BATCH_SIZE, num_classes):
            steps += 1
            optimizer.zero_grad()
            x = x.cuda()
            y_ = alphago_sl_policy(x)
            loss = loss_fn(y_, y.cuda()) 
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()

            if steps >= total_steps:
                break

        print('='*50)
        print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss / steps))
        _, argmax = torch.max(y_, dim=1)
        train_acc = compute_acc(argmax, y)
        print("Epoch {}, Acc(train) : {}".format(epoch+1, train_acc))

        alphago_sl_policy.eval()
        x, y = next(iter(test_generator.generate(BATCH_SIZE, num_classes)))
        x = x.cuda()
        y_ = alphago_sl_policy(x)
        loss = loss_fn(y_, y.cuda()) 
        print("Epoch {}, Loss(val) : {}".format(epoch+1, loss.item()))
        _, argmax = torch.max(y_, dim=1)
        test_acc = compute_acc(argmax, y)
        print("Epoch {}, Acc(val) : {}".format(epoch+1, test_acc))

        torch.save({
            'model_state_dict': alphago_sl_policy.state_dict(),
            'loss': loss,
        }, current_path + f"\\checkpoints\\alphago_sl_policy_epoch_{epoch+1}_v{version+1}.pt")

if __name__ == '__main__':
    main()