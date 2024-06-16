from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder

from dlgo.networks import small
from torch.nn import Sequential, Linear


go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
num_games = 100

encoder = OnePlaneEncoder((go_board_rows, go_board_cols))  # First we create an encoder of board size.

processor = GoDataProcessor(encoder=encoder.name())  # Then we initialize a Go Data processor with it.

generator = processor.load_go_data('train', num_games, use_generator=True)  # From the processor we create two data generators, for training and testing.
test_generator = processor.load_go_data('test', num_games, use_generator=True)

