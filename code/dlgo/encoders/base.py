import importlib


class Encoder:
    def name(self):  # Lets us support logging or saving the name of the encoder our model is using.
        raise NotImplementedError
    
    def encode(self, game_state):  # Turn a Go board into a numeric data.
        raise NotImplementedError
    
    def encode_point(self, point):  # Turn a Go board point into an integer index.
        raise NotImplementedError
    
    def decode_point_index(self, index):  # Turn an integer index back into a Go board point.
        raise NotImplementedError
    
    def num_points(self):  # Number of points on the board, i.e. board width times board height.
        raise NotImplementedError
    
    def shape(self):  # Shape of the encoded board structure.
        raise NotImplementedError


def get_encoder_by_name(name, board_size):  # We can create encoder instances by referencing their name.
    if isinstance(board_size, int):
        board_size = (board_size, board_size)  # If board_size is one integer, we create a square board from it.
    module = importlib.import_module('dlgo.encoders.' + name)
    constructor = getattr(module, 'create')  # Each encoder implementation will have to provide a "create" function that provides an instance.
    return constructor(board_size)