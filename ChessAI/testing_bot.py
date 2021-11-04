import chess
import chess.polyglot
import random
import numpy as np
import torch
import math

def convert_fen_to_encoding(fen_string):
    one_hot_dict = {'P': '100000000000', 'N': '010000000000', 'B': '001000000000', 'R': '000100000000', 'Q': '000010000000', 'K': '000001000000', 'p': '000000100000', 'n': '000000010000', 'b': '000000001000', 'r': '000000000100', 'q': '000000000010', 'k': '000000000001', '.': '000000000000'}
    fen_string_props = fen_string.split(' ')
    rows = chess.Board(fen_string).__str__().split('\n')
    squares_encoding = []
    for row in rows:
        squares_encoding.append(list(map(lambda x: [int(char) for char in one_hot_dict[x]], row.split(' '))))

    if fen_string_props[1] == 'w':
        turn = [1, 0]
    elif fen_string_props[1] == 'b':
        turn = [0, 1]
    else:
        turn = [0, 0]

    castle_privileges = fen_string_props[2]
    castle_privileges_encoding = [int(x in castle_privileges) for x in ['K', 'Q', 'k', 'q']]

    row_encoding = []
    for row in squares_encoding:
        for square in row:
            row_encoding = row_encoding + square

    full_encoding = row_encoding + turn + castle_privileges_encoding
    
    return full_encoding

def filter_mates(eval):
    if '#' in str(eval):
        if '+' in str(eval):
            return 20000
        if '-' in str(eval):
            return -20000
        else:
            return 0
    return int(eval)

def convert_to_pawn_advantage(output):
    output *= 0.2250
    output += 0.5385
    output = max(output, 1e-10)
    output = min(output, 1-1e-10)
    return 400 * math.log10(output/(1-output))

def predict_model(fen):
    encoding = convert_fen_to_encoding(fen)

    model = torch.nn.Sequential(
        torch.nn.Linear(774, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Linear(2048, 1),
    )

    model.load_state_dict(torch.load('models/more_models/model_70.pt', map_location='cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    with torch.no_grad():
        encoding = torch.Tensor(encoding)
        if torch.cuda.is_available():
            encoding = encoding.cuda()
        output = model(torch.unsqueeze(encoding, dim=0))
        # print(output.item(), convert_to_pawn_advantage(output.item()))

    return convert_to_pawn_advantage(output.item())

class TestingBot:
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        opening_books = ['vitamin17','Human','Titans','baron30']

        for book in opening_books:
            with chess.polyglot.open_reader(f'data/{book}.bin') as reader:
                board_hash = chess.polyglot.zobrist_hash(board)
                if reader.get(board_hash) is not None:
                    print(f'Found in {book} book.')
                    return reader.weighted_choice(board_hash).move

        best_value = 1000000
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            # print(move)
            best_anti_value = -1000000
            for second_move in board.legal_moves:
                board.push(second_move)
                fen = board.fen()
                value = predict_model(fen)
                if (value > best_anti_value):
                    best_anti_value = value
                board.pop()

            if best_anti_value < best_value:
                best_value = best_anti_value
                best_move = move
            board.pop()

        print(best_move)
        return best_move

        # depth = 3
        # best_value = -1000000
        # for move in board.legal_moves:
        #     board.push(move)
        #     value = self.tree_search(board, depth, -1000000, 1000000)
        #     board.pop()
            
        #     if value > best_value:
        #         best_value = value
        #         best_move = move

        return best_move

        # print("Random Move")
        # moves = list(board.legal_moves)
        # return moves[int(len(moves) * random.random())]

    def tree_search(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.material_evaluate(board)

        best_value = -1000000
        for move in board.legal_moves:
            board.push(move)
            value = -self.tree_search(board, depth - 1, -beta, -alpha)
            board.pop()
            if value > best_value:
                best_value = value
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return best_value

    '''
    @staticmethod
    def tree_search(board, height, alpha, beta, alphaTurn):
        if height==0 or board.is_game_over():
            return .material_evaluate(board)
    '''    

    @staticmethod
    def material_evaluate(board):
        if board.is_checkmate():
            board.outcome(chess.Termination()) #TODO: Reimplement 

        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0

        material = 0

        type_pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        value_pieces = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10000}
        color_pieces = np.array([chess.WHITE, chess.BLACK])
        all_pieces = []
        for type_piece in type_pieces:
            for color_piece in color_pieces:
                all_pieces.append((type_piece, color_piece))

        for piece, color in all_pieces:
            if color == chess.WHITE:
                material += len(board.pieces(piece, color)) * value_pieces[piece]
            if color == chess.BLACK:
                material -= len(board.pieces(piece, color)) * value_pieces[piece]

        return material
        