import torch
from torch.serialization import save
from tqdm.contrib import tenumerate
import numpy as np
import pandas as pd
import chess
import matplotlib.pyplot as plt
import math
import time

class ChessDataset(torch.utils.data.Dataset):
    '''Chess dataset'''

    def __init__(self, file_path, encoded=False, save_path=None):
        if not encoded:
            df = pd.read_csv(file_path)
            row_encodings = []
            for _, row in df.iterrows():
                row_evaluation = [1/(1+10**(filter_mates(row['Evaluation'])/(-400)))]
                row_encoding = row_evaluation + convert_fen_to_encoding(row['FEN'])
                row_encodings.append(row_encoding)

            row_encodings = np.array(row_encodings)

            columns_list = ['Evaluation']
            for i in range(774):
                columns_list.append('Encoding_' + str(i))
            df_encoded = pd.DataFrame(row_encodings, columns=columns_list)

            if save_path is None:
                df_encoded.to_csv('data/EncodedDataset.csv', index=False)
            else:
                df_encoded.to_csv(save_path, index=False)
        
        else:
            df_encoded = pd.read_csv(file_path)

        self.input = torch.Tensor(df_encoded.iloc[:, 1:].to_numpy())
        # input(self.input)
        self.output = torch.Tensor(df_encoded['Evaluation'])

        # self.output = torch.minimum(self.output, torch.quantile(self.output, 0.88))
        # self.output = torch.maximum(self.output, torch.quantile(self.output, 0.10))

        # input(torch.mean(self.output))
        # input(torch.std(self.output))

        self.output = self.output - torch.mean(self.output)
        self.output = self.output / torch.std(self.output)

        self.output = self.output.reshape((-1,1))

        # input(self.input.size())
        # input(self.output.size())
        
        if torch.cuda.is_available():
            self.input = self.input.cuda()
            self.output = self.output.cuda()

        # input(self.output[:25])

    def __getitem__(self, index):
        return {'input': self.input[index], 'output': self.output[index]}

    def __len__(self):
        return len(self.input)

        
piece_onehotindex = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11, '.': -1}
board_encoding = [0] * (64 * 12)

def convert_fen_to_encoding(fen_string): # TODO: SPEED THIS UP
    encoding_times = [] 
    encoding_times.append(("Begin", time.time()))

    
    one_hot_dict = {'P': [1,0,0,0,0,0,0,0,0,0,0,0], 
                    'N': [0,1,0,0,0,0,0,0,0,0,0,0], 
                    'B': [0,0,1,0,0,0,0,0,0,0,0,0], 
                    'R': [0,0,0,1,0,0,0,0,0,0,0,0], 
                    'Q': [0,0,0,0,1,0,0,0,0,0,0,0], 
                    'K': [0,0,0,0,0,1,0,0,0,0,0,0], 
                    'p': [0,0,0,0,0,0,1,0,0,0,0,0], 
                    'n': [0,0,0,0,0,0,0,1,0,0,0,0], 
                    'b': [0,0,0,0,0,0,0,0,1,0,0,0], 
                    'r': [0,0,0,0,0,0,0,0,0,1,0,0], 
                    'q': [0,0,0,0,0,0,0,0,0,0,1,0], 
                    'k': [0,0,0,0,0,0,0,0,0,0,0,1], 
                    '.': [0,0,0,0,0,0,0,0,0,0,0,0]}
    

    fen_string_props = fen_string.split(' ') # Refer to: https://tynedalechess.wordpress.com/2017/11/05/fen-strings-explained/
    assert len(fen_string_props)==6
    encoding_times.append(("Split FEN_STRING", time.time()))

    # Store Board Positions
    rows = chess.Board(fen_string).__str__().split('\n')
    encoding_times.append(("chess.Board", time.time()))

    squares_encoding = []
    for row in rows:
        print(row)
        squares_encoding += list(map(lambda x: one_hot_dict[x], row.split(' ')))
    
    # print(squares_encoding)
    print(len(squares_encoding), type(squares_encoding), 
        len(squares_encoding[0]), type(squares_encoding[0]),
            type(squares_encoding[0][0]))
    encoding_times.append(("Board Positions Will's", time.time()))

    rows = chess.Board(fen_string).__str__().replace(" ", "").split('\n')
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            if piece_onehotindex[rows[i][j]]!=-1:
                board_encoding[12*8*i + 12*j + piece_onehotindex[rows[i][j]]] = 1
    encoding_times.append(("Board Positions Davin's Superior Code", time.time()))

    # Store Turn
    if fen_string_props[1] == 'w':
        turn = [1, 0]
    elif fen_string_props[1] == 'b':
        turn = [0, 1]
    else:
        print("Nobody's turn?")
        assert 0==1
    encoding_times.append(("Turn", time.time()))

    # Store Castle Privileges
    castle_privileges = fen_string_props[2]
    castle_privileges_encoding = [int(x in castle_privileges) for x in ['K', 'Q', 'k', 'q']]
    encoding_times.append(("Castle Privileges", time.time()))

    # Store Flattened Encoding
    flattened_squares_encoding = []
    for square in squares_encoding:
        flattened_squares_encoding += square
    print(flattened_squares_encoding)
    print("Two methods are equal?", board_encoding==flattened_squares_encoding)
    print([(i, board_encoding[i], flattened_squares_encoding[i]) 
            for i in range(768) if board_encoding[i]!=flattened_squares_encoding[i]])

    full_encoding = flattened_squares_encoding + turn + castle_privileges_encoding
    encoding_times.append(("Combine and Flatten Encoding", time.time()))

    
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            if piece_onehotindex[rows[i][j]]!=-1:
                board_encoding[12*8*i + 12*j + piece_onehotindex[rows[i][j]]] = 0
    
    encoding_times.append(("End", time.time()))
    encoding_durations = [(encoding_times[i][0], encoding_times[i][1] - encoding_times[i-1][1]) for i in range(1, len(encoding_times))]
    print(encoding_durations)
    
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

def main():
    dataset = ChessDataset('data/chessData_toy.csv', encoded=False)

    '''

    train_len = int(len(dataset)*0.8) 
    test_len = len(dataset) - train_len

    data_train, data_test = torch.utils.data.random_split(dataset, [train_len, test_len])

    # Load data
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=512, shuffle=True)

    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(774, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1),
    )
    # Loss and optimization functions
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    # Train model
    for epoch in range(1, 101):
        sum_loss = 0
        for _, elem in tenumerate(train_loader):
            # Forward pass
            output = model(elem['input'])
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, elem['output'])
            sum_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            # if batch_idx % 100 == 0:
            #     print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        avg_loss = sum_loss / len(train_loader)
        print(f'Average Loss Epoch {epoch}: {avg_loss}')

        # Test model
        model.eval()
        with torch.no_grad():
            sum_loss = 0
            for _,elem in tenumerate(test_loader):
                output = model(elem['input'])
                loss = loss_fn(output, elem['output'])
                sum_loss += loss.item()
            avg_loss = sum_loss / len(test_loader)
            print(f'Average Test Loss Epoch {epoch}: {avg_loss}')

        if epoch % 10 == 0:
            # Save model
            torch.save(model.state_dict(), f'model_{epoch}.pt')

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

    model.load_state_dict(torch.load('model_70.pt', map_location='cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    with torch.no_grad():
        encoding = torch.Tensor(encoding)
        if torch.cuda.is_available():
            encoding = encoding.cuda()
        output = model(torch.unsqueeze(encoding, dim=0))
        print(output.item(), convert_to_pawn_advantage(output.item()))

    return convert_to_pawn_advantage(output.item())

    # with torch.no_grad():
    #     for n in range(15):
    #         output = model(torch.unsqueeze(dataset[n]['input'], dim=0))
    #         print(output.item(), convert_to_pawn_advantage(output.item()))
    #         print(dataset[n]['output'].item(), convert_to_pawn_advantage(dataset[n]['output'].item()))
    #         print('-----')
    '''
        

if __name__ == "__main__":
    main()
    # predict_model('r1b1k2r/pppp1ppp/8/6B1/2QNn3/P1P5/2P2PPP/R3K2R b KQkq - 0 11')
