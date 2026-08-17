from torch import tensor
from transformers import GPT2Tokenizer

class PlainDataLoader(object):
    def __init__(
        self,
        # data_path: str,
        tokenID: list,
        tokenizer: GPT2Tokenizer,
        max_len: int, # <= the block size of the model
    ) -> None:
        # NOTE: we assume the data file is separated lines in PlainDataLoader
        self.tokenID = tokenID
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target = self.tokenID + [tokenizer.eos_token_id]
        self.data = [tokenizer.bos_token_id] + self.tokenID

        self.idx = 0
        self.x, self.y = [], []
        self.xw, self.yw = [], []

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.data):
            print(f'Semantic feature vec:\n {tensor(self.x[1:])}')
            raise StopIteration
        else:
            x_data = self.data[self.idx]
            y_data = self.target[self.idx]
            # print(f'x: {self.tokenizer.decode(x_data)}-->y: {self.tokenizer.decode(y_data)}')
            self.x.append(x_data)
            self.y.append(y_data)

            self.idx += 1

            self.xw = self.x[-self.max_len:]
            self.yw = self.y[-self.max_len:]
            return {
                'x': self.xw,
                'y': self.yw
            }

