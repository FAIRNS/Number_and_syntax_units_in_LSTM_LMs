import os
import torch

class Dictionary(object):
    def __init__(self, path=None):
        self.word2idx = {}
        self.idx2word = []
        if path:
            self.load(path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def load(self, path):
        with open(path, 'r') as f:
            for line in f:
                self.add_word(line.rstrip('\n'))

    def save(self, path):
        with open(path, 'w') as f:
            for w in self.idx2word:
                f.write('{}\n'.format(w))

class Corpus(object):
    def __init__(self, path, vocab):
        self.dictionary = Dictionary(vocab)
        try:
            self.train, self.valid, self.test = torch.load(
                    open(os.path.join(path, 'corpus.pth'), 'rb'))
        except:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))
            print('Corpus added')
            with open(os.path.join(path, 'corpus.pth'), 'wb') as f:
                torch.save((self.train, self.valid, self.test), f)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        tokens = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
