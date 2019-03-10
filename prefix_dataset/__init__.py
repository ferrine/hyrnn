import torch.utils.data
import torch.nn.utils
import pickle
import os


class PrefixDatset(torch.utils.data.Dataset):
    def __init__(self, root, type="train"):
        if type not in {"train", "test", "valid"}:
            raise ValueError("f{type} not in {\"train\", \"test\", \"valid\"}")
        self.data = pickle.load(open(os.path.join(root, type), "rb"))
        self.id2word = pickle.load(open(os.path.join(root, "id_to_word"), "rb"))
        self.word2id = pickle.load(open(os.path.join(root, "word_to_id"), "rb"))

    def __getitem__(self, item):
        first, first_len, second, second_len, label = self.data[item]
        return (torch.tensor(first), torch.tensor(second)), label

    def __len__(self):
        return len(self.data)


def packing_collate_fn(sequences):
    sentences, labels = zip(*sequences)
    firsts, seconds = zip(*sentences)
    return (torch.nn.utils.rnn.pad_sequence(firsts), torch.nn.utils.rnn.pad_sequence(seconds)), torch.tensor(labels)
