import torch.utils.data
import torch.nn.utils
import collections
import pickle
import os
import errno
import hashlib


class PrefixDataset(torch.utils.data.Dataset):
    wgets = [
        "https://github.com/dalab/hyperbolic_nn/raw/master/prefix_{num}_dataset/dev",
        "https://github.com/dalab/hyperbolic_nn/raw/master/prefix_{num}_dataset/id_to_word",
        "https://github.com/dalab/hyperbolic_nn/raw/master/prefix_{num}_dataset/test",
        "https://github.com/dalab/hyperbolic_nn/raw/master/prefix_{num}_dataset/train",
        "https://github.com/dalab/hyperbolic_nn/raw/master/prefix_{num}_dataset/word_to_id",
    ]
    md5 = {
        10: dict(
            dev="866becf7824cabbe31c3c65e54625893",
            id_to_word="69b79d947667fa12e8a42b1f19dd738e",
            test="532ccc6cde8b03e4669818498eec4199",
            train="a76befe750169fbe1bd849172314af37",
            word_to_id="b8d7501194055cf88953da8b065fdf1e",
        ),
        30: dict(
            dev="3da6b66756c667f9010b9470ca3e52a0",
            id_to_word="69b79d947667fa12e8a42b1f19dd738e",
            test="5ff3459d645cc4bfe526e8a7f4de5633",
            train="e7c09d477188fc7f1d9ccaf48811a8dd",
            word_to_id="df11bdd245435418f220575430bec77f",
        ),
        50: dict(
            dev="5f8e7882c24f8a05e1a616d16715e06f",
            id_to_word="69b79d947667fa12e8a42b1f19dd738e",
            test="a7df3a053d939c9cee61ad331a3af7d3",
            train="3b7f4b068ca77f5affba9496fff2efc8",
            word_to_id="f7977e8f6c8c8bddf279abcfe9c99864",
        ),
    }

    def __init__(self, root, num=10, split="train", download=False):
        assert num in {10, 30, 50}
        assert split in {"train", "test", "valid"}
        self.num = num
        self.split = split
        self.root = root
        if download:
            self.download()
        else:
            self._check_integrity()
        name = {"train": "train", "test": "test", "valid": "dev"}[split]
        self.data = pickle.load(open(os.path.join(root, self._suffix, name), "rb"))
        self.id2word = pickle.load(
            open(os.path.join(root, self._suffix, "id_to_word"), "rb")
        )
        self.word2id = pickle.load(
            open(os.path.join(root, self._suffix, "word_to_id"), "rb")
        )

    def __getitem__(self, item):
        first, first_len, second, second_len, label = self.data[item]
        return (torch.tensor(first), torch.tensor(second)), label

    def __len__(self):
        return len(self.data)

    @property
    def _suffix(self):
        return "prefix_{}".format(self.num)

    def _check_integrity(self):
        root = self.root
        for filename, md5 in self.md5[self.num].items():
            fpath = os.path.join(root, self._suffix, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        root = self.root
        for url in self.wgets:
            name = url.rsplit("/", 1)[-1]
            download_url(
                url.format(num=self.num),
                os.path.join(root, self._suffix),
                name,
                self.md5[self.num][name],
            )

    def __repr__(self):
        fmt_str = "Dataset {}-{}\n".format(self.__class__.__name__, self.num)
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Number of words: {}\n".format(len(self.id2word))
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root/data Location: {}\n".format(
            os.path.join(self.root, self._suffix)
        )
        return fmt_str

    @property
    def vocab_size(self):
        return len(self.id2word)


PrefixBatch = collections.namedtuple("PrefixBatch", "sequences,alignment,label")


def packing_collate_fn(sequences):
    sentences, labels = zip(*sequences)
    first, second = zip(*sentences)
    first = list(enumerate(first))
    second = list(enumerate(second))
    first.sort(key=lambda t: -len(t[1]))
    second.sort(key=lambda t: -len(t[1]))
    idx_first, first = zip(*first)
    idx_second, second = zip(*second)
    align = [idx_second.index(idx) for idx in idx_first]

    return PrefixBatch(
        (
            torch.nn.utils.rnn.pack_sequence(first),
            torch.nn.utils.rnn.pack_sequence(second),
        ),
        torch.tensor(align),
        torch.tensor(labels)[list(idx_first)],
    )


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, "rb") as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath)
