import numpy as np
import codecs
import operator
from collections import defaultdict

class TSGeneratorBucket(object):
    def __init__(self, data_file, n_pair=None, vocab_src=None, vocab_tgt=None, max_len=40,
                 min_len=5, n_freq_src=25000, n_freq_tgt=25000,
				 src_bucket_step=4, tgt_bucket_step=8, batch_size=32,
                 train=True):
        super(TSGeneratorBucket, self).__init__()
        self.data_file = data_file
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_len = max_len
        self.min_len = min_len
        self.n_pair = n_pair
        self.n_freq_src = n_freq_src
        self.n_freq_tgt = n_freq_tgt
        self.UNK = '<unk>'
        self.EOS = '<eos>'
        self.BOS = '<bos>'
        self.EMPTY_ID = -1
        self.src_bucket_step = src_bucket_step
        self.tgt_bucket_step = tgt_bucket_step
        self.batch_size = batch_size
        self.train = train
        self.data = self._load_data(self.data_file)

    def _load_data(self, data_file):

        sent_src = []
        sent_tgt = []
        for i in xrange(len(data_file)):
            sent_src += codecs.open(data_file[i][0], "r", "utf-8").read().split('\n')
            sent_tgt += codecs.open(data_file[i][1], "r", "utf-8").read().split('\n')

        print "# pairs (before): ", len(sent_src)

        src_tgt_sent = [(src, tgt) for src, tgt in zip(sent_src, sent_tgt)
                        if len(src.strip().split()) <= self.max_len
                        and len(src.strip().split()) >= self.min_len
                        and len(tgt.strip().split()) <= self.max_len
                        and len(tgt.strip().split()) >= self.min_len]
        sent_src, sent_tgt = zip(*src_tgt_sent)
        print "# pairs (after): ", len(sent_src)

        if self.n_pair:
            sent_src = sent_src[:self.n_pair]
            sent_tgt = sent_tgt[:self.n_pair]

        self.n_pair = len(sent_src)

        if not self.vocab_src:
            self.vocab_src, self.total_vocab_src = self.const_vocab(sent_src, self.n_freq_src)

        if not self.vocab_tgt:
            self.vocab_tgt, self.total_vocab_tgt = self.const_vocab(sent_tgt, self.n_freq_tgt)

        self.n_voc_src = len(self.vocab_src)
        self.n_voc_tgt = len(self.vocab_tgt)

        sent_src = [self.word2id(sent.strip().split(), self.vocab_src) for sent in sent_src]
        sent_tgt = [self.word2id(sent.strip().split(), self.vocab_tgt) for sent in sent_tgt]

        buckets = defaultdict(list)
        pair_id = 0
        batches = []
        for src_ids, tgt_ids in zip(sent_src, sent_tgt):
            while len(src_ids) % self.src_bucket_step > 0:
                src_ids.append(self.EMPTY_ID)
            if self.train:
                tgt_ids.append(self.vocab_tgt[self.EOS])
                while len(tgt_ids) % self.tgt_bucket_step > 0:
                    tgt_ids.append(self.EMPTY_ID)
                buckets[len(src_ids), len(tgt_ids)].append((src_ids, tgt_ids))
            else:
                # when testing lets batch it with unique id so that we can keep the order
                tgt_ids.append(self.vocab_tgt[self.EOS])
                while len(tgt_ids) % self.tgt_bucket_step > 0:
                    tgt_ids.append(self.EMPTY_ID)
                buckets[pair_id].append((src_ids, tgt_ids))
                samples = [(src_ids, tgt_ids)]
                src_ids_lst, tgt_ids_lst = zip(*samples)
                src_ids_arr = np.asarray(src_ids_lst, dtype=np.int32).T
                tgt_ids_arr = np.asarray(tgt_ids_lst, dtype=np.int32).T
                batch = src_ids_arr, tgt_ids_arr
                batches.append(batch)
            pair_id += 1
        if self.train:
            self.buckets = buckets
            batches = self.get_batches(self.batch_size)

        return batches

    def get_batches(self, batch_size=None):
        batch_size = batch_size if batch_size else self.batch_size
        batches = []
        for samples in self.buckets.values():
            np.random.shuffle(samples)
            for i in range(0, len(samples), batch_size):
                src_ids_lst, tgt_ids_lst = zip(*samples[i:i + batch_size])
                src_ids_arr = np.asarray(src_ids_lst, dtype=np.int32).T
                tgt_ids_arr = np.asarray(tgt_ids_lst, dtype=np.int32).T
                batch = src_ids_arr, tgt_ids_arr
                batches.append(batch)
        return batches

    def const_vocab(self, sents, n_freq=25000):
        vocab_n = {}
        for sent in sents:
            words = sent.strip().split()
            for w in words:
                vocab_n[w] = vocab_n.get(w, 0) + 1
        vocab_n = sorted(vocab_n.items(), key=operator.itemgetter(1), reverse=True)
        total_vocab = len(vocab_n)
        vocab_n = [k for k, v in vocab_n[:n_freq]]

        vocab = {self.UNK: 0, self.EOS: 1, self.BOS: 2}
        vocab.update(dict((c, i + len(vocab)) for i, c in enumerate(vocab_n)))
        return vocab, total_vocab

    def word2id(self, words, vocab):
        return [vocab.get(word, vocab[self.UNK]) for word in words]