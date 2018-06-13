import os
import time
import random

import chainer

from nltk.translate import bleu_score
from utils import translate_post
from generators import TSGeneratorBucket

from LstmLstm import LstmLstm
from NseLstm import NseLstm

if __name__ == '__main__':

	dir_path = 'path-to-data/'
	gpu = -1
	n_epochs   = 100   # number of epochs
	n_units   = 300  # number of units per layer
	train_batch_size = 32
	eval_batch_size = 32

	if gpu >= 0:
		chainer.cuda.get_device(gpu).use()

	begin_time = time.time()
	train_generator = TSGeneratorBucket(data_file=[(dir_path + 'train.src', dir_path + 'train.tgt')],
										# n_pair = 1000,
										max_len = 80,
										min_len = 4,
										n_freq_src = 30000,
										n_freq_tgt = 30000,
										batch_size=train_batch_size)
	print 'secs for building train data={}'.format(time.time() - begin_time)

	vocab_src = train_generator.vocab_src
	vocab_tgt = train_generator.vocab_tgt

	dev_generator = TSGeneratorBucket(data_file=[(dir_path + 'dev.src', dir_path + 'dev.tgt')],
										# n_pair = 100,
										max_len = 80,
										min_len = 4,
										vocab_src = vocab_src,
										vocab_tgt = vocab_tgt,
										train = False,
										batch_size=eval_batch_size)

	test_generator = TSGeneratorBucket(data_file=[(dir_path + 'test.src', dir_path + 'test.tgt')],
										# n_pair = 100,
										max_len = 80,
										min_len = 4,
										vocab_src = vocab_src,
										vocab_tgt = vocab_tgt,
										train = False,
										batch_size=eval_batch_size)

	n_train = train_generator.n_pair
	n_dev = dev_generator.n_pair
	n_test = test_generator.n_pair

	print "n_pairs train: ", n_train
	print "n_pairs dev: ", n_dev
	print "n_pairs test: ", n_test
	print "n_voc_src: ", train_generator.n_voc_src
	print "n_voc_tgt: ", train_generator.n_voc_tgt
	print "total n_voc_src: ", train_generator.total_vocab_src
	print "total n_voc_tgt: ", train_generator.total_vocab_tgt

	idx_vocab_src = {v:k for k,v in vocab_src.iteritems()}
	idx_vocab_tgt = {v:k for k,v in vocab_tgt.iteritems()}

	train = train_generator.data
	dev = dev_generator.data
	test = test_generator.data

	print "batch_size: ", train_batch_size
	print "GPU:", gpu

	print "Building model ..."
	model = NseLstmAt(train_generator.n_voc_src, train_generator.n_voc_tgt, n_units, gpu, train_generator.vocab_tgt)
	# model = LstmLstmAt(train_generator.n_voc_src, train_generator.n_voc_tgt, n_units, gpu, train_generator.vocab_tgt)
	print "model: ", model
	model.init_optimizer()
	max_epch = 0
	max_tr = 0
	max_dev = 0
	max_test = 0
	print "Training ..."
	for i in xrange(0, n_epochs):
		print "Epoch={}".format(i)

		train = train_generator.get_batches()
		random.shuffle(train)

		preds = []
		preds_true = []
		train_loss = 0

		begin_time = time.time()
		for j, batch in enumerate(train):
			x_batch, y_batch = batch
			preds_true.extend(y_batch.T.tolist())
			y_s, loss = model.train(x_batch, len(y_batch[0]), y_batch, fixed_emb=False)
			preds.extend(y_s)
			train_loss += loss.data

		print "train loss: ", train_loss/(j+1)
		print 'secs per train epoch={}'.format(time.time() - begin_time)

		preds_true, preds = translate_post(preds_true, preds, vocab_src, vocab_tgt)
		bleu = bleu_score.corpus_bleu(preds_true, preds, smoothing_function=bleu_score.SmoothingFunction().method1) * 100
		print "train BLEU: ", bleu

		preds = []
		preds_true = []
		dev_loss = 0
		for j, batch in enumerate(dev):
			x_batch, y_batch = batch
			preds_true.extend(y_batch.T.tolist())
			y_s, loss  = model.predict(x_batch, 60, y_batch)
			preds.extend(y_s)
			dev_loss += dev_loss.data
		print "dev loss:", dev_loss/(j+1)

		preds_true, preds = translate_post(preds_true, preds, vocab_src, vocab_tgt)
		bleu = bleu_score.corpus_bleu(preds_true, preds, smoothing_function=bleu_score.SmoothingFunction().method1) * 100
		print "dev BLEU (greedy search): ", bleu

		preds = []
		preds_true = []
		for j, batch in enumerate(dev):
			x_batch, y_batch = batch
			preds_true.extend(y_batch.T.tolist())
			for x in x_batch.T.tolist():
				y_s = model.translate_beam(x, beam=10, max_length=60)
				preds.extend(y_s[:1])

		preds_true, preds = translate_post(preds_true, preds, vocab_src, vocab_tgt)
		bleu = bleu_score.corpus_bleu(preds_true, preds, smoothing_function=bleu_score.SmoothingFunction().method1) * 100
		print "dev BLEU (beam search): ", bleu
