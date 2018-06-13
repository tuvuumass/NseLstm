import math
import sys
import time
import copy
import numpy as np
import six
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import chainer.links as L
import chainer

class NseLstm(chainer.Chain):

	"""docstring for NseLstmAt"""
	def __init__(self, n_voc_src, n_voc_trg, n_units, gpu, word_idx_trg):
		super(NseLstm, self).__init__(
			embed_src = L.EmbedID(n_voc_src, 512, ignore_label=-1),
			src_linear = L.Linear(512, n_units),
			h_x = L.Linear(2 * n_units, 2 * n_units),
			e_lstm = L.StatelessLSTM(n_units, n_units),
			e_lstm1 = L.StatelessLSTM(2 * n_units, n_units),

			embed_trg = L.EmbedID(n_voc_trg, 512, ignore_label=-1),
			trg_linear = L.Linear(512, n_units),
			d_lstm = L.StatelessLSTM(n_units, n_units),
			d_lstm1 = L.StatelessLSTM(n_units, n_units),
			d_in = L.Linear(n_units*2, n_units),

			w_ap = L.Linear(n_units, n_units),
			w_aw = L.Linear(n_units, n_units),
			w_we = L.Linear(n_units, 1),

			fc1 = L.Linear(n_units*2, n_units),
			fc2 = L.Linear(n_units, n_units),
			l_y = L.Linear(n_units, n_voc_trg))
		self.__word_idx_trg = word_idx_trg
		self.__n_units = n_units
		self.__gpu = gpu
		self.__mod = cuda.cupy if gpu >= 0 else np
		for param in self.params():
			data = param.data
			data[:] = np.random.uniform(-0.1, 0.1, data.shape)
		if gpu >= 0:
			cuda.get_device(gpu).use()
			self.to_gpu()

	def init_optimizer(self):
		self.__opt = optimizers.Adam(alpha=0.0003, beta1=0.9, beta2=0.999, eps=1e-08)
		self.__opt.setup(self)
		self.__opt.add_hook(chainer.optimizer.GradientClipping(15))
		self.__opt.add_hook(chainer.optimizer.WeightDecay(0.00003))

	def __attend_fast(self, hs, q, batch_size, train):
		w_a = F.reshape(F.batch_matmul(F.dropout(hs, ratio=0.0, train=train), q), (batch_size, -1))
		w_a = F.softmax(w_a)
		s_c = F.reshape(F.batch_matmul(w_a, hs, transa=True), (batch_size, -1))

		return s_c

	def __forward(self, train, x_batch, y_limit = 25, y_batch = None, fixed_emb=False):
		model = self
		n_units = self.__n_units
		mod = self.__mod
		word_idx_trg = self.__word_idx_trg
		src_len, batch_size = x_batch.shape
		trg_len, _ = y_batch.shape
			
		list_x = []
		for x in x_batch:
			x_data = mod.array(x, dtype=np.int32)
			x = Variable(x_data, volatile=not train)
			x = model.embed_src(x)
			x = model.src_linear(x)
			if fixed_emb:
				x = Variable(x.data, volatile=not train)
			list_x.append(x)

		list_h1 = F.concat([F.reshape(x, (batch_size, 1, n_units)) for x in list_x], axis=1)

		zeros = mod.zeros((batch_size, n_units), dtype=np.float32)
		lstm_h = Variable(zeros, volatile=not train)
		lstm_c = Variable(zeros, volatile=not train)
		lstm_h1 = Variable(zeros, volatile=not train)
		lstm_c1 = Variable(zeros, volatile=not train)

		list_a = []
		full_shape = (batch_size, src_len, n_units)

		for xs in list_x:
			lstm_c, lstm_h = model.e_lstm(lstm_c, lstm_h, F.dropout(xs, ratio=0.2, train=train))
			h1_w = F.reshape(F.batch_matmul(list_h1, lstm_h), (batch_size, -1))

			h1_a = F.softmax(h1_w)
			h2_r = F.reshape(F.batch_matmul(h1_a, list_h1, transa=True), (batch_size, -1))

			lr = model.h_x(F.concat([lstm_h, h2_r], axis=1))
			lstm_c1, lstm_h1 = model.e_lstm1(lstm_c1, lstm_h1, F.dropout(lr, ratio=0.2, train=train))
			list_a.append(lstm_h1)
			list_h1 = F.broadcast_to(F.reshape((1 - h1_a), (batch_size, src_len, 1)), full_shape) * list_h1
			list_h1 += F.broadcast_to(F.reshape(h1_a, (batch_size, src_len, 1)), full_shape)*F.broadcast_to(F.reshape(lstm_h1, (batch_size, 1, n_units)), full_shape)

		list_a = F.concat([F.reshape(h, (batch_size, 1, n_units)) for h in list_a], axis=1)

		ha = Variable(zeros, volatile=not train)
		x_data = mod.array([word_idx_trg['<bos>'] for _ in range(batch_size)], dtype=np.int32)
		t = Variable(x_data, volatile=not train)
		preds = [[] for _ in range(batch_size)]
		accum_loss = 0
		y_len = trg_len if train else y_limit
		for l in range(0, y_len):
			x = model.embed_trg(t)
			x = model.trg_linear(x)
			if fixed_emb:
				x = Variable(x.data, volatile=not train)
			x = model.d_in(F.concat([x, ha], axis=1))

			lstm_c, lstm_h = model.d_lstm(lstm_c, lstm_h, F.dropout(x, ratio=0.2, train=train))
			lstm_c1, lstm_h1 = model.d_lstm1(lstm_c1, lstm_h1, F.dropout(lstm_h, ratio=0.2, train=train))
			
			ha = self.__attend_fast(list_a, lstm_h1, batch_size, train)
			ha = F.relu(model.fc1(F.concat([ha, lstm_h1], axis=1)))
			ha = F.relu(model.fc2(ha))
			y = model.l_y(F.dropout(ha, ratio=0.2, train=train))
			output = cuda.to_cpu(y.data.argmax(1))
			for k in range(batch_size):
				preds[k].append(output[k])

			if l < trg_len:
				x_data = mod.array(y_batch[l], dtype=np.int32)
				t = Variable(x_data, volatile=not train)
				accum_loss += F.softmax_cross_entropy(y, t)
			if not train:
				x_data = mod.array(output, dtype=np.int32)
				t = Variable(x_data, volatile=not train)
		
		return preds, accum_loss

	def translate_beam(self, x_batch, max_length=40, beam=6):
		train = False
		mod = self.__mod
		word_idx_trg = self.__word_idx_trg
		n_units = self.__n_units
		model = self
		src_len, batch_size = len(x_batch), 1

		list_x = []
		for x in x_batch:
			x_data = mod.array([x], dtype=np.int32)
			x = Variable(x_data, volatile=not train)
			x = model.embed_src(x)
			x = model.src_linear(x)
			list_x.append(x)

		list_h1 = F.concat([F.reshape(x, (batch_size, 1, n_units)) for x in list_x], axis=1)

		zeros = mod.zeros((batch_size, n_units), dtype=np.float32)
		lstm_h = Variable(zeros, volatile=not train)
		lstm_c = Variable(zeros, volatile=not train)
		lstm_h1 = Variable(zeros, volatile=not train)
		lstm_c1 = Variable(zeros, volatile=not train)

		encoder_states = []
		full_shape = (batch_size, src_len, n_units)

		for xs in list_x:
			lstm_c, lstm_h = model.e_lstm(lstm_c, lstm_h, F.dropout(xs, ratio=0.2, train=train))
			h1_w = F.reshape(F.batch_matmul(list_h1, lstm_h), (batch_size, -1))

			h1_a = F.softmax(h1_w)
			h2_r = F.reshape(F.batch_matmul(h1_a, list_h1, transa=True), (batch_size, -1))

			lr = model.h_x(F.concat([lstm_h, h2_r], axis=1))
			lstm_c1, lstm_h1 = model.e_lstm1(lstm_c1, lstm_h1, F.dropout(lr, ratio=0.2, train=train))
			encoder_states.append(lstm_h1)
			list_h1 = F.broadcast_to(F.reshape((1 - h1_a), (batch_size, src_len, 1)), full_shape) * list_h1
			list_h1 += F.broadcast_to(F.reshape(h1_a, (batch_size, src_len, 1)), full_shape)*F.broadcast_to(F.reshape(lstm_h1, (batch_size, 1, n_units)), full_shape)
		
		encoder_states = F.concat([F.reshape(h, (1, 1, n_units)) for h in encoder_states], axis=1)
		encoder_states = Variable(encoder_states.data, volatile=not train) 

		lstm_h = Variable(lstm_h.data, volatile=not train)
		lstm_c = Variable(lstm_c.data, volatile=not train)
		lstm_h1 = Variable(lstm_h1.data, volatile=not train)
		lstm_c1 = Variable(lstm_c1.data, volatile=not train)
		rnn_states = [(lstm_c, lstm_h), (lstm_c1, lstm_h1)]
		ha = Variable(zeros, volatile=not train)

		ys = mod.array([word_idx_trg['<bos>']], dtype=np.int32)

		sum_ws = mod.zeros(1, 'f')
		result = [[]] * beam
		for i in six.moves.range(max_length):

			ws_concat, rnn_states, ha = self.step_decode(ys, ha, rnn_states, encoder_states, train)
			ws_concat = F.log_softmax(ws_concat)
			ws_concat = cuda.to_cpu(ws_concat.data)

			if i != 0:
				eos_sent_ids = np.flatnonzero(cuda.to_cpu(ys) == word_idx_trg['<eos>'])
				ws_concat[eos_sent_ids.tolist(), :] = - float('inf')
				ws_concat[eos_sent_ids.tolist(), word_idx_trg['<eos>']] = 0.

			ys_list, ws_list = self.get_topk(ws_concat, beam, axis=1)

			ys_concat = mod.concatenate(ys_list, axis=0)
			sum_ws_list = [ws + sum_ws for ws in ws_list]
			sum_ws_concat = cuda.to_cpu(mod.concatenate(sum_ws_list, axis=0))

			# Get top-k
			idx_list, sum_w_list = self.get_topk(sum_ws_concat, beam, axis=0)
			idx_concat = mod.stack(idx_list, axis=0)
			ys = ys_concat[idx_concat]
			sum_ws = mod.stack(sum_w_list, axis=0)

			if i != 0:
				old_idx_list = (idx_concat % beam).tolist()
			else:
				old_idx_list = [0] * beam

			result = [result[idx] + [y]
					  for idx, y in zip(old_idx_list, ys.tolist())]

			if mod.all(ys == word_idx_trg['<eos>']):
				break

			new_rnn_states = []
			for c,h in rnn_states:
				h = F.stack([Variable(h.data[idx], volatile=not train) for idx in old_idx_list], axis=0)
				c = F.stack([Variable(c.data[idx], volatile=not train) for idx in old_idx_list], axis=0)
				new_rnn_states.append((c,h))
			rnn_states = new_rnn_states

			ha = F.stack([Variable(ha.data[idx], volatile=not train) for idx in old_idx_list], axis=0)
			encoder_states = F.stack([Variable(encoder_states.data[idx], volatile=not train) for idx in old_idx_list], axis=0)

		return result

	def get_topk(self, x, k=5, axis=1):
		ids_list = []
		scores_list = []
		mod = self.__mod
		for i in six.moves.range(k):
			ids = x.argmax(axis=axis).astype('i')
			if axis == 0:
				scores = x[ids]
				x[ids] = - float('inf')
			else:
				scores = x[np.arange(ids.shape[0]), ids]
				x[np.arange(ids.shape[0]), ids] = - float('inf')
			ids_list.append(mod.array(ids))
			scores_list.append(mod.array(scores))
		return ids_list, scores_list

	def step_decode(self, x, ha, rnn_states, encoder_states, train):
		model = self
		batch_size, input_length, hidden_dim = encoder_states.data.shape
		new_states = []

		x = Variable(x, volatile=not train)
		x = model.embed_trg(x)
		x = model.trg_linear(x)
		x = model.d_in(F.concat([x, ha], axis=1))

		lstm_c, lstm_h = model.d_lstm(rnn_states[0][0], rnn_states[0][1], F.dropout(x, ratio=0.2, train=train))
		lstm_c1, lstm_h1 = model.d_lstm1(rnn_states[1][0], rnn_states[1][1], F.dropout(lstm_h, ratio=0.2, train=train))
		new_states.append((lstm_c, lstm_h))
		new_states.append((lstm_c1, lstm_h1))
		
		ha = self.__attend_fast(encoder_states, lstm_h1, batch_size, train)
		ha = F.relu(model.fc1(F.concat([ha, lstm_h1], axis=1)))
		ha = F.relu(model.fc2(ha))
		y = model.l_y(F.dropout(ha, ratio=0.2, train=train))

		return y, new_states, ha

	def train(self, x_batch, y_limit, y_batch, fixed_emb=False):
		preds, accum_loss = self.__forward(True, x_batch, y_limit, y_batch=y_batch, fixed_emb=fixed_emb)
		self.__opt.zero_grads()
		accum_loss.backward()
		self.__opt.update()
		return preds, accum_loss

	def predict(self, x_batch, y_limit, y_batch = None):
		return self.__forward(False, x_batch, y_limit, y_batch)