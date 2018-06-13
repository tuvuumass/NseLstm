#!/usr/bin/python

def translate_post(y_true, y_pred, word_idx_src, word_idx_trg):
	y_true_str = []
	for ys in y_true:
		ys.insert(0, -1)
		ys = ys[:ys.index(word_idx_trg['<eos>'])]
		ys = ys[(len(ys) - ys[::-1].index(-1)):]
		y_true_str.append([[str(y) for y in ys]])

	y_pred_str = []
	for ys in y_pred:
		ys.append(word_idx_trg['<eos>'])
		y_pred_str.append([str(y) for y in ys[:ys.index(word_idx_trg['<eos>'])]])
	return y_true_str, y_pred_str