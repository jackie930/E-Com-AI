# This file contains the evaluation functions

import re
import editdistance

sentiment_word_list = ['positive', 'negative', 'neutral']
aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']


# read list
# with open('tag.txt') as f:
#   tag_list = [i.replace("\n","") for i in f.readlines()]


def compute_scores_huabao(pred_seqs, gold_seqs, sents, io_format, task):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []
    all_labels_with_tag = []
    all_predictions_with_tag = []

    for i in range(num_samples):
        gold_list, gold_extractions_with_aspect = extract_spans_extraction(gold_seqs[i], 4)
        pred_list, pred_extractions_with_aspect = extract_spans_extraction(pred_seqs[i], 4)

        all_labels.append(gold_list)
        all_predictions.append(pred_list)

        all_labels_with_tag.append(gold_extractions_with_aspect)
        all_predictions_with_tag.append(pred_extractions_with_aspect)

    print("\nResults of raw output, total")
    raw_scores = compute_f1_scores(all_predictions_with_tag, all_labels_with_tag)
    print(raw_scores)

    print("\nResults of raw output, only tag category & sentiment")
    raw_scores_2 = compute_f1_scores(all_predictions, all_labels)
    print(raw_scores_2)

    return raw_scores, all_labels_with_tag, all_predictions_with_tag


def extract_spans_extraction(seq, label_len=2):
    extractions = []
    extractions_with_aspect = []
    all_pt = seq.split('; ')

    # print ("<<< all_pt", all_pt)
    for pt in all_pt:
        if label_len == 4:
            try:
                a, b, c, d = pt[1:-1].split(', ')
            except ValueError:
                continue
        extractions.append((b, d))
        extractions_with_aspect.append((a, b, c, d))
    return extractions, extractions_with_aspect


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    res = {}

    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        gold_pt[i] = list(set(gold_pt[i]))
        pred_pt[i] = list(set(pred_pt[i]))

        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'total precision': precision, 'total recall': recall, 'total f1': f1}

    return scores
