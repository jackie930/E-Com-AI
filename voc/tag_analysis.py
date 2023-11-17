import pandas as pd


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


def filter_tags(tag_ls, input_seq, filter_method):
    if filter_method == 'filter out':
        filtered_data = [item for item in input_seq if item[0] not in tag_ls]
    elif filter_method == 'filter in':
        filtered_data = [item for item in input_seq if item[0] in tag_ls]
    return filtered_data


def calc(picklefile, tag_ls=[], filter_method='none'):
    df = pd.read_pickle(picklefile)
    pred_seqs = df['outputs']
    gold_seqs = df['targets']
    all_labels, all_predictions = [], []
    all_labels_with_tag = []
    all_predictions_with_tag = []
    num_samples = len(gold_seqs)
    for i in range(num_samples):
        gold_list, gold_extractions_with_aspect = extract_spans_extraction(gold_seqs[i], 4)
        pred_list, pred_extractions_with_aspect = extract_spans_extraction(pred_seqs[i], 4)

        ##if filter tags
        if filter_method == 'filter out':
            gold_list_update = filter_tags(tag_ls, gold_list, filter_method)
            pred_list_update = filter_tags(tag_ls, pred_list, filter_method)
        elif filter_method == 'filter in':
            gold_list_update = filter_tags(tag_ls, gold_list, filter_method)
            pred_list_update = filter_tags(tag_ls, pred_list, filter_method)
        elif filter_method == 'none':
            gold_list_update = gold_list
            pred_list_update = pred_list

        all_labels.append(gold_list_update)
        all_predictions.append(pred_list_update)

    print("\nResults of raw output, only tag category & sentiment")
    raw_scores_2 = compute_f1_scores(all_predictions, all_labels)
    print(raw_scores_2)
    return

#全部评估
calc('results-tasd-huabaoallnoadjust-extraction-allres.pickle')

#过滤部分标签评估
calc('results-tasd-huabaoallnoadjust-extraction-allres.pickle',['Output performance','Portability','Charging accessories'],'filter out')

#只选用部分标签评估
calc('results-tasd-huabaoallnoadjust-extraction-allres.pickle',['User manual','Customer support','Battery capacity'],'filter in')