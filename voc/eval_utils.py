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

#read list
#with open('tag.txt') as f:
 #   tag_list = [i.replace("\n","") for i in f.readlines()]


def extract_spans_extraction(task, seq,label_len=2):
    extractions = []
    extractions_with_aspect = []
    if task == 'uabsa' and seq.lower() == 'none':
        return []
    else:
        if task in ['uabsa', 'aope']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b = pt.split(', ')
                except ValueError:
                    a, b = '', ''
                extractions.append((a, b))
        elif task in ['tasd', 'aste']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                if label_len==2:
                    try:
                        a, b = pt.split(', ')
                    except ValueError:
                        a, b = '', ''
                    extractions.append((b))
                    extractions_with_aspect.append((a,b))
                elif label_len==4:
                    try:
                        a, b, c, d = pt.split(', ')
                    except ValueError:
                        a, b, c, d = '', '', '', ''
                    extractions.append((b,d))
                    extractions_with_aspect.append((a, b,c,d))
        return extractions,extractions_with_aspect

def extract_spans_extraction_custom(task, seq):
    extractions_tag = []
    extractions_total = []
    if task == 'uabsa' and seq.lower() == 'none':
        return []
    else:
        if task in ['tasd', 'aste']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b, c = pt.split(', ')
                except ValueError:
                    a, b, c= '', '', ''
                extractions_tag.append((b))
                extractions_total.append((a,b,c))
        return extractions_tag,extractions_total


def extract_spans_annotation(task, seq):
    if task in ['aste', 'tasd']:
        extracted_spans = extract_triplets(seq)
    elif task in ['aope', 'uabsa']:
        extracted_spans = extract_pairs(seq)

    return extracted_spans


def extract_pairs(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    pairs = []
    for ap in aps:
        # the original sentence might have
        try:
            at, ots = ap.split('|')
        except ValueError:
            at, ots = '', ''

        if ',' in ots:  # multiple ots
            for ot in ots.split(', '):
                pairs.append((at, ot))
        else:
            pairs.append((at, ots))
    return pairs


def extract_triplets(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    triplets = []
    for ap in aps:
        try:
            a, b, c = ap.split('|')
        except ValueError:
            a, b, c = '', '', ''

        # for ASTE
        if b in sentiment_word_list:
            if ',' in c:
                for op in c.split(', '):
                    triplets.append((a, b, op))
            else:
                triplets.append((a, b, c))
        # for TASD
        else:
            if ',' in b:
                for ac in b.split(', '):
                    triplets.append((a, ac, c))
            else:
                triplets.append((a, b, c))

    return triplets


def recover_terms_with_editdistance(original_term, sent):
    words = original_term.split(' ')
    new_words = []
    for word in words:
        edit_dis = []
        for token in sent:
            edit_dis.append(editdistance.eval(word, token))
        smallest_idx = edit_dis.index(min(edit_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term


def fix_preds_uabsa(all_pairs, sents):
    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # AT not in the original sentence
                if pair[0] not in ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                if pair[1] not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(pair[1], sentiment_word_list)
                else:
                    new_sentiment = pair[1]

                new_pairs.append((new_at, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_aope(all_pairs, sents):
    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # print(pair)
                # AT not in the original sentence
                if pair[0] not in ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                # OT not in the original sentence
                ots = pair[1].split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)

                new_pairs.append((new_at, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


# for ASTE
def fix_preds_aste(all_pairs, sents):
    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # two formats have different orders
                p0, p1, p2 = pair
                # for annotation-type
                if p1 in sentiment_word_list:
                    at, ott, ac = p0, p2, p1
                    io_format = 'annotation'
                # for extraction type
                elif p2 in sentiment_word_list:
                    at, ott, ac = p0, p1, p2
                    io_format = 'extraction'

                # print(pair)
                # AT not in the original sentence
                if at not in ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(at, sents[i])
                else:
                    new_at = at

                if ac not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(ac, sentiment_word_list)
                else:
                    new_sentiment = ac

                # OT not in the original sentence
                ots = ott.split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)
                if io_format == 'extraction':
                    new_pairs.append((new_at, new_ot, new_sentiment))
                else:
                    new_pairs.append((new_at, new_sentiment, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_tasd(all_pairs, sents):
    #todo： the fix should be fix words to belong to orig sentence， fix category to belong to category list
    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # print(pair)
                # AT not in the original sentence
                sents_and_null = ' '.join(sents[i]) + 'NULL'
                if pair[0] not in sents_and_null:
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                # AC not in the list
                acs = pair[0]
                new_ac_list = []
                for ac in acs:
                    if ac not in tag_list:
                        new_ac_list.append(recover_terms_with_editdistance(ac, tag_list))
                        print ("<<<< fixed", recover_terms_with_editdistance(ac, tag_list))
                    else:
                        new_ac_list.append(ac)
                new_ac = ', '.join(new_ac_list)

                #if pair[2] not in sentiment_word_list:
                 #   new_sentiment = recover_terms_with_editdistance(pair[2], sentiment_word_list)
                #else:
                 #   new_sentiment = pair[2]

                new_pairs.append(new_ac)
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs

def fix_preds_tagcls(all_pairs, sents):
    #fix preds for tag classsification
    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                '''
                # AT not in the original sentence
                sents_and_null = ' '.join(sents[i]) + 'NULL'
                if pair[0] not in sents_and_null:
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]
                '''

                # AC not in the list
                #print("<<<before adjust aspect", pair)
                if pair not in tag_list:
                    new_ac = recover_terms_with_editdistance(pair, tag_list)
                else:
                    new_ac = pair

                # if pair[2] not in sentiment_word_list:
                #   new_sentiment = recover_terms_with_editdistance(pair[2], sentiment_word_list)
                # else:
                #   new_sentiment = pair[2]

                new_pairs.append(new_ac)
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_pred_with_editdistance(all_predictions, sents, task):
    if task == 'uabsa':
        fixed_preds = fix_preds_uabsa(all_predictions, sents)
    elif task == 'aope':
        fixed_preds = fix_preds_aope(all_predictions, sents)
    elif task == 'aste':
        fixed_preds = fix_preds_aste(all_predictions, sents)
    elif task == 'tasd':
        #fixed_preds = fix_preds_tasd(all_predictions, sents)
        #update tag fix
        fixed_preds = fix_preds_tagcls(all_predictions, sents)
    else:
        print("*** Unimplemented Error ***")
        fixed_preds = all_predictions

    return fixed_preds


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


def compute_f1_scores_tag(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    res = {}
    # update tag list
    flatten = lambda x: [subitem for item in x for subitem in flatten(item)] if type(x) is list else [x]
    res1 = flatten(pred_pt)
    res2 = flatten(gold_pt)
    res1.extend(res2)
    tag_list_gold = list(set(res1))
    for i in tag_list_gold:
        res[i] = {}
        res[i]['n_tp'] = 0
        res[i]['n_gold'] = 0
        res[i]['n_pred'] = 0

    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        gold_pt[i] = list(set(gold_pt[i]))
        pred_pt[i] = list(set(pred_pt[i]))

        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for j in gold_pt[i]:
            res[j]["n_gold"] += 1

        for t in pred_pt[i]:
            res[t]["n_pred"] += 1
            if t in gold_pt[i]:
                res[t]["n_tp"] += 1
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    # calculate score by tag
    for i in res:
        res[i]['precision'] = float(res[i]['n_tp']) / float(res[i]['n_pred']) if res[i]['n_pred'] != 0 else 0
        res[i]['recall'] = float(res[i]['n_tp']) / float(res[i]['n_gold']) if res[i]['n_gold'] != 0 else 0
        res[i]['f1'] = 2 * res[i]['precision'] * res[i]['recall'] / (res[i]['precision'] + res[i]['recall']) if res[i]['precision'] != 0 or res[i]['recall'] != 0 else 0

    scores = {'total precision': precision, 'total recall': recall, 'total f1': f1}
    print ("<<<< res", res)
    return scores,res


def compute_scores(pred_seqs, gold_seqs, sents, io_format, task):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []
    all_labels_with_tag = []
    all_predictions_with_tag = []

    for i in range(num_samples):
    #for i in range(10):
        if io_format == 'annotation':
            gold_list,gold_extractions_with_aspect = extract_spans_annotation(task, gold_seqs[i])
            pred_list,pred_extractions_with_aspect = extract_spans_annotation(task, pred_seqs[i])
        elif io_format == 'extraction':
            gold_list,gold_extractions_with_aspect = extract_spans_extraction(task, gold_seqs[i],2)
            pred_list,pred_extractions_with_aspect = extract_spans_extraction(task, pred_seqs[i],2)

        all_labels.append(gold_list)
        all_predictions.append(pred_list)

        all_labels_with_tag.append(gold_extractions_with_aspect)
        all_predictions_with_tag.append(pred_extractions_with_aspect)

    print("\nResults of raw output, only tag category")
    raw_scores, res_tag = compute_f1_scores_tag(all_predictions, all_labels)
    print(raw_scores)

    print("\nResults of raw output, total")
    raw_scores_2 = compute_f1_scores(all_predictions_with_tag, all_labels_with_tag)
    print(raw_scores_2)

    # fix the issues due to generation
    all_predictions_fixed = fix_pred_with_editdistance(all_predictions, sents, task)
    print ("raw results: ", all_predictions[:5])
    print ("all_predictions_fixed: ", all_predictions_fixed[:5])

    print("\nResults of fixed output")
    fixed_scores, fixed_res_tag = compute_f1_scores_tag(all_predictions_fixed, all_labels)
    print(fixed_scores)

    return raw_scores, res_tag, fixed_scores, all_labels, all_predictions, all_predictions_fixed,fixed_res_tag, all_labels_with_tag,all_predictions_with_tag


def compute_scores_jj(pred_seqs, gold_seqs, sents, io_format, task):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []
    all_labels_with_tag = []
    all_predictions_with_tag = []

    for i in range(num_samples):
        gold_list_tag,gold_extractions_total = extract_spans_extraction_custom(task, gold_seqs[i])
        pred_list_tag,pred_extractions_total = extract_spans_extraction_custom(task, pred_seqs[i])

        all_labels.append(gold_list_tag)
        all_predictions.append(pred_list_tag)

        all_labels_with_tag.append(gold_extractions_total)
        all_predictions_with_tag.append(pred_extractions_total)

    print("\nResults of raw output: only tag")
    raw_scores = compute_f1_scores(all_predictions, all_labels)
    print(raw_scores)

    print("\nResults of raw output: total")
    raw_scores = compute_f1_scores(all_predictions_with_tag, all_labels_with_tag)
    print(raw_scores)


    return raw_scores