import pandas as pd
import re

df = pd.read_csv("prediction.csv")


def split_sentence(line):
    sentences = re.split(r"; ", line)
    return list(set(sentences))


def convert_relis_sentence(sentence):
    ans = None
    segs = re.match(r"the relationship between (.*?) and (.*?) is (.*)", sentence)
    if segs is not None:
        segs = segs.groups()
        relation = segs[2].strip()
        rel = ''
        if relation == "hyponym":
            rel = "is_a"
        elif relation == "producer":
            rel = "produces"
        elif relation == "synonyms":
            rel = "is_synon"
        elif relation == "acronym":
            rel = "is_acron"
        elif relation == "heightens":
            rel = "increases_risk_of"
        elif relation == "antecedent":
            rel = "anaphora"

        ans = (segs[0].strip(), rel, segs[1].strip())

    elif sentence == "there are no relations in the document":
        ans = ("", "no relations", "")
    return ans


def do_eval(preds, golden):
    num_missing = 0
    fp = 0
    fn = 0
    tn = 0

    produce_fp, produce_tp, produce_fn = 0, 0, 0
    is_a_fp, is_a_tp, is_a_fn = 0, 0, 0
    iro_fp, iro_tp, iro_fn = 0, 0, 0
    syn_fp, syn_tp, syn_fn = 0, 0, 0
    acro_fp, acro_tp, acro_fn = 0, 0, 0
    anaphora_fp, anaphora_tp, anaphora_fn = 0, 0, 0

    columns = ['doc_name', 'gold_rels', 'pred_rels', 'rel_type', 'fp_prediction', "fn_prediction"]
    df = pd.DataFrame(columns=columns)
    idx = 0
    for gold, pred in zip(golden, preds):
        idx += 1
        gold_arg1_set, gold_arg2_set, gold_rel_set, gold_set = set(), set(), set(), set()
        pred_arg1_set, pred_arg2_set, pred_rel_set, pred_set = set(), set(), set(), set()
        gold_rel = ""

        for tp in gold:
            gold_rel = tp[1].strip().lower()
            if gold_rel != "no relations":
                arg1 = tp[0].strip().lower()
                arg2 = tp[2].strip().lower()
                gold_arg1_set.add(arg1)
                gold_arg2_set.add(arg2)
                gold_rel_set.add(gold_rel)
                gold_set.add((arg1, arg2, gold_rel))

        if pred:
            for p_tp in pred:
                rel = p_tp[1].strip().lower()
                if rel == "no relations" and gold_rel == "no relations":
                    tn += 1
                    continue

                elif gold_rel == "no relations" and rel != "no relations":
                    fp += len(pred_set)
                    continue

                elif gold_rel != "no relations" and rel == "no relations":
                    fn += len(gold_set)
                    continue

                arg1 = p_tp[0].strip().lower()
                arg2 = p_tp[2].strip().lower()
                pred_arg1_set.add(arg1)
                pred_arg2_set.add(arg2)
                pred_rel_set.add(rel)
                pred_set.add((arg1, arg2, rel))

        fp_rel = pred_rel_set - gold_rel_set

        new_rows = [{'doc_name': idx, 'gold_rels': gold_rel_set, 'pred_rels': pred_rel_set, 'rel_type': "",
                     'fp_prediction': "", 'fn_prediction': ""}]
        df = df.append(new_rows, ignore_index=True)

        fp_dic = dict()
        for x in fp_rel:
            fp_lst = set()
            for y in pred_set:
                if y[2] == x:
                    fp_lst.add(y)

            if x == "produces":
                produce_fp += len(fp_lst)
            elif x == "is_a":
                is_a_fp += len(fp_lst)
            elif x == "increases_risk_of":
                iro_fp += len(fp_lst)
            elif x == "is_synon":
                syn_fp += len(fp_lst)
            elif x == "is_acron":
                acro_fp += len(fp_lst)
            elif x == "anaphora":
                anaphora_fp += len(fp_lst)

            fp_dic[x] = fp_lst

        fn_dic = dict()
        fp_dic_2 = dict()

        for z in gold_rel_set:
            gold = set()
            prediction = set()
            for f in gold_set:
                if f[2] == z:
                    gold.add(f)
            for g in pred_set:
                if g[2] == z:
                    prediction.add(g)

            fn_rel = gold - prediction
            tp_rel = gold.intersection(prediction)
            fp_rels = prediction - gold

            if len(fn_rel) or len(tp_rel) or len(fp_rels) > 0:
                if z == "produces":
                    produce_fn += len(fn_rel)
                    produce_tp += len(tp_rel)
                    produce_fp += len(fp_rels)
                elif z == "is_a":
                    is_a_fn += len(fn_rel)
                    is_a_tp += len(tp_rel)
                    is_a_fp += len(fp_rels)
                elif z == "increases_risk_of":
                    iro_fn += len(fn_rel)
                    iro_tp += len(tp_rel)
                    iro_fp += len(fp_rels)
                elif z == "is_synon":
                    syn_fn += len(fn_rel)
                    syn_tp += len(tp_rel)
                    syn_fp += len(fp_rels)
                elif z == "is_acron":
                    acro_fn += len(fn_rel)
                    acro_tp += len(tp_rel)
                    acro_fp += len(fp_rels)
                elif z == "anaphora":
                    anaphora_fn += len(fn_rel)
                    anaphora_tp += len(tp_rel)
                    anaphora_fp += len(fp_rels)

                fn_dic[z] = fn_rel
                fp_dic_2[z] = fp_rels

        if len(fp_dic) > 0:
            for a in fp_dic.keys():
                new_rows = [{'doc_name': "", 'gold_rels': "", 'pred_rels': "", 'rel_type': a,
                             'fp_prediction': fp_dic[a], 'fn_prediction': ""}]
                df = df.append(new_rows, ignore_index=True)

        if len(fn_dic) > 0 or len(fp_dic_2) > 0:
            for b in fn_dic.keys():
                new_rows = [{'doc_name': "", 'gold_rels': "", 'pred_rels': "", 'rel_type': b,
                             'fp_prediction': fp_dic_2[b], 'fn_prediction': fn_dic[b]}]
                df = df.append(new_rows, ignore_index=True)

    # Produce Scores
    P_prod = produce_tp / (produce_tp + produce_fp)
    R_prod = produce_tp / (produce_tp + produce_fn)
    Fscore_prod = 2 * P_prod * R_prod / (P_prod + R_prod)

    print("Produce precision is: ", P_prod)
    print("Produce Recall is: ", R_prod)
    print("Produce F-score is: ", Fscore_prod)

    # Anaphora Scores
    P_ana = anaphora_tp / (anaphora_tp + anaphora_fp)
    R_ana = anaphora_tp / (anaphora_tp + anaphora_fn)
    Fscore_ana = 2 * P_ana * R_ana / (P_ana + R_ana)

    print("Anaphora precision is: ", P_ana)
    print("Anaphora Recall is: ", R_ana)
    print("Anaphora F-score is: ", Fscore_ana)

    # is_a Scores
    P_is_a, R_is_a, Fscore_is_a = 0, 0, 0
    if is_a_tp == 0:
        Fscore_is_a = 0
    else:
        P_is_a = is_a_tp / (is_a_tp + is_a_fp)
        R_is_a = is_a_tp / (is_a_tp + is_a_fn)
        Fscore_is_a = 2 * P_is_a * R_is_a / (P_is_a + R_is_a)

    print("is_a precision is: ", P_is_a)
    print("is_a Recall is: ", R_is_a)
    print("is_a F-score is: ", Fscore_is_a)

    # is_acron Scores
    P_is_acron, R_is_acron, Fscore_is_acron = 0, 0, 0
    if acro_tp == 0:
        Fscore_is_acron = 0

    else:
        P_is_acron = acro_tp / (acro_tp + acro_fp)
        R_is_acron = acro_tp / (acro_tp + acro_fn)
        Fscore_is_acron = 2 * P_is_acron * R_is_acron / (P_is_acron + R_is_acron)

    print("is_acron precision is: ", P_is_acron)
    print("is_acron Recall is: ", R_is_acron)
    print("is_acron F-score is: ", Fscore_is_acron)

    # is_synon Scores
    P_is_synon, R_is_synon, Fscore_is_synon = 0,0,0
    if syn_tp == 0:
        Fscore_is_synon = 0
    else:
        P_is_synon = syn_tp / (syn_tp + syn_fp)
        R_is_synon = syn_tp / (syn_tp + syn_fn)
        Fscore_is_synon = 2 * P_is_synon * R_is_synon / (P_is_synon + R_is_synon)

    print("is_synon precision is: ", P_is_synon)
    print("is_synon Recall is: ", R_is_synon)
    print("is_synon F-score is: ", Fscore_is_synon)

    # increase_risk_of Scores
    P_iro, R_iro, Fscore_iro = 0,0,0
    if iro_tp == 0:
        Fscore_iro = 0
    else:
        P_iro = iro_tp / (iro_tp + iro_fp)
        R_iro = iro_tp / (iro_tp + iro_fn)
        Fscore_iro = 2 * P_iro * R_iro / (P_iro + R_iro)

    print("increase_risk_of precision is: ", P_iro)
    print("increase_risk_of Recall is: ", R_iro)
    print("increase_risk_of F-score is: ", Fscore_iro)

    # Overall
    fp = produce_fp + is_a_fp + iro_fp + syn_fp + acro_fp + anaphora_fp
    fn += produce_fn + is_a_fn + iro_fn + syn_fn + acro_fn + anaphora_fn
    tp = produce_tp + is_a_tp + iro_tp + syn_tp + acro_tp + anaphora_tp

    P_overall = tp / (tp + fp)
    R_overall = tp / (tp + fn)
    F_overall = 2 * P_overall * R_overall / (P_overall + R_overall)

    print("Overall precision is: ", P_overall)
    print("Overall Recall is: ", R_overall)
    print("Overall F-score is: ", F_overall)

    # df.to_csv(
    #     "/Users/shashankgupta/Documents/Raredis/BioMedLM/same_data_as_bioGPT/without_token_copy_instruction/with_ent_type/rel_is/results/temp350/error_analysis_rel_type_latest.csv",
    #     index=False)


gold_lines = []
all_lines = []
for x in range(len(df)):
    gold = df["text"][x].split("output:")[1].strip()
    if "thank you for the information." in gold:
        gold = gold.replace("thank you for the information.", "")

    if len(gold) > 0 and gold[-1] == ".":
        gold_lines.append(gold[:-1])
    else:
        gold_lines.append(gold)

    pred = df["prediction"][x].split("output:")[1].strip()
    if "thank you for the information." in pred:
        pred = pred.replace("thank you for the information.", "")

    if len(pred) > 0 and pred[-1] == ".":
        all_lines.append(pred[:-1])
    else:
        all_lines.append(pred)

preds = []
golden = []
cnt = 0
fail_cnt = 0

for i, line in enumerate(zip(gold_lines, all_lines)):
    cnt += 1
    ret = []
    gold = []
    gold_sentences = split_sentence(line[0])
    pred_sentences = split_sentence(line[1])
    for sen in gold_sentences:
        gold_ans = convert_relis_sentence(sen)
        if gold_ans is not None:
            gold.append(gold_ans)
    golden.append(gold)

    for pred_Sen in pred_sentences:
        pred_ans = convert_relis_sentence(pred_Sen)
        if pred_ans is not None:
            ret.append(pred_ans)
    preds.append(ret)

do_eval(preds, golden)
