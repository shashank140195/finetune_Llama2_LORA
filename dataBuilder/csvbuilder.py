import os
import json
import pandas as pd

data_dir = "/Users/home/"


def sort_triples(triples, text):
    sorted_triples = sorted(triples, key=lambda x: text.find(x['rel']))
    return sorted_triples


def build_target_seq_relis_llama2(triples):
    answer = "thank you for the information."
    for z in triples:
        rel = z["rel"].lower()
        if rel == "no relation":
            answer = "there are no relations in the document; "

        else:
            arg1 = z["arg1"].lower()
            arg2 = z["arg2"].lower()
            rel = z["rel"].lower()

            if rel == "is_a":
                answer += f"the relationship between {arg1} and {arg2} is hyponym; "

            elif rel == "produces":
                answer += f"the relationship between {arg1} and {arg2} is producer; "

            elif rel == "is_synon":
                answer += f"the relationship between {arg1} and {arg2} is synonyms; "

            elif rel == "is_acron":
                answer += f"the relationship between {arg1} and {arg2} is acronym; "

            elif rel == "increases_risk_of":
                answer += f"the relationship between {arg1} and {arg2} is heightens; "

            elif rel == "anaphora":
                answer += f"the relationship between {arg1} and {arg2} is antecedent; "

    return answer[:-2] + "."


def loader(fname, fn):
    ret = []
    null_cnt = 0
    suc_cnt = 0
    null_flag = False
    column_names = ["doc_name", "text"]
    df = pd.DataFrame(columns=column_names)

    with open(fname, "r", encoding="utf8") as fr:
        data = json.load(fr)
    for pmid, v in data.items():
        content = v["abstract"].strip()

        # content = content.lower()
        if v["triples"] is None or len(v["triples"]) == 0:
            if not null_flag:
                print(f"Following PMID in {fname} has no extracted triples:")
                null_flag = True
            print(f"{pmid} ", end="")
            null_cnt += 1

        else:
            triples = v['triples']
            # triples = sort_triples(triples, content)
            answer = fn(triples)
            text = f"{content}\noutput: {answer}"
            suc_cnt += 1
        new_row = [{'doc_name': pmid, 'text': text}]
        df = df.append(new_row, ignore_index=True)

    df.to_csv(f"/Users/home/{split}.csv")


def worker(fname, fn, split):
    loader(fname, fn)


for split in ['train', 'valid', 'test']:
    worker(os.path.join(f"{data_dir}", f"{split}.json"), build_target_seq_relis_llama2, split)
