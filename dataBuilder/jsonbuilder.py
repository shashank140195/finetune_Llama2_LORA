import os, json
from os.path import splitext, basename

split = "valid"

# path to raw files
ann_path = f"/Users/home/{split}/ann"
text_path = f"/Users/home/{split}/input_text"

final_doc = dict()
null_cnt = 0

with open(f'/Users/home/{split}.json', 'w') as outfile:
    for document in os.listdir(ann_path):
        relation_present = 0

        with open(os.path.join(ann_path, document), "r") as in_f:
            doc_type = splitext(basename(document))[1]
            if doc_type == ".ann":
                document = document.replace(".ann", "")
                text = open(os.path.join(text_path, "%s.txt" % document)).read()
                text = text.replace("\u2019", "'")
                text = text.replace("\u201c", '"')
                text = text.replace("\u201d", '"')
                text = text.replace("\u2013", '-')
                text = text.replace("\u2018", "'")
                text = text.replace("\u00ed", "i")
                text = text.replace("\u00f6", "o")
                text = text.replace("\u00e9", "e")
                text = text.replace("\u00e7", "c")
                text = text.replace("\u00a0", " ")
                text = text.replace("\u00e8", "e")
                text = text.replace("\u2014", '-')
                text = text.replace("\u00ba", ' ')
                text = text.replace("\u03b2", 'B')

                content = text.strip().replace('\n', ' ')
                content = content.lower().strip()
                instruction = "as a relation extractor assistant, identify the relations in this document. do not " \
                              "generate any tokens outside of the document.\n"
                #
                content = f"instruction:{instruction}document:{content}\n"

                ent_dict = {}
                triples = []

                for line in in_f:
                    if line[0] == "T":
                        ent_data = line.split("\t")
                        ent_type = ent_data[1].split()[0]
                        mention = ent_data[-1].strip().replace('\n', '')

                        if ent_type == "ANAPHOR":
                            mention = "\"" + mention + "\""

                        # only entity spans
                        # ent_dict[ent_data[0]] = mention.lower()

                        # include entity type
                        ent_dict[ent_data[0]] = (ent_type + " " + mention).lower()

                    elif line[0] == "R":
                        temp = dict()
                        rel_data = line.split("\t")
                        relation_present = 1
                        relation_type = rel_data[1].split()[0]
                        rel_arg_1 = rel_data[1].split()[1][5:]
                        rel_arg_2 = rel_data[1].split()[2][5:]
                        temp["rel"] = relation_type.lower()

                        if rel_arg_1 in ent_dict.keys():
                            temp["arg1"] = ent_dict[rel_arg_1]
                        else:
                            print(f"Entity ID {rel_arg_1} does not exist in the annotation file of {document}")
                            continue

                        if rel_arg_2 in ent_dict.keys():
                            temp["arg2"] = ent_dict[rel_arg_2]
                        else:
                            print(f"Entity ID {rel_arg_2} does not exist in the annotation file of {document}")
                            continue

                        triples.append(temp)

                if relation_present == 0:
                    triples.append({"rel": "no relation"})
                    print(f"Following document {document} has no relations: ")
                    final_doc[document] = {"abstract": content}
                    final_doc[document]["triples"] = triples
                    final_doc[document]["pmid"] = document
                    null_cnt += 1

                else:
                    final_doc[document] = {"abstract": content}
                    final_doc[document]["triples"] = triples
                    final_doc[document]["pmid"] = document

    json.dump(final_doc, outfile)
