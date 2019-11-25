# Usage: python ./OpenNMT-py/translate.py -model model.pt -src devel.input -output pred.txt -replace_unk -verbose --max_length 50 -attn_debug > debug.txt
#        python attention_visualization.py --input debug.txt

import sys
import argparse
import numpy as np
import re


def read_data(args):

    with open(args.input, "rt", encoding="utf-8") as f:

        preline = ""
        attn_block = False
        data = []
        pred_scores = []

        for line in f:
            line = line.strip()
            if not line:
                preline = ""
                attn_block = False
                continue

            if line.startswith('SENT '):
                if len(data) > 0:
                    #data[-1][0] = source
                    for i in range(1, len(data[-1])):
                        try:
                            data[-1][i][0] = target[i-1]
                        except IndexError:
                            data[-1][i][0] = '</s>'

                    data[-1][0] = source
                data.append([])
                source = eval(line[line.index(':')+2:])
            if re.search("^PRED \d+", line):
                target = line[line.index(':')+2:].split()
            if preline.startswith("PRED SCORE:"):
                attn_block = True
                pred_scores.append(float(preline.split(':')[1].strip()))
            if line.startswith("PRED AVG SCORE:"):
                attn_block = False
            if attn_block:
                data[-1].append(line.split())
            preline = line

    return data, pred_scores

def process_data(data):

    processed_data = []

    for sentence in data:
        source, rest = sentence[0], sentence[1:]
        target = [ w[0] for w in rest ]
        weights = [ w[1:] for w in rest ]
        weights = [[float(j.replace("*", "")) for j in i] for i in weights]

        processed_data.append( (source, target, weights) )

    return processed_data



def longest_cont(array):

    longest = [[array[0]]]
    for val in array[1:]:
        if val == longest[-1][-1]+1:
            longest[-1].append(val)
        else:
            longest.append([val])

    return longest


def prune_empty_regions(source, target, weights):

    # dummy solution to remove zero value regions from the heatmap
    # should be rewritten with numpy magic

    weight_array = np.array(weights)

    empty_columns = [] # indices of colums where all rows are close to zero

    for i in range(len(source)): # i is a column index
        is_empty = True
        for j in range(len(target)): # j is a row index
            if weights[j][i] >= 0.05:
                is_empty = False
                break
        if is_empty:
            empty_columns.append(i)

    cont_empty_subseq = longest_cont(empty_columns)

    keep_columns = [i for i in range(len(source)) if i not in empty_columns] # all non empty columns

    for subseq in cont_empty_subseq: # add small empty regions

        if False and len(subseq) > 10: # prune!!!
            for i in subseq[:3]+subseq[-3:]:
                keep_columns.append(i)
            source[subseq[2]] = source[subseq[2]]+"..."
            source[subseq[-3]] = "..."+source[subseq[-3]]
        else: # keep!!!
            for i in subseq:
                keep_columns.append(i)

    keep_columns.sort()


    a = weight_array[:, keep_columns]
    source_ = [word for i, word in enumerate(source) if i in keep_columns ]


    return source_, target, a


def calc_attn_errors(data):

    errors = []
    for i, (source, target, weights) in enumerate(data):

        source, target, weights = prune_empty_regions(source, target, weights)

        ## Calculate error scores
        copy_error_scores = []
        repeat_error_scores = []
        for j in range(len(source)):
            copy_error_scores.append(0.0)
            repeat_error_scores.append(0.0)
            """if re.search("</\w+>$", source[j]):
                in_tag = False
                print(np.mean(copy_error_scores),j)
                continue
            elif re.search("^<\w+>", source[j]):
                in_tag = True
                copy_error_scores = []
                continue"""
            for i,w in enumerate(weights[:,j]):
                if weights[i,j] > 0.01 and (target[i] != source[j] or re.search("</?\w+>", target[i])):
                    copy_error_scores[-1] += weights[i,j]
                    print("copy", weights[i,j],source[j],target[i])
                if weights[i,j] > 0.01 and weights[i,j] < max(weights[:,j]):
                    if target[i] == source[j] and source[j] not in list("–▁ .,-"):
                        repeat_error_scores[-1] += weights[i,j]
                        print("repeat", weights[i,j], target[i])

            repeat_error_scores[-1] /= source.count(source[j])

        errors.append((np.sum(copy_error_scores), np.sum(repeat_error_scores)))
        print("Copy error:", errors[-1][0])
        print("Repeat error:", errors[-1][0])

    return errors

def main(args):

    data, pred_scores = read_data(args)
    data = process_data(data)

    attn_errors = calc_attn_errors(data)
    """errs = np.array(attn_errors)
    errs /= errs.max(axis=0)
    pred = np.exp(np.array(pred_scores))
    pred /= pred.max(axis=0)"""
    with open(args.output, 'w') as out:
        for p_score, (c_err, r_err), d in zip(np.exp(pred_scores), attn_errors, data):
            pred = d[1][:-1]
            print("%.4f\t%.4f\t%.4f\t%s" % (p_score, c_err, r_err, ' '.join(pred)), file=out)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='debug.txt', help="")
    parser.add_argument('--output', '-o', type=str, default='scoring.txt', help="")

    args = parser.parse_args()


    main(args)
