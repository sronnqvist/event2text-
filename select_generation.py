import sys

candidates = []
for row in sys.stdin:
    if row:
        length, min_length, id, prob, copy_err, repeat_err, sent = row.split('\t')
        if len(candidates) == 0 or candidates[-1][1] == id:
            candidates.append(((float(repeat_err), float(copy_err), 1-float(prob)), id, sent))
        else:
            candidates.sort()
            print(candidates[0][2].strip())
            candidates = []
