import sys

candidates = []
for row in sys.stdin:
    if row:
        length, min_length, id, prob, copy_err, repeat_err, sent = row.split('\t')
        if len(candidates) == 0 or candidates[-1][1] == id:
            repeat_err = float(repeat_err)
            if repeat_err < 0.75:
                repeat_err = 0.0
            candidates.append(((repeat_err, float(copy_err), 1-float(prob)), id, sent))
        else:
            candidates.sort()
            print(candidates[0][2].strip())
            candidates = []
