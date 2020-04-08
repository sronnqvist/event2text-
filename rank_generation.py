import sys

def print_ranking(candidates):
    candidates.sort()
    for _,id,sent in candidates:
        print("%s\t%s" % (id, sent.strip()))

candidates = []
for row in sys.stdin:
    if row:
        length, min_length, id, prob, copy_err, repeat_err, sent = row.split('\t')
        repeat_err = float(repeat_err)
        if repeat_err < 0.75:
            repeat_err = 0.0
        if len(candidates) > 0 and candidates[-1][1] != id:
            print_ranking(candidates)
            candidates = []
        candidates.append(((float(copy_err), float(repeat_err), 1-float(prob)), id, sent))
else:
    print_ranking(candidates)
