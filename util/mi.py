import numpy as np
import math
def mi(pos, neg):
    P = np.array([[pos / 2, (1 - pos) / 2],
                  [neg / 2, (1 - neg) / 2]])
    Px = np.sum(P, axis=1)
    Py = np.sum(P, axis=0)
    res = 0.0
    for i in range(2):
        for j in range(2):
            if P[i,j] > 0 and Px[i] > 0 and Py[j] > 0:
                res += P[i,j] * math.log(P[i, j] / Px[i] / Py[j])
                return res

fns = ['bert_amazon_polarity.csv', 'bert_yelp_funny_balanced.csv'] # , 'yelp_polarity.csv', 'book_spoiler_balanced.csv']
for fn in fns:
    results = []
    mapping = {}
    for line in open('../search/log/' + fn):
        if len(line.strip().split(',')) != 4:
            continue
        token, _, pos, neg = line.strip().split(',')
        mapping[token] = (float(pos), float(neg))
        results.append((mi(float(pos), float(neg)), token))
        # break
        results.sort(reverse=True)
        print(fn)
        for score, token in results[:30]:
            print(score, token, mapping[token])
            print('-------------------------')
