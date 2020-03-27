import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam


def append_game_pairs(game, X, y):
    for i, g0 in enumerate(game[:-1]):
        for j, g1 in enumerate(game[i:]):
            g0_feat = (float(g0['fact']), float(g0['form']))#len(g0['text'].split()))
            g1_feat = (float(g1['fact']), float(g1['form']))#len(g1['text'].split()))
            if g0_feat > g1_feat:
                #sign = -1.
                reverse = 1.
                #elif g0_feat == g1_feat:
                #    sign = 0.
            else:
                #sign = 1.
                reverse = 0.
            #y.append(sign)
            #y.append(-sign)
            x = []
            for g in [g0, g1]:
                #x += [1. if g['type'] == t else 0. for t in ['result', 'goal', 'penalty', 'save']]
                #x.append({'short': 1., 'medium': 2., 'long': 3.}[g['length']])
                #x.append(float(g['min_length']))
                x.append(float(g['prob']))
                x.append(float(g['ambiguity']))
                x.append(float(g['repetition']))
                for feat1 in ['prob', 'ambiguity', 'repetition']:
                    for feat2 in ['prob', 'ambiguity', 'repetition']:
                        x.append(float(g[feat1])*float(g[feat2]))
            if x[len(x)//2:] == x[:len(x)//2]:
                continue
            y.append(reverse)
            X.append(x)
            y.append(1-reverse)
            X.append(x[len(x)//2:]+x[:len(x)//2])


X, y = [], []
with open("candidate_eval_.csv") as csvfile:
    game = []
    game_id = None
    for row in csv.DictReader(csvfile):
        if game_id is None:
            game_id = row['id']
        elif game_id == row['id']:
            game.append(row)
        else:
            append_game_pairs(game, X, y)
            game = [row]
            game_id = row['id']
    else:
        append_game_pairs(game, X, y)


"""        y_fact.append(float(row['fact']))
        y_form.append(float(row['form']))
        x = [1. if row['type'] == t else 0. for t in ['result', 'goal', 'penalty', 'save']]
        x.append({'short': 1., 'medium': 2., 'long': 3.}[row['length']])
        x.append(float(row['min_length']))
        x.append(float(row['prob']))
        x.append(float(row['ambiguity']))
        x.append(float(row['repetition']))
        x.append(float(len(row['text'].split())))
        X.append(x)"""

#X, y = np.array(X), to_categorical(np.array(y))
X, y = np.array(X), np.array(y)

#clf = LogisticRegression(random_state=0).fit(X, y_sign)

input = Input(shape=(X.shape[1],))
#hidden = Dense(3, activation='tanh')(input)
#output = Dense(2, activation='softmax')(input)
output = Dense(1, activation='sigmoid')(input)
model = Model(inputs=input, outputs=output)

#model.compile(optimizer=Adam(lr=0.001), loss='mean_absolute_error', metrics=['accuracy'])
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

hist = model.fit(X, y, batch_size=32, verbose=1, epochs=20)

correct = 0.
incorrect = 0.
correct_order = 0.
incorrect_order = 0.
with open("candidate_eval_.csv") as csvfile:
    game = []
    game_id = None
    for row in csv.DictReader(csvfile):
        if game_id is None:
            game_id = row['id']
        elif game_id == row['id']:
            game.append(row)
        else:
            ranking = []
            signs = {i:[] for i in range(len(game))}
            for i, g0 in enumerate(game):
                ranking.append(i)
                for j, g1 in enumerate(game):
                    if i == j:
                        continue
                    x = []
                    for g in [g0, g1]:
                        #x += [1. if g['type'] == t else 0. for t in ['result', 'goal', 'penalty', 'save']]
                        #x.append({'short': 1., 'medium': 2., 'long': 3.}[g['length']])
                        #x.append(float(g['min_length']))
                        x.append(float(g['prob']))
                        x.append(float(g['ambiguity']))
                        x.append(float(g['repetition']))
                        for feat1 in ['prob', 'ambiguity', 'repetition']:
                            for feat2 in ['prob', 'ambiguity', 'repetition']:
                                x.append(float(g[feat1])*float(g[feat2]))
                    #signs[i].append(model.predict(np.array(x).reshape(1,-1))[0][0])
                    s0_1 = model.predict(np.array(x).reshape(1,-1))[0][0]
                    s1_0 = model.predict(np.array(x[len(x)//2:]+x[:len(x)//2]).reshape(1,-1))[0][0]
                    #for s,i in sorted([(np.mean(s),i) for i,s in signs.items()]):
                    if s0_1+s1_0 <= 1.0:
                        # Correct order
                        if i<j:
                            ranking.insert(i,j)
                    else:
                        if i<j:
                            ranking.insert(i+1,j)
                    for s,g in sorted([(s0_1+s1_0,g0), (1.,g1)], key=lambda x:x[0]):
                        #print("%.3f\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%s" % (s, int(game[i]['fact']), int(game[i]['form']), int(game[i]['min_length']), float(game[i]['prob']), float(game[i]['ambiguity']), float(game[i]['repetition']), game[i]['text']))
                        pass
                        #print("%.3f\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%s" % (s, int(g['fact']), int(g['form']), int(g['min_length']), float(g['prob']), float(g['ambiguity']), float(g['repetition']), g['text']))
                    g0_feat = (float(g0['fact']), float(g0['form']))#len(g0['text'].split()))
                    g1_feat = (float(g1['fact']), float(g1['form']))#len(g1['text'].split()))
                    if g0_feat <= g1_feat:
                        correct_order += 1
                        if s0_1+s1_0 <= 1.0:
                            correct += 1
                        else:
                            incorrect += 0
                    else:
                        incorrect_order += 1
                        if s0_1+s1_0 <= 1.0:
                            incorrect += 1
                        else:
                            correct += 0
                    #print()
            print(ranking)
            game = [row]
            game_id = row['id']

print("Accuracy: %.2f" % (correct/(correct+incorrect)))
print("Baseline: %.2f" % (correct_order/(correct_order+incorrect_order)))
