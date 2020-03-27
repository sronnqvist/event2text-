import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam


def get_example(event):
    severity = 3-(float(event['fact'])*2+float(event['form']))#len(g0['text'].split()))
    x = []
    #x.append(float(event['min_length']))
    x = [1. if row['type'] == t else 0. for t in ['result', 'goal', 'penalty', 'save']]
    #x.append({'short': 1., 'medium': 2., 'long': 3.}[row['length']])
    x.append(float(event['prob']))
    x.append(float(event['ambiguity']))
    x.append(float(event['repetition']))
    #for feat1 in ['prob', 'ambiguity', 'repetition']:
    #    for feat2 in ['prob', 'ambiguity', 'repetition']:
    #        x.append(float(event[feat1])*float(event[feat2]))
    return x, severity


def append_examples(game, X, y):
    for i, event in enumerate(game):
        x, y_ = get_example(event)
        y.append(y_)
        X.append(x)


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
            append_examples(game, X, y)
            game = [row]
            game_id = row['id']
    else:
        append_examples(game, X, y)


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

X, y = np.array(X), to_categorical(np.array(y))
#X, y = np.array(X), np.array(y)

#clf = LogisticRegression(random_state=0).fit(X, y_sign)

fact_error_rates = []
form_error_rates = []

for exp in range(10):

    input = Input(shape=(X.shape[1],))
    #hidden = Dense(3, activation='tanh')(input)
    output = Dense(4, activation='softmax')(input)
    #output = Dense(1, activation='sigmoid')(input)
    model = Model(inputs=input, outputs=output)

    #model.compile(optimizer=Adam(lr=0.001), loss='mean_absolute_error', metrics=['accuracy'])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X, y, batch_size=1, verbose=1, epochs=20)


    top_fact_errors = 0
    top_form_errors = 0
    all_fact_errors = 0
    all_form_errors = 0
    event_count = 0

    with open("candidate_eval_.csv") as csvfile:
        game = []
        game_id = None
        for row in csv.DictReader(csvfile):
            if game_id is None:
                game_id = row['id']
            elif game_id == row['id']:
                game.append(row)
            else:
                events = []
                for i, event in enumerate(game):
                    x, _ = get_example(event)

                    severity = sum(model.predict(np.array(x).reshape(1,-1))[0]*np.array([0,1,2,3]))
                    events.append((severity, event))
                first = True
                for severity, event in sorted(events, key=lambda x:x[0]):
                    all_fact_errors += 1-int(event['fact'])
                    all_form_errors += 1-int(event['form'])
                    event_count += 1
                    if first:
                        top_fact_errors += 1-int(event['fact'])
                        top_form_errors += 1-int(event['form'])
                        first = False
                    #print(severity, event)
                    print("%.3f\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%s" % (severity, int(event['fact']), int(event['form']), int(event['min_length']), float(event['prob']), float(event['ambiguity']), float(event['repetition']), event['text']))
                    """g0_feat = (float(g0['fact']), float(g0['form']))#len(g0['text'].split()))
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
                    """
                print()
                game = [row]
                game_id = row['id']

    #print("Accuracy: %.2f" % (correct/(correct+incorrect)*100))
    #print("Baseline: %.2f" % (correct_order/(correct_order+incorrect_order)*100))

    print("Top fact error rate: %.2f (%.2f)" % (top_fact_errors/event_count, all_fact_errors/event_count))
    print("Top form error rate: %.2f (%.2f)" % (top_form_errors/event_count, all_form_errors/event_count))
    fact_error_rates.append(top_fact_errors/event_count)
    form_error_rates.append(top_form_errors/event_count)
    print()


#print("in3+type,ep20,bs1")
print("Mean top fact error rate", np.mean(fact_error_rates))
print("Mean top form error rate", np.mean(form_error_rates))
