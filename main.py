import numpy as np
from generate_dataset import read_input_file

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

import pdb, random
random.seed(1234)

print "Enter the name of the config file in python import notation.\n File must be in config directory. [Default config.default]"
imp = raw_input()
if not imp:
    imp = "config.default"
imported = __import__(imp, fromlist=[imp.split(".")[1]])
vectorizers = imported.vectorizers
vectorizers_params = imported.vectorizers_params
selectors = imported.selectors
selectors_params = imported.selectors_params
learners = imported.learners
learners_params = imported.learners_params


pipelines = []
params = []

count = 0
for (v_name, vectorizer), v_params in zip(vectorizers,vectorizers_params):
    for (s_name, selector), s_params in zip(selectors, selectors_params):
        for (l_name, learner), l_params in zip(learners, learners_params):
            p = [(v_name, vectorizer), (s_name, selector), (l_name, learner)]
            name = "{} {} {}".format(v_name, s_name, l_name)
            prms = dict(v_params, **dict(s_params, **l_params))
            pipelines.append((name,count,Pipeline(p)))
            params.append(prms)
            count += 1

def score_classifiers(lines, targets, selections):
    scores = []
    lines_and_targets = list(zip(lines, targets))
    random.shuffle(lines_and_targets)
    lines, targets = zip(*lines_and_targets)
    targets = np.vstack(targets)
    num_samples = len(lines)
    train_lines = lines[:8*num_samples/10]
    train_targets = targets[:8*num_samples/10].ravel()
    test_lines = lines[8*num_samples/10:]
    test_targets = targets[8*num_samples/10:].ravel()
    for select in selections:
        g = GridSearchCV(pipelines[select][2], params[select], n_jobs=1)
        g.fit(train_lines, train_targets)
        score = g.score(test_lines, test_targets)
        print str(score) + ", " + str(pipelines[select][0])
        scores.append((score, g, pipelines[select][1], params[select], pipelines[select][0]))
    #print scores
    return scores


def output_predictions(estimator, train_lines, train_targets, test_lines):
    estimator.fit(train_lines, train_targets.ravel())
    ys = estimator.predict(test_lines).astype(int).ravel().tolist()
    with open("out.csv","w") as out:
        out.write("id,prediction\n\r")
        out.write("\n\r".join([str(i)+","+str(pred) for i, pred in enumerate(ys)]))

def main():
    test_set = False
    for name, count, p in pipelines:
        print "{}. {}".format(count, name)
    print "\n\n Select a combination or a comma seperated list of combinations. 'a' will select every option"
    l = raw_input().strip()
    if l == "a":
        selections = list(range(len(pipelines)))
    else:
        selections = [int(x.strip()) for x in l.split(",")]
        if len(selections) == 1:
            print "\nEnter the location of the test set file,\nIf empty, no predictions will be output"
            print "[d=data/ml_dataset_test_in.csv]"
            file_location = raw_input().strip()
            test_set = bool(file_location)
            if file_location == "d":
                file_location = "data/ml_dataset_test_in.csv"
            if test_set:
                test_lines, _ = read_input_file(quick_n_dirty=False, file_name=file_location, test_set=test_set)
    lines, targets = read_input_file(quick_n_dirty=False)
    scores = score_classifiers(lines, targets, selections)
    scores.sort(key=lambda x: -1*x[0])
    best_combo = scores[0]
    print "Best combination is "+ str(best_combo[4])
    print "With parameters: "
    for param in best_combo[3].keys():
        print "\t" + param + ": " + str(best_combo[1].best_estimator_.get_params()[param])
    if test_set:
        output_predictions(estimator=best_combo[1].best_estimator_, train_lines=lines, train_targets=targets, test_lines=test_lines)

if __name__=="__main__":
    main()
