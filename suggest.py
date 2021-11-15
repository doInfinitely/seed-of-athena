from train import f, g, p, partials, load_parameters
import argparse
import pickle
import os
import os.path

# These hyper-parameters determine how fast the student theta changes
DELTA = 0.5
STEPS = 1

# Suggestion probability, This hyperparameter controls the difficulty of the questions
#PROB = 0.5 # Tries to give questions the model predicts the student has a 50% of getting correct, i.e, the maximum information gain criterion.
PROB = 0.8

def load_parameters():
    t = []
    for root, dirs, files in os.walk("tester_parameters"):
        for filename in files:
            with open(os.path.join(root, filename)) as f:
                t_temp = dict()
                firstline = True
                for line in f:
                    if firstline:
                        firstline = False
                        continue
                    splitline = line.strip().split('\t')
                    t_temp[int(splitline[0])] = float(splitline[1])
            t.append(t_temp)
    a = []
    b = []
    c = []
    for root, dirs, files in os.walk("question_parameters"):
        for filename in files:
            with open(os.path.join(root, filename)) as f:
                a_temp = dict()
                b_temp = dict()
                c_temp = dict()
                firstline = True
                for line in f:
                    if firstline:
                        firstline = False
                        continue
                    splitline = line.strip().split('\t')
                    a_temp[int(splitline[0])] = float(splitline[1])
                    b_temp[int(splitline[0])] = float(splitline[2])
                    c_temp[int(splitline[0])] = float(splitline[3])
            a.append(a_temp) 
            b.append(b_temp)
            c.append(c_temp)
    return t, a, b, c

def mean(t):
    return sum([t[x] for x in t])/len(t)

def performance_probability(theta, correct_qids, incorrect_qids, a, b, c):
    probability = [1 for x in a]
    for i in range(len(probability)):
        for x in correct_qids:
            probability[i] *= p(theta, a[i][x], b[i][x], c[i][x])
        for x in incorrect_qids:
            probability[i] *= (1-p(theta, a[i][x], b[i][x], c[i][x]))
    return mean(probability)

def backwards(theta, correct_qids, incorrect_qids, a, b, c):
    pt = 0
    var = 0
    for x in correct_qids:
        temp = sum([partials(theta, a[i][x], b[i][x], c[i][x], 1)[0] for i in range(len(a))])/len(a)
        var += sum([(partials(theta, a[i][x], b[i][x], c[i][x], 1)[0]-temp)**2 for i in range(len(a))])/len(a)
        pt += temp
    for x in incorrect_qids:
        temp = sum([partials(theta, a[i][x], b[i][x], c[i][x], 0)[0] for i in range(len(a))])/len(a)
        #pt += sum([partials(theta, a[i][x], b[i][x], c[i][x], 0)[0] for i in range(len(a))])/len(a)
        var += sum([(partials(theta, a[i][x], b[i][x], c[i][x], 0)[0]-temp)**2 for i in range(len(a))])/len(a)
        pt += temp
    return pt, var

def update_theta(theta, variance, correct_qids, incorrect_qids, a, b, c):
    for i in range(STEPS):
        temp1, temp2 = backwards(theta, correct_qids, incorrect_qids, a, b, c)
        theta += DELTA*temp1
        variance += DELTA*temp2
    return theta, variance

def get_suggestion(suggestions, correct_qids, incorrect_qids, num=1):
    output = []
    for x in suggestions:
        if x not in correct_qids and x not in incorrect_qids:
            output.append(x)
        if len(output) >= num:
            break
    return output

def suggest_questions(theta, a, b, c):
    return [y[1] for y in sorted([(abs(sum([p(theta, a[i][x], b[i][x], c[i][x]) for i in range(len(a))])/len(a)-PROB),x) for x in a[0]])]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes in an initial theta and correct and incorrect qids and returns an updated theta')
    parser.add_argument('initial_theta', metavar='T', type=float, nargs='?', help='inital theta')
    parser.add_argument('initial_variance', metavar='V', type=float, nargs='?', help='inital variance')
    parser.add_argument('--correct', metavar='C', type=int, nargs='*', help='ids of correctly answered questions')
    parser.add_argument('--incorrect', metavar='I', type=int, nargs='*', help='ids of incorrectly answered questions')
    parser.add_argument('--number', metavar='N', type=int, default=1, nargs='?', help='number of suggestions to output')
    args = parser.parse_args()
    t, a, b, c = load_parameters()
    if args.initial_theta == None:
        theta = sum([mean(x) for x in t])/len(t)
        variance = sum([(mean(x)-theta)**2 for x in t])/len(t)
    else:
        theta = args.initial_theta
        variance = args.initial_variance
    correct = args.correct
    if args.correct == None:
        correct = []
    incorrect = args.incorrect
    if args.incorrect == None:
        incorrect = []
    #print(performance_probability(-6, [2552], [], a, b, c))
    #print(performance_probability(10, [2552], [], a, b, c))
    theta, variance = update_theta(theta, variance, correct, incorrect, a, b, c)
    print('theta: ', theta, ', variance: ', variance, ', suggestion: ', get_suggestion(suggest_questions(theta, a, b, c), correct, incorrect, args.number))
