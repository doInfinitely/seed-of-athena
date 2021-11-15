from train import construct_dataset
from suggest import update_theta, suggest_questions, get_suggestion, load_parameters, mean
import pickle

def get_sug(theta, a, b, c, ID, correct, incorrect):
    return get_suggestion(suggest_questions(theta, a, b, c), correct[ID], incorrect[ID])[0]

def mean_squared_error(t, estimated):
    return sum([(t[key]-estimated[key])**2 for key in estimated])/len(t)

# some code to simulate running the adaptive procedure on the original dataset, not very interesting because the data is sparse,
if __name__ == '__main__':
    DELTA = 0.005
    #D, qids, ids = construct_dataset()
    D, qids, ids, cnts = pickle.load(open('dataset.p', 'rb'))
    t, a, b, c = load_parameters()
    estimated_t = []
    #print(mean(t))
    correct = dict()
    incorrect = dict()
    for iteration in range(120):
        estimated_t.append(dict())
        for ID in ids:
            if iteration == 0:
                theta = mean(t)
                correct[ID] = []
                incorrect[ID] = []
            else:
                theta = estimated_t[iteration-1][ID]
                q = get_sug(theta, a, b, c, ID, correct, incorrect)
                if D[q][ID]:
                    correct[ID].append(q)
                else:
                    incorrect[ID].append(q)
                theta = update_theta(theta, correct[ID], incorrect[ID], a, b, c)
            estimated_t[-1][ID] = theta
        print(iteration, mean_squared_error(t,estimated_t[-1]))
    print(t)
    print(estimated_t[-1])
