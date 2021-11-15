import pickle
import matplotlib.pyplot as plt
from suggest import load_parameters

FAILURE_RATE = 0.15

def find_tester_cutoff(t, fail_rate):
    thetas = sorted([t[key] for key in t])
    return thetas[round(len(thetas)*fail_rate)]
    
D, qids, ids, cnts = pickle.load(open('dataset.p', 'rb'))
t, a, b, c = load_parameters()

print(find_tester_cutoff(t, FAILURE_RATE))
#plt.hist([t[key] for key in t])
#plt.show()


