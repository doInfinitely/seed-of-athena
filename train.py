import sqlite3
from sqlite3 import OperationalError
import matplotlib.pyplot as plt
import tqdm
import re
import random
import math
import numpy as np
import time
import math
import pickle
from numba import cuda

DELTA = 0.001
EPOCHS_PER_CHECKPOINT = 1000
MAX_THREADS_PER_BLOCK = 1024
CONTINUOUS_BENCHMARKING = False
PLOT_AFTER_CHECKPOINT = False
MODE="numba"#"default"
ENSEMBLE_SIZE = 10

# load training data from database
# since responses of individual examinees were not stored (just answer percentages, we construct a dataset consistent with the percentages)
def construct_dataset(num_testers=0):
    filename = 'answers.tsv'
    with open(filename) as f:
        header = None
        dataset = dict()
        question_ids = set()
        answer_counts = dict()
        ids = {i for i in range(num_testers)}
        if not num_testers:
            ids = {0}
        for line in f:
            if header == None:
                header = {x:i for i,x in enumerate(line.strip().split('\t'))}
                print(header)
            else:
                splitline = line.strip().split('\t')
                #ID = int(splitline[header['id']])
                question_id = int(splitline[header['question_id']])
                if question_id not in answer_counts:
                    answer_counts[question_id] = 0
                answer_counts[question_id] += 1
                is_correct = bool(int(splitline[header['is_correct']]))
                try:
                    percent = float(splitline[header['percentage_of_answer']])
                except ValueError:
                    pass
                if is_correct:
                    question_ids.add(question_id)
                    dataset[question_id] = dict()
                    if not num_testers:
                        dataset[question_id][0] = percent*0.01
                    else:
                        for ID in range(num_testers):
                            dataset[question_id][ID] = int(random.random() <= percent*0.01)
        return dataset, question_ids, ids, answer_counts

# Here we replace a with e^a to confine the term to non-negative numbers
def f(x):
    try:
        return 1.0/(1+math.exp(-x))
    except OverflowError:
        return 0
def g(t, a, b):
    return math.exp(a)*(t-b)
def p(t, a, b, c):
    return c + (1-c)*f(g(t, a, b))
def partials(t, a, b, c, Dij):
    intermediate = f(g(t,a,b))*f(-g(t,a,b))
    dLdt = (Dij-(1-Dij))*(1-c)*intermediate*math.exp(a)
    dLdb = -dLdt
    dLda = -(Dij-(1-Dij))*(1-c)*intermediate*(t-b)*math.exp(a)
    dLdc = Dij*(1-f(g(t,a,b)))-(1-Dij)*(1-f(g(t,a,b)))
    return dLdt, np.array([dLda, dLdb, dLdc])

def log_likelihood(D, t, a, b, c):
    L = 0
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            probability = p(t[j], a[i], b[i], c[i])
            try:
                L += math.log(D[i][j]*probability+(1-D[i][j])*(1-probability))
            except ValueError:
                pass
    return L

@cuda.jit
def log_likelihood_numba(D, t, a, b, c, log_like):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % D.shape[0]
    j = (tx+ty*bw) // D.shape[0]
    if j < D.shape[1]:
        probability = c[i] + (1-c[i])*1.0/(1+math.exp(-(math.exp(a[i])*(t[j]-b[i]))))
        cuda.atomic.add(log_like, 0, math.log(D[i][j]*probability+(1-D[i][j])*(1-probability)))

def backward(D, t, a, b, c, qids, ids):
    pt = [0 for x in ids]
    pq = [np.array([0.0,0.0,0.0]) for x in qids]
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            pj, pi = partials(t[j], a[i], b[i], c[i], D[i][j])
            pt[j] += pj
            pq[i] += pi
    return pt, pq

@cuda.jit
def backward_numba(D, t, a, b, c, pt, pq):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % D.shape[0]
    j = (tx+ty*bw) // D.shape[0]
    if j < D.shape[1]:
        intermediate = 1.0/(1+math.exp(math.exp(a[i])*(t[j]-b[i])))*1.0/(1+math.exp(-math.exp(a[i])*(t[j]-b[i])))
        dLdt = (D[i,j]-(1-D[i,j]))*(1-c[i])*intermediate*math.exp(a[i])
        dLdb = -dLdt
        dLda = -(D[i,j]-(1-D[i,j]))*(1-c[i])*intermediate*(t[j]-b[i])*math.exp(a[i])
        dLdc = D[i,j]*(1-1.0/(1+math.exp(math.exp(a[i])*(t[j]-b[i]))))-(1-D[i,j])*(1-1.0/(1+math.exp(math.exp(a[i])*(t[j]-b[i]))))
        cuda.atomic.add(pt, j, dLdt)
        cuda.atomic.add(pq, (i,0), dLda)
        cuda.atomic.add(pq, (i,1), dLdb)
        cuda.atomic.add(pq, (i,2), dLdc)

def gradient(delta, t, a, b, c, pt, pq, qids, ids):
    for x in range(len(qids)):
        a[x] += delta*pq[x][0]
        b[x] += delta*pq[x][1]
        #c[x] += delta*pq[x][2]
        #c[x] = max(0, c[x])
        #c[x] = min(1, c[x])
        pass
    #for x in range(len(ids)):
    #    t[x] += delta*pt[x]

def writeout(t, a, b, c, qids, ids, i):
    with open('tester_parameters/{}.tsv'.format(i), 'w') as f:
        f.write('\t'.join(['id', 'theta'])+'\n')
        for x in ids:
            f.write('\t'.join([str(x), str(t[x]) if x in t else ''])+'\n')
    with open('question_parameters/{}.tsv'.format(i), 'w') as f:
        f.write('\t'.join(['qid', 'a', 'b', 'c'])+'\n')
        for x in qids:
            f.write('\t'.join([str(x), str(a[x]) if x in a else '', str(b[x]) if x in b else '', str(c[x]) if x in c else ''])+'\n')

def load_parameters(i):
    t = dict()
    with open('tester_parameters/{}.tsv'.format(i)) as f:
        firstline = True
        for line in f:
            if firstline:
                firstline = False
                continue
            splitline = line.strip().split('\t')
            t[int(splitline[0])] = float(splitline[1])
    a = dict()
    b = dict()
    c = dict()
    with open('question_parameters/{}.tsv'.format(i)) as f:
        firstline = True
        for line in f:
            if firstline:
                firstline = False
                continue
            splitline = line.strip().split('\t')
            a[int(splitline[0])] = float(splitline[1])
            b[int(splitline[0])] = float(splitline[2])
            c[int(splitline[0])] = float(splitline[3])
    return t, a, b, c

def toNumpy(D, t, a, b, c, qids, ids):
    matrix = np.zeros((len(qids), len(ids)))
    for i,x in enumerate(qids):
        for j,y in enumerate(ids):
            matrix[i,j] = D[x][y]
    return matrix, np.array([t[x] for x in ids]), np.array([a[x] for x in qids]
            ), np.array([b[x] for x in qids]), np.array([c[x] for x in qids])


def toDictionary(D, t, a, b, c, qids, ids):
    dictionary = dict()
    for i,x in enumerate(qids):
        if x not in dictionary:
            dictionary[x] = dict()
        for j,y in enumerate(ids):
            dictionary[x][y] = D[i,j]
    T = {x:t[i] for i,x in enumerate(ids)}
    A = {x:a[i] for i,x in enumerate(qids)}
    B = {x:b[i] for i,x in enumerate(qids)}
    C = {x:c[i] for i,x in enumerate(qids)}
    return dictionary, T, A, B, C

def train(mode=None):
    try:
        i,j = pickle.load(open('checkpoint.p', 'rb'))
        D, qids, ids, cnts = pickle.load(open('dataset.p', 'rb'))
        t, a, b, c = load_parameters(i)
    except IOError:
        D, qids, ids, cnts = construct_dataset()
        pickle.dump((D, qids, ids, cnts), open('dataset.p', 'wb'))
        #t = {x:np.random.normal() for x in ids}
        t = {x:0 for x in ids}
        a = {x:np.random.normal() for x in qids}
        b = {x:np.random.normal() for x in qids}
        c = {x:1/cnts[x] for x in qids}
        i = 0
        j = 0
    #if mode == 'numba' or mode == 'pytorch':
    D, t, a, b, c = toNumpy(D, t, a, b, c, qids, ids)
    while i < ENSEMBLE_SIZE:   
        while True:
            likelihoods = []
            print("Model: ", i, " Checkpoint count: ", j)
            bar = tqdm.trange(EPOCHS_PER_CHECKPOINT, leave=True)
            count = 0
            for x in bar:
                if CONTINUOUS_BENCHMARKING or count == 0 or count == EPOCHS_PER_CHECKPOINT-1:
                    if False and mode == "numba":
                        # this calculation requires 64 bit precision and numba uses 32-bit, so blocked off
                        log_like = np.zeros((1))
                        threadsperblock = min(MAX_THREADS_PER_BLOCK, D.shape[0])
                        blockspergrid = math.ceil(D.shape[0]*D.shape[1]/threadsperblock)
                        log_likelihood_numba[blockspergrid, threadsperblock](D, t, a, b, c, log_like)
                        log_like = log_like[0]
                    else:
                        log_like = log_likelihood(D, t, a, b, c)
                        print(log_like)
                    likelihoods.append(log_like)
                    if math.isnan(log_like):
                        break
                if mode == "numba":
                    pt = np.zeros((len(ids)))
                    pq = np.zeros((len(qids), 3)) 
                    threadsperblock = min(MAX_THREADS_PER_BLOCK, D.shape[0])
                    blockspergrid = math.ceil(D.shape[0]*D.shape[1]/threadsperblock)
                    backward_numba[blockspergrid, threadsperblock](D, t, a, b, c, pt, pq)
                else:
                    pt, pq = backward(D, t, a, b, c, qids, ids)
                gradient(DELTA, t, a, b, c, pt, pq, qids, ids)
                count += 1
            j += 1
            if PLOT_AFTER_CHECKPOINT:
                plt.plot(likelihoods)
                plt.show(block=False)
                plt.pause(5)
                plt.close()
            # terminate training when the model fails to improve between checkpoints
            if likelihoods[0] > likelihoods[-1]:
                break
            # write out parameters after each checkpoint
            DTemp, tTemp, aTemp, bTemp, cTemp = toDictionary(D, t, a, b, c, qids, ids)
            writeout(tTemp,aTemp,bTemp,cTemp,qids,ids, i)
            pickle.dump((i,j), open('checkpoint.p', 'wb'))
        j = 0
        i += 1
        D, qids, ids, cnts = construct_dataset()
        t = {x:0 for x in ids}
        a = {x:np.random.normal() for x in qids}
        b = {x:np.random.normal() for x in qids}
        c = {x:1/cnts[x] for x in qids}
        D, t, a, b, c = toNumpy(D, t, a, b, c, qids, ids)

if __name__ == '__main__':
    train(mode='numba')
