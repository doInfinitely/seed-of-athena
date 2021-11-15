import argparse
from suggest import load_parameters

# prints finds the cut value
def find_theta(t, percentile):
    thetas = sorted([t[key] for key in t])
    return thetas[round(len(thetas)*percentile/100)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes in an initial theta and correct and incorrect qids and returns an updated theta')
    parser.add_argument('percentile', metavar='P', type=float, nargs=1, help='The percetile you wish to correspond with a theta')
    args = parser.parse_args()
    t = # this needs to be a list of observsed theta values
    print(find_theta(t, args.percentile[0]))
