import sys
# arguments: mean and variance

mean = float(sys.argv[1])
variance = float(sys.argv[2])
z_table = {99:2.576, 98: 2.326, 95: 1.96, 90:1.645}

for x in z_table:
    radius = variance**0.5*z_table[x]
    print("{}% confidence band".format(x), (mean-radius,mean+radius))
