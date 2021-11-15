
param = {i:[] for i in range(10)}
header = dict()
with open('question_parameters.csv') as f:
    firstLine = True
    for line in f:
        splitLine = [x.strip() for x in line.split(',')]
        if firstLine:
            for i,x in enumerate(splitLine):
                header[x] = i
            print(header)
            firstLine = False
            continue
        print(header['model'])
        param[int(splitLine[header['model']])].append(splitLine)
for i in range(10):
    with open('question_parameters/{}.tsv'.format(i), 'w') as f:
        f.write('\t'.join(["qid","a","b","c"])+'\n')
        for x in param[i]:
            print(x[:4])
            f.write("%s\t%s\t%s\t%s\n" % tuple(x[:4]))
            
