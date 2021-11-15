import subprocess

def call_suggest(theta, var, correct, incorrect):
    cmd = ['python', 'suggest.py', theta, var]
    if len(correct):
        cmd.append('--correct')
        cmd.extend(correct)
    if len(incorrect):
        cmd.append('--incorrect')
        cmd.extend(incorrect)
    print("CURRENT COMMAND:")
    print(' '.join(cmd))
    process = subprocess.run(cmd,stdout=subprocess.PIPE,universal_newlines=True)
    print(process.stdout.strip())
    theta = process.stdout.split(':')[1].split(',')[0].strip()
    var = process.stdout.split(':')[2].split(',')[0].strip()
    print("RESULTING THETA: ", theta, "RESULTING VARIANCE: ", var)
    print()
    return theta, var


with open('test-data-562224.csv') as f:
    theta = '0.0'
    var = '0.0'
    correct = []
    incorrect = []
    firstLine = True
    for line in f:
        splitLine = [x.strip() for x in line.split(',')]
        if firstLine:
            firstLine = False
            continue
        qid = splitLine[1]
        try:
            iscorrect = bool(int(splitLine[2]))
        except ValueError:
            continue
        temp = [incorrect, correct]
        temp[int(iscorrect)].append(qid)
        print("CURRENT QUESTION: ", qid, "GETS CORRECT: ", iscorrect)
        theta, var = call_suggest(theta, var, correct, incorrect)

