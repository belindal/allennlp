import pdb
import time
import argparse
from getch import getch
import os

parser = argparse.ArgumentParser(description='Run setting')
parser.add_argument('policy_type',
                    type=str,
                    default='pairwise',
                    help='how to poll the user (pairwise or discrete)')
parser.add_argument('input_document',
                    type=str,
                    help='input document containing examples to label')

args = parser.parse_args()
QUERY_TYPE = vars(args)['policy_type']
doc_name = vars(args)['input_document']

if QUERY_TYPE != "pairwise" and QUERY_TYPE != "discrete":
    raise ValueError("bad argument, should pass in either 'pairwise' or 'discrete'")

user_doc_name = "user_" + doc_name[:len(doc_name) - 13] + "_labels.txt"
examples = []
user_answers = []
new_ants = {}
time_per_example = []

with open(doc_name, 'r', encoding='unicode_escape') as f:
    for line in f:
        examples.append(line)

try:
    i = 0
    with open(user_doc_name, 'r') as wf:
        for line in wf:
            user_answers.append(bool(line.strip().split('\t')[0] == 'True'))
            if len(line.strip().split('\t')) >= 3:
                if line.strip().split('\t')[0] != 'True':
                    new_ants[i] = str(line.strip().split('\t')[1])
                time_per_example.append(float(line.strip().split('\t')[2]))
            i += 1
    num_already_queried = len(user_answers)
except:
    num_already_queried = 0

i = num_already_queried
try:
    while i < len(examples):
        os.system("clear")
        print(examples[i].strip())
        start_time = time.time()
        print("Are these two coreferent? y/[n] ('q' to quit with save, 'p' to go back to previous example): ")
        val = getch()
        if val.startswith('y') or val.startswith('Y'):
            if i >= len(user_answers):
                user_answers.append(True)
            else:
                user_answers[i] = True
        elif val.startswith('q') or val.startswith('Q'):
            break
        elif val.startswith('p') or val.startswith('P'):
            i -= 2
            if i < 0:
                i = -1
        else:
            if i >= len(user_answers):
                user_answers.append(False)
            else:
                user_answers[i] = False
            if QUERY_TYPE == "discrete":
                new_item = input("What is the *first* appearance of the entity that the white-highlighted text refers to? (copy from document): ")
                new_ants[i] = new_item
        end_time = time.time()
        if i >= len(time_per_example):
            time_per_example.append(end_time - start_time)
        else:
            time_per_example[i] += (end_time - start_time)
        print()
        i += 1
except:
    # do nothing
    print()

wf = open(user_doc_name, 'w')
for i, answer in enumerate(user_answers):
    wf.write(str(answer) + "\t")
    if i in new_ants:
        wf.write(new_ants[i])
    wf.write("\t" + str(time_per_example[i]))
    wf.write("\n")
wf.close()

