import pdb

docs = ["annotations/belinda_discrete_labels.txt", "annotations/gabi_discrete_labels.txt"]

def avg(lst):
    print(str(sum(lst)) + " / " + str(len(lst)))
    return sum(lst) / len(lst)

total_times = []
discrete_times = []
discrete_times_no_zeros = []
pairwise_times = []
i = 0
for doc in docs:
    with open(doc) as f:
        for line in f:
            line = line.split("\t")
            total_times.append(float(line[2]))
            if len(line) > 3:
                discrete_times_no_zeros.append(float(line[3]))
                discrete_times.append(float(line[3]))
            else:
                discrete_times.append(0)
            pairwise_times.append(total_times[i] - discrete_times[i])
            i += 1

print("Average time to answer pairwise question: " + str(avg(pairwise_times)))
print("Average time to answer discrete question: " + str(avg(discrete_times_no_zeros)))
print("pairwise / discrete: " + str(avg(discrete_times_no_zeros) / avg(pairwise_times) + 1))
