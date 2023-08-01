import os
import sys
import math
import time
from numpy import exp
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score

start = time.perf_counter()
test_file = sys.argv[1]
boundary_file = sys.argv[2]
model_file = sys.argv[3]
sys_output = sys.argv[4]

# max gap between lgprob of the best path and lgprob of the kept path
beam_size = int(sys.argv[5])

# the number of POS tags chosen for the given word
topN = int(sys.argv[6])

# the max number of paths kept alive at each position after pruning
topK = int(sys.argv[7])

b1 = open(os.path.join(os.path.dirname(__file__), boundary_file), 'r').read().split("\n")[:-1]
model_formatted = open(os.path.join(os.path.dirname(__file__), model_file), 'r').read().split("\n")[:-1]
test_data = open(os.path.join(os.path.dirname(__file__), test_file), 'r').read().split("\n")

boundaries = []
for line in b1:
    boundaries.append(int(line))

# MODEL FORMATTING

# creating hashable representation of the model
defaults = {}
ranges = []
tagset = set()
twotagset = set()
label_set = []

# figure out the boundaries in the model file, and create tagsets
for i in range(len(model_formatted)):
    if model_formatted[i].startswith("FEATURES FOR CLASS"):
        label = model_formatted[i].split()[-1]
        default = float(model_formatted[i + 1].split()[-1])
        defaults[label] = default
        ranges.append((label, i + 2))
        label_set.append(label)
    if model_formatted[i].startswith(" prevT="):
        tag = model_formatted[i].split()[0]
        tagset.add(tag)
    if model_formatted[i].startswith(" prevTwoTags="):
        twotag = model_formatted[i].split()[0]
        twotagset.add(twotag)


# extracts a dictionary of features:values/weights for class label as defined by a range in the model file
def model_tracker(range1, range2):
    class_label = defaultdict(float)
    for i in range(range1, range2):
        pair = model_formatted[i].split()
        feat = pair[0]
        weight = float(pair[1])
        class_label[feat] = weight
    return class_label


# compiles all those dictionaries together into a dictionary of dictionaries
weights = {}
for i in range(len(ranges)):
    label, index_start = ranges[i]
    if i == len(ranges) - 1:
        index_end = len(model_formatted)
    else:
        index_end = ranges[i + 1][1] - 2
    weights[label] = model_tracker(index_start, index_end)


# BEAM SEARCH

# tree class to implement beam search with; bp points to parent
class Tree:
    def __init__(self, label, prob, seqprob, bp, gold, instance):
        self.label = label
        self.prob = prob
        self.seqprob = seqprob
        self.bp = bp
        self.gold = gold
        self.instance = instance

    def __repr__(self):
        return str((self.instance, self.gold, self.label, self.prob, self.seqprob))


# returns the history vector for each word
def historizer(test_line):
    divided = test_line.split()
    instance, gold_pos = divided[:2]
    # these may vary according to each instance
    history = list(filter('1'.__ne__, divided[2:]))
    return instance, gold_pos, history


# find the topN most likely tags for a particular history set
def topN_finder(history):
    distribution = Counter()
    for label in weights:
        distribution[label] = 0.0

    Z = 0
    for label in distribution:
        lbd = defaults[label]
        sum = 0

        for feature in history:
            sum += weights[label][feature]

        # cast numerator to label dictionary
        numerator = exp(lbd + sum)
        distribution[label] = numerator
        Z += numerator

    # divide by Z to get probabilities
    for label in distribution:
        distribution[label] = (distribution[label] / Z)

    return distribution.most_common(topN)


# prunes the tree
def pruner(tree, position):
    # each node at position i stores a tag for w_i and a probability for the sequence so far
    children = tree[position]

    # find max_prob
    max_prob = children[0].seqprob
    for node in children:
        if node.seqprob > max_prob:
            max_prob = node.seqprob

    # for each node at position i, keep node if it's topK and meets inequality constraint
    kept_children = []
    for child in sorted(children, key=lambda x: x.seqprob, reverse=True)[:topK]:
        if child.prob + beam_size >= max_prob:
            kept_children.append(child)

    return kept_children

# takes a sentence and a sentence boundary and returns a tree meeting the topK and topN parameters
def beam(sentence, boundary):
    tree = {0: [Tree("BOS", 0.0, 0.0, None, "BOS", "Root")]}
    tree[1] = []
    w1_inst, w1_gold, w1_history = historizer(sentence[0])
    w1_history.append("prevT=BOS")
    w1_history.append("prevTwoTags=BOS+BOS")

    # get topN tags for w_1 and form nodes s_1,j
    for w_1candidate, w_1prob in topN_finder(w1_history):
        child = Tree(w_1candidate, math.log(w_1prob, 10), math.log(w_1prob, 10), tree[0][0], w1_gold, w1_inst)
        tree[1].append(child)

    # iterate through remaining words
    for i in range(2, boundary + 1):
        # prune the tree
        tree[i - 1] = pruner(tree, i - 1)
        tree[i] = []
        # for each surviving node
        for parent_node in tree[i - 1]:
            # form history vector for current word
            inst, gold, history = historizer(sentence[i - 1])
            prevT = "prevT=" + parent_node.label

            if i == 2:
                prevTwoTags = "prevTwoTags=BOS+" + parent_node.label
            else:
                prevTwoTags = "prevTwoTags=" + parent_node.bp.label + "+" + parent_node.label

            # add tag features before calculating topN POS
            if prevT in tagset:
                history.append(prevT)

            if prevTwoTags in twotagset:
                history.append(prevTwoTags)

            for candidate, prob in topN_finder(history):
                prob = math.log(prob, 10)
                seqprob = prob + parent_node.seqprob
                child = Tree(candidate, prob, seqprob, parent_node, gold, inst)
                tree[i].append(child)
    return tree


# build the tree path up
def backtrace(node, pred_labels, inst_list, gold_list, word_prob):
    # base case
    if node.bp is None:
        return

    # grab onto all the things we care about
    pred_labels.append(node.label)
    inst_list.append(node.instance)
    gold_list.append(node.gold)
    word_prob.append(10 ** node.prob)

    # recursive call
    backtrace(node.bp, pred_labels, inst_list, gold_list, word_prob)

    return list(zip(list(reversed(inst_list)), list(reversed(gold_list)),
                    list(reversed(pred_labels)), list(reversed(word_prob))))


accuracy_scores = []
# write it all to file
with open(sys_output, 'w') as d:
    d.write("%%%%% test data:\n")
    for i in range(len(boundaries)):

        # segment each sentence out
        boundary = boundaries[i]
        if i == 0:
            last_boundary = 0
        else:
            last_boundary = boundaries[i - 1] + last_boundary

        sentence = test_data[last_boundary:last_boundary+boundary]

        # keep track of labels for accuracy scores
        true_y = []
        pred_y = []

        max_prob = 0
        best_path = None
        for end_node in beam(sentence, boundary)[boundary]:
            if (10**end_node.seqprob) > max_prob:
                max_prob = 10**end_node.seqprob
                best_path = end_node

        for line in (backtrace(best_path, [], [], [], [])):
            d.write(line[0] + " " + line[1] + " " + line[2] + " " + str(line[3]) + "\n")
            true_y.append(line[1])
            pred_y.append(line[2])

        accuracy = accuracy_score(true_y, pred_y)
        accuracy_scores.append(accuracy)

print(sum(accuracy_scores)/len(accuracy_scores))
end = time.perf_counter()
print(end-start)