import sys
import random
import warnings
import numpy as np
import networkx as nx
from typing import NamedTuple
from datetime import datetime
from enum import Enum


class ValueType(Enum):
    unknown = -1
    false = 0
    true = 1


class VariableType(Enum):
    unknown = -1
    query = 0
    evidence = 1


class Node(NamedTuple):
    node: int
    numParents: int
    parents: str
    probabilities: str
    type: VariableType
    value: ValueType


def AorB(networkFile):
    data = np.loadtxt(networkFile, delimiter='\n', dtype='str')
    networkData = []
    for element in data:
        element = element.split()
        networkData.append(element)

    for line in networkData:
        section = 0
        numParents = 0
        for element in line:
            element = element.replace(':', '')
            element = element.replace('[', '')
            element = element.replace(']', '')
            if section == 0:
                section += 1
            elif section == 1:
                if element[:1] == 'n':
                    numParents += 1
        if numParents > 2:
            return 'B'
    return 'A'


def constructNetwork(networkFile, choice):
    data = np.loadtxt(networkFile, delimiter='\n', dtype='str')
    networkData = []
    for element in data:
        element = element.split()
        networkData.append(element)

    DG = nx.DiGraph()
    nodes = {}
    for line in networkData:
        counter = 0
        section = 0
        node = 0
        numParents = 0
        parents = []
        probabilities = []
        probabilityRow = []
        for element in line:
            element = element.replace(':', '')
            element = element.replace('[', '')
            element = element.replace(']', '')
            if section == 0:
                node = element[-1:]
                section += 1
            elif section == 1:
                if element[:1] == 'n':
                    numParents += 1
                    parents.append(element[-1:])
                elif not element:
                    section += 1
                else:
                    if choice == 'A':
                        section += 1
                        counter += 1
                        probabilityRow.append(float(element))
                    else:
                        if element == '0.7':
                            probabilityRow.append(0.3)
                            probabilityRow.append(0.7)
                        elif element == '0.8':
                            probabilityRow.append(0.2)
                            probabilityRow.append(0.8)
                        elif element == '0.9':
                            probabilityRow.append(0.1)
                            probabilityRow.append(0.9)
                        else:
                            probabilityRow.append(1-(float(element)))
                            probabilityRow.append(float(element))
                        probabilities.append(probabilityRow)
                        probabilityRow = []
                        section += 1
            else:
                if choice == 'A':
                    probabilityRow.append(float(element))
                    if counter == 0:
                        counter += 1
                    else:
                        probabilities.append(probabilityRow)
                        probabilityRow = []
                        counter = 0
                else:
                    if element == '0.7':
                        probabilityRow.append(0.3)
                        probabilityRow.append(0.7)
                    elif element == '0.8':
                        probabilityRow.append(0.2)
                        probabilityRow.append(0.8)
                    elif element == '0.9':
                        probabilityRow.append(0.1)
                        probabilityRow.append(0.9)
                    else:
                        probabilityRow.append(1 - (float(element)))
                        probabilityRow.append(float(element))
                    probabilities.append(probabilityRow)
                    probabilityRow = []

        nodeData = Node(node, numParents, parents, probabilities, None, None)
        nodes[node] = (nodeData)
        DG.add_node(node, data=nodeData)

    for i in range(len(nodes)):
        node = nodes[str(i+1)]
        if node.parents:
            for parent in node.parents:
                DG.add_edge(parent, node.node)

    return nodes, DG


def readQuery(queryFile, nodes, DG):
    data = np.loadtxt(queryFile, delimiter=',', dtype='str')
    i = 0
    for state in data:
        i += 1
        node = nodes[str(i)]
        if state == '?':
            type = VariableType.query
            value = ValueType.unknown
        elif state == '-':
            type = VariableType.unknown
            value = ValueType.unknown
        else:
            type = VariableType.evidence
            if state == 't':
                value = ValueType.true
            else:
                value = ValueType.false
        nodeData = Node(node.node, node.numParents, node.parents, node.probabilities, type, value)
        nodes[str(i)] = (nodeData)
        DG.node[str(i)]['data'] = nodeData


def recurseParents(DG, i, path):
    path.append(i)
    node = DG.node[str(i)]['data']
    if node.numParents == 0:
        return
    else:
        for i in range(node.numParents):
            recurseParents(DG, node.parents[i], path)


def samplePath(DG, sample, probabilityRow, path, i, counter):
    key = path[i]
    if str(key) not in sample.keys():
        node = DG.node[str(key)]['data']
        if node.numParents == 0:
            if probabilityRow[counter] <= node.probabilities[0][0]:
                sample[node.node] = ValueType.false
            else:
                sample[node.node] = ValueType.true
        elif node.numParents == 1:
            if sample[str(node.parents[0])] == ValueType.true:
                if probabilityRow[counter] <= node.probabilities[1][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            else:
                if probabilityRow[counter] <= node.probabilities[0][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
        elif node.numParents == 2:
            if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true:
                if probabilityRow[counter] <= node.probabilities[3][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            elif sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false:
                if probabilityRow[counter] <= node.probabilities[2][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            elif sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true:
                if probabilityRow[counter] <= node.probabilities[1][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            else:
                if probabilityRow[counter] <= node.probabilities[0][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
        elif node.numParents == 3:
            if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.true:
                if probabilityRow[counter] <= node.probabilities[7][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.false:
                if probabilityRow[counter] <= node.probabilities[6][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.true:
                if probabilityRow[counter] <= node.probabilities[5][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.false:
                if probabilityRow[counter] <= node.probabilities[4][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.true:
                if probabilityRow[counter] <= node.probabilities[3][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.false:
                if probabilityRow[counter] <= node.probabilities[2][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.true:
                if probabilityRow[counter] <= node.probabilities[1][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            else:
                if probabilityRow[counter] <= node.probabilities[0][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
        counter += 1
    return counter


def rejectionSampling(DG, numSamples):
    random.seed(datetime.now())
    probabilities = []
    for i in range(numSamples):
        probabilityRow = []
        for j in range(8):
            probabilityRow.append(random.uniform(0, 1))
        probabilities.append(probabilityRow)

    paths = []
    for i in DG.nodes:
        path = []
        recurseParents(DG, i, path)
        path = path[::-1]
        paths.append(path)

    samples = []
    for probabilityRow in probabilities:
        counter = 0
        sample = {}
        for path in paths:
            for i in range(len(path)):
                counter = samplePath(DG, sample, probabilityRow, path, i, counter)
        samples.append(sample)

    consistentSamples = []
    numConsistentSamples = 0
    for sample in samples:
        isConsistent = True
        for i in DG.nodes:
            node = DG.node[i]['data']
            if node.type == VariableType.evidence:
                if node.value != sample[node.node]:
                    isConsistent = False
        if isConsistent:
            consistentSamples.append(sample)
            numConsistentSamples += 1

    samplesQueryTrue = 0
    for sample in consistentSamples:
        for i in DG.nodes:
            node = DG.node[i]['data']
            if node.type == VariableType.query:
                if sample[node.node] == ValueType.true:
                    samplesQueryTrue += 1

    print('Rejection Sampling:')
    if numConsistentSamples > 0:
        print('Probability of query: %f' % (samplesQueryTrue/numConsistentSamples))
    else:
        print('Probability of query: 0')


def weightedSamplePath(DG, sample, probabilityRow, path, i, counter):
    key = path[i]
    weight = 1
    if str(key) not in sample.keys():
        node = DG.node[str(key)]['data']
        if node.type == VariableType.evidence:
            sample[node.node] = node.value
            if node.value == ValueType.false:
                column = 0
            else:
                column = 1

            if node.numParents == 0:
                row = 0
            elif node.numParents == 1:
                if sample[str(node.parents[0])] == ValueType.true:
                    row = 1
                else:
                    row = 0
            elif node.numParents == 2:
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true:
                    row = 3
                elif sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false:
                    row = 2
                elif sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true:
                    row = 1
                else:
                    row = 0
            elif node.numParents == 3:
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.true:
                    row = 7
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.false:
                    row = 6
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.true:
                    row = 5
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.false:
                    row = 4
                if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.true:
                    row = 3
                if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.false:
                    row = 2
                if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.true:
                    row = 1
                else:
                    row = 0
            weight = node.probabilities[row][column]
        else:
            if node.numParents == 0:
                if probabilityRow[counter] <= node.probabilities[0][0]:
                    sample[node.node] = ValueType.false
                else:
                    sample[node.node] = ValueType.true
            elif node.numParents == 1:
                if sample[str(node.parents[0])] == ValueType.true:
                    if probabilityRow[counter] <= node.probabilities[1][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                else:
                    if probabilityRow[counter] <= node.probabilities[0][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
            elif node.numParents == 2:
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true:
                    if probabilityRow[counter] <= node.probabilities[3][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                elif sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false:
                    if probabilityRow[counter] <= node.probabilities[2][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                elif sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true:
                    if probabilityRow[counter] <= node.probabilities[1][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                else:
                    if probabilityRow[counter] <= node.probabilities[0][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
            elif node.numParents == 3:
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.true:
                    if probabilityRow[counter] <= node.probabilities[7][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.false:
                    if probabilityRow[counter] <= node.probabilities[6][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.true:
                    if probabilityRow[counter] <= node.probabilities[5][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                if sample[str(node.parents[0])] == ValueType.true and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.false:
                    if probabilityRow[counter] <= node.probabilities[4][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.true:
                    if probabilityRow[counter] <= node.probabilities[3][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.true and sample[str(node.parents[2])] == ValueType.false:
                    if probabilityRow[counter] <= node.probabilities[2][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                if sample[str(node.parents[0])] == ValueType.false and sample[str(node.parents[1])] == ValueType.false and sample[str(node.parents[2])] == ValueType.true:
                    if probabilityRow[counter] <= node.probabilities[1][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
                else:
                    if probabilityRow[counter] <= node.probabilities[0][0]:
                        sample[node.node] = ValueType.false
                    else:
                        sample[node.node] = ValueType.true
            counter += 1
    return counter, weight


def likelihoodWeightingSampling(DG, numSamples):
    random.seed(datetime.now())
    numNonevidence = 0
    for i in DG.nodes:
        node = DG.node[i]['data']
        if node.type != VariableType.evidence:
            numNonevidence += 1

    probabilities = []
    for i in range(numSamples):
        probabilityRow = []
        for j in range(numNonevidence):
            probabilityRow.append(random.uniform(0, 1))
        probabilities.append(probabilityRow)

    paths = []
    for i in DG.nodes:
        path = []
        recurseParents(DG, i, path)
        path = path[::-1]
        paths.append(path)

    samples = []
    weightAllSamples = 0
    for probabilityRow in probabilities:
        counter = 0
        totalWeight = 1
        sample = {}
        weightedSample = []
        for path in paths:
            for i in range(len(path)):
                counter, weight = weightedSamplePath(DG, sample, probabilityRow, path, i, counter)
                totalWeight = totalWeight * weight
        weightedSample.append(sample)
        weightedSample.append(totalWeight)
        weightAllSamples += totalWeight
        samples.append(weightedSample)

    weightQueryTrue = 0
    for weightedSample in samples:
        sample = weightedSample[0]
        weight = weightedSample[1]
        for i in DG.nodes:
            node = DG.node[i]['data']
            if node.type == VariableType.query:
                if sample[node.node] == ValueType.true:
                    weightQueryTrue += weight

    print('Weighted Likelihood Sampling:')
    print('Probability of query: %f' % (weightQueryTrue / weightAllSamples))


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    if len(sys.argv) != 4:
        sys.exit('Usage: python Project6.py [network_file] [query_file] [#_samples]')
    networkFile = sys.argv[1]
    queryFile = sys.argv[2]
    samples = int(sys.argv[3])

    choice = AorB(networkFile)
    nodes, DG = constructNetwork(networkFile, choice)
    readQuery(queryFile, nodes, DG)

    rejectionSampling(DG, samples)
    print()
    likelihoodWeightingSampling(DG, samples)


if __name__ == '__main__':
    main()