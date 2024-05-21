from itertools import chain, combinations
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import bigtree

global viz_tree_dict
class Node:
    def __init__(self, itemName, frequency, parentNode):
        self.itemName = itemName
        self.count = frequency
        self.parent = parentNode
        self.children = {}
        self.next = None
        self.path = 'Null'

    def increment(self, frequency):
        self.count += frequency

    def display(self, ind=1):
        print('-' * ind, self.itemName, ' ', self.count)
        for child in list(self.children.values()):
            print('-' * ind, 'parent:', self.itemName)
            child.display(ind+1)
    def visualize(self):
        global viz_tree_dict
        for child in list(self.children.values()):
            if self.itemName != child.itemName:
                path = self.path + '/' + str(child.itemName)
            else:
                path = self.path
            child.path = path
            if path not in viz_tree_dict.keys():
                viz_tree_dict[path] = {'count': child.count}
            else:
                viz_tree_dict[path]['count'] += child.count
            child.visualize()

def constructTree(itemSetList, frequency, minSup):
    headerTable = defaultdict(int)
    # Counting frequency and create header table
    for idx, itemSet in enumerate(tqdm(itemSetList)):
        for item in itemSet:
            headerTable[item] += frequency[idx]

    # Deleting items below minSup
    headerTable = dict((item, sup) for item, sup in headerTable.items() if sup >= minSup)
    if(len(headerTable) == 0):
        return None, None

    # HeaderTable column [Item: [frequency, headNode]]
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]

    # Init Null head node
    fpTree = Node('Null', 1, None)
    # Update FP tree for each cleaned and sorted itemSet
    for idx, itemSet in enumerate(tqdm(itemSetList)):
        itemSet = [item for item in itemSet if item in headerTable]
        itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
        # Traverse from root to leaf, update tree with given item
        currentNode = fpTree
        for item in itemSet:
            currentNode = updateTree(item, currentNode, headerTable, frequency[idx])

    return fpTree, headerTable

def updateHeaderTable(item, targetNode, headerTable):
    if(headerTable[item][1] == None):
        headerTable[item][1] = targetNode
    else:
        currentNode = headerTable[item][1]
        # Traverse to the last node then link it to the target
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode

def updateTree(item, treeNode, headerTable, frequency):
    if item in treeNode.children:
        # If the item already exists, increment the count
        treeNode.children[item].increment(frequency)
    else:
        # Create a new branch
        newItemNode = Node(item, frequency, treeNode)
        treeNode.children[item] = newItemNode
        # Link the new branch to header table
        updateHeaderTable(item, newItemNode, headerTable)

    return treeNode.children[item]

def ascendFPtree(node, prefixPath):
    if node.parent != None:
        prefixPath.append(node.itemName)
        ascendFPtree(node.parent, prefixPath)

def findPrefixPath(basePat, headerTable):
    # First node in linked list
    treeNode = headerTable[basePat][1]
    condPats = []
    frequency = []
    while treeNode != None:
        prefixPath = []
        # From leaf node all the way to root
        ascendFPtree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            # Storing the prefix path and it's corresponding count
            condPats.append(prefixPath[1:])
            frequency.append(treeNode.count)

        # Go to next node
        treeNode = treeNode.next
    return condPats, frequency

def mineTree(headerTable, minSup, preFix, freqItemList):
    stack = [(headerTable, minSup, preFix)]

    while stack:
        headerTable, minSup, preFix = stack.pop()
        
        sortedItemList = [item[0] for item in sorted(list(headerTable.items()), key=lambda p: p[1][0], reverse=True)]
        
        for item in sortedItemList:
            newFreqSet = preFix.copy()
            newFreqSet.add(item)
            freqItemList.append(newFreqSet)

            conditionalPattBase, frequency = findPrefixPath(item, headerTable)
            conditionalTree, newHeaderTable = constructTree(conditionalPattBase, frequency, minSup)

            if newHeaderTable:
                stack.append((newHeaderTable, minSup, newFreqSet))

def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def getSupport(testSet, itemSetList):
    count = 0
    for itemSet in itemSetList:
        if(set(testSet).issubset(itemSet)):
            count += 1
    return count

def associationRule(freqItemSet, itemSetList, minConf):
    rules = []
    for itemSet in tqdm(freqItemSet):
        subsets = powerset(itemSet)
        itemSetSup = getSupport(itemSet, itemSetList)
        if itemSetSup <= minConf: continue
        for s in subsets:
            confidence = float(itemSetSup / getSupport(s, itemSetList))
            if(confidence > minConf):
                rules.append([set(s), set(itemSet.difference(s)), confidence])
    return rules

def getFrequencyFromList(itemSetList):
    frequency = [1 for i in range(len(itemSetList))]
    return frequency


def fpgrowth(itemSetList, minSupRatio, minConf, visualize=False, generate_rule=True):
    global viz_tree_dict
    viz_tree_dict = dict()
    frequency = getFrequencyFromList(itemSetList)
    minSup = len(itemSetList) * minSupRatio
    fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
    if(fpTree == None):
        print('No frequent item set')
    else:
        fpTree.visualize()
        if visualize:
            bigtree.dict_to_tree(viz_tree_dict).show(attr_list=['count'])
        freqItems = []
        mineTree(headerTable, minSup, set(), freqItems)
        if generate_rule:
            rules = associationRule(freqItems, itemSetList, minConf)
        else: rules=[]
        return freqItems, rules
