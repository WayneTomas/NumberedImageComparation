'''
File: \train.py
Project: NumberRecongization
Created Date: Saturday March 31st 2018
Author: Huisama
-----
Last Modified: Saturday March 31st 2018 11:03:45 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

from network import NetWork

import sys

args = {
    '-t': None,
    '-b': None
}

index = 1

while index < len(sys.argv):
    i = sys.argv[index]
    if i == '-t':
        args['-t'] = sys.argv[index + 1]
        index += 1
    elif i == '-b':
        args['-b'] = sys.argv[index + 1]
        index += 1
    else:
        raise Exception('Wrong parameter of %s' % sys.argv[index])
    index += 1

episode = int(args['-t']) if args['-t'] != None else 200
batch_size = int(args['-b']) if args['-b'] != None else 128

network = NetWork()
network.build_network()

network.train(episode, batch_size)
