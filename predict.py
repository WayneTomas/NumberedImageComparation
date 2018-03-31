from resource import DataSet
from network import NetWork

import numpy as np

dataset = DataSet('', 0)
network = NetWork()

import sys

args = {
    '-p1': None,
    '-p2': None,
    '-m': None
}

index = 1
while index < len(sys.argv):
    i = sys.argv[index]
    if i == '-p1':
        args['-p1'] = sys.argv[index + 1]
        index += 1
    elif i == '-p2':
        args['-p2'] = sys.argv[index + 1]
        index += 1
    elif i == '-m':
        args['-m'] = sys.argv[index + 1]
        index += 1
    else:
        raise Exception('Wrong parameter of %d', sys.argv[index])
    index += 1

if args['-p1'] == None or args['-p2'] == None or args['-m'] == None:
    raise Exception('Parameters are not completely.')

p1 = args['-p1']
p2 = args['-p2']
m = args['-m']

processed_data = np.array([dataset.get_image_data((p1, p2))])
network.load_model(m)

predict = network.predict(processed_data)

if predict:
    print('Same.')
else:
    print('Not same.')
