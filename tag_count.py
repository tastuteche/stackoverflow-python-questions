import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#%matplotlib inline
import math

tags = pd.read_csv("./pythonquestions/Tags.csv", encoding='latin1')


def plot_tags(tagCount):

    x, y = zip(*tagCount)

    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 0.8, 50)]

    area = [i / 4000 for i in list(y)]   # 0 to 15 point radiuses
    plt.figure(figsize=(9, 8))
    plt.ylabel("Number of question associations")
    for i in range(len(y)):
        plt.plot(i, y[i], marker='o', linestyle='', ms=area[i], label=x[i])

    plt.legend(numpoints=1)
    # plt.show()
    plt.savefig('tag_count.png', dpi=200)
    plt.clf()
    plt.cla()
    plt.close()


tagCount = collections.Counter(list(tags['Tag'])).most_common(19)
print(tagCount)
plot_tags(filter(lambda x: x[0] != 'python', tagCount))
