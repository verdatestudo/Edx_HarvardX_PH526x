import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster.bicluster import SpectralCoclustering
import numpy as np

def produce_pcolor(data):
    '''
    Takes a data pandas dataframe and produces a heatmap.
    '''
    plt.figure(figsize=(10, 10))
    plt.pcolor(data)
    plt.colorbar()
    plt.axis('tight')
    plt.savefig('{}.png'.format(data.name))


whisky = pd.read_csv('whiskies.txt')
whisky['Region'] = pd.read_csv('regions.txt') #add region data for each whisky

flavors = whisky.iloc[:, 2:14] # get cols whch are related to flavor

corr_flavors = pd.DataFrame.corr(flavors)
corr_flavors.name = 'corr_flavors' #show correlation between different types of flavor.
corr_whisky = pd.DataFrame.corr(flavors.transpose()) # .corr compares columns vs each other, so use transpose to move whiskies to columns for comparison.
corr_whisky.name = 'corr_whisky' #show correlation between different types of whisky.

# produce charts
produce_pcolor(corr_flavors)
produce_pcolor(corr_whisky)


# the whiskies were made in six different regions, so we expect six different clusters.
# co-clustering approx. can be done with words in rows and documents in columns (example from video). in this case, flavor in rows, whiskies in column.
model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whisky)

#print(model.rows_)
#print(np.sum(model.rows_, axis=1))
#print(model.rows_labels_)

whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index) # whisky index, cluster as value
whisky = whisky.ix[np.argsort(model.row_labels_)] # sort and order by clusters
whisky = whisky.reset_index(drop=True)

corrs = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
corrs = np.array(corrs)

plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.axis('tight')
plt.title('Original')
plt.subplot(122)
plt.pcolor(corrs)
plt.axis('tight')
plt.title('Rearranged')
plt.savefig('{}.png'.format('correlations'))
