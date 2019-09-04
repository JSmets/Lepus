
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'../submission/'))
from utils import list_files, imshow


csv_file = '../data/DeepLearning/DeepLearningExport.csv'

data = pd.read_csv(csv_file)
data.head()


print('The amount of present species and humans is:', data.taxon_name.count())

# drop data with empty images
data_clr = data.drop(data.index[data.taxon_name.isna()],0)

#%% empty/human/animal images imbalancing

df = data.copy()

print(df.taxon_name.isna().sum())
print((df.taxon_name=='[TEAM]').sum())
print((df.taxon_name.notna().values & (df.taxon_name!='[TEAM]').values).sum())

df['category'] = 'animal'
df.category[df.taxon_name.isna()] = 'empty'
df.category[df.taxon_name=='[TEAM]'] = 'human'

count_by_category = df.groupby(['category']).count()

count_by_category_norm = count_by_category / df.count()

f=plt.figure()
count_by_category_norm.file_id.plot.bar(ax=f.gca())
plt.ylabel('images proportion')
plt.title('Data imbalancing for each category')
plt.tight_layout()
plt.ylim([0, 1])
plt.show()

#%% species imbalancing

count_by_species = data_clr.groupby(['taxon_name']).count()

f=plt.figure()
count_by_species.file_id.plot.bar(ax=f.gca())
plt.ylabel('images counting')
plt.title('Data imbalancing for each species')
plt.tight_layout()
plt.show()

#%% empty/human/animal Additional information as improvment

count_by_category_period = df.groupby(['category','file_period']).count()

category_period_statistics = pd.DataFrame(count_by_category_period.file_id / count_by_category.file_id)
category_period_statistics = category_period_statistics.unstack(fill_value=0)
print(category_period_statistics.head())

f=plt.figure()
category_period_statistics.plot.bar(stacked=False, ax=f.gca())
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('day/night/twilight proportion')
plt.title('Proportion of day/night/twilight for each category')
plt.tight_layout()
plt.show()

#%% species Additional information as improvment

count_by_species_period = data_clr.groupby(['taxon_name','file_period']).count()

species_period_statistics = pd.DataFrame(count_by_species_period.file_id / count_by_species.file_id)
species_period_statistics = species_period_statistics.unstack(fill_value=0)
print(species_period_statistics.head()*100)

f=plt.figure()
species_period_statistics.plot.bar(stacked=False, ax=f.gca())
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('proportion in %')
plt.title('Proportion of day/night/twilight for each species')
plt.tight_layout()
plt.show()

#%% Examples of images

random.seed(4)
rand_path = random.sample(list(data_clr.file_path),8)

# Create figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 5))

for i, axis in enumerate(axes.flatten()):
    img = cv2.imread(directory + rand_path[i][8:]).astype(np.float32)/255
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    axis.set_title('Image: {}'.format(rand_path[i][8:]))
    axis.imshow(img)
    axis.get_xaxis().set_visible(False) # disable x-axis
    axis.get_yaxis().set_visible(False) # disable y-axis
    
plt.show()

#%% Timelaps images

df = data.set_index('file_id')

directory = '../data/DeepLearning/DeepLearning/'

idx = 14

title1 = 'Timelaps image ' + str(idx) + ' with label: ' + str(df.taxon_name[idx])
img1 = cv2.imread(directory + df.file_path[idx][8:]).astype(np.float32)/255
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

idx = idx+1

title2 = 'Timelaps image ' + str(idx) + ' with label: ' + str(df.taxon_name[idx])
img2 = cv2.imread(directory + df.file_path[idx][8:]).astype(np.float32)/255
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

title3 = 'Difference of the two images'
img3 = 0.5*(img2 - img1) + 0.5

# Create figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

axes[0].set_title(title1)
axes[0].imshow(img1)
axes[0].get_xaxis().set_visible(False) # disable x-axis
axes[0].get_yaxis().set_visible(False) # disable y-axis

axes[1].set_title(title2)
axes[1].imshow(img2)
axes[1].get_xaxis().set_visible(False) # disable x-axis
axes[1].get_yaxis().set_visible(False) # disable y-axis

axes[2].set_title(title3)
axes[2].imshow(img3)
axes[2].get_xaxis().set_visible(False) # disable x-axis
axes[2].get_yaxis().set_visible(False) # disable y-axis

plt.show()

#%% Events number and images number

events = df.groupby(['event_id']).count()

f=plt.figure()
plt.subplot(121)
events.file_path.hist(ax=f.gca(), bins=50)
plt.ylabel('Number of events')
plt.xlabel('Number of images')
plt.title('Number of images per events')
plt.tight_layout()
plt.show()

plt.subplot(122)
bins = np.arange(0, events.file_path.max() + 0.5) - 0.5
events.file_path.hist(ax=f.gca(),bins=bins)
plt.ylabel('Number of events')
plt.xlabel('Number of images')
plt.title('Number of images per events')
plt.tight_layout()
plt.xlim([0, 25])
plt.show()

