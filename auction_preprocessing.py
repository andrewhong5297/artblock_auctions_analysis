# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:59:42 2021

@author: Andrew

SELECT "projectId", count("status") mints FROM artblocks_mints_expanded
WHERE "status"='confirmed'
GROUP BY 1
ORDER BY mints DESC

we can reasonably use projects 118, 133, 137, 110, and maybe 131. Others don't have enough data
"""

import pandas as pd
import numpy as np
from scipy import stats

"""

preprocessing auction data

"""
#will need to do something special for dutch versus normal auctions later... maybe can just be a dummy var flag for now. 
auctions = pd.read_csv(r'C:/Users/Andrew/OneDrive - nyu.edu/Documents/Python Script Backup/artblock_auctions_analytics/datasets/auctions_818.csv')
projects_keep = [118,110,133,137,131]
auctions = auctions[auctions["projectId"].isin(projects_keep)]

auctions = auctions[auctions["blocknumber"]!=0] #@todo: delete dropped, figure out how to deal with this issue later. 

to_remove_indicies = []
for project in list(set(auctions["projectId"])):
    #check for outliers
    auction_spec = auctions[auctions["projectId"]==project]
    all_times = pd.Series(list(set(auction_spec.blocknumber)))
    to_remove_blocktimes = all_times[(np.abs(stats.zscore(all_times)) > 2)] #remove values more than 2 std dev of time away. likely not part of main auction time
    if len(to_remove_blocktimes)==0:
        break
    
    to_remove_indicies.extend(auction_spec.index[auction_spec['blocknumber'].isin(to_remove_blocktimes)].tolist())

auctions.drop(index=to_remove_indicies, inplace=True)

auctions["sender"] = auctions["sender"].apply(lambda x: x.lower())
auctions["gas_eth"] = auctions["gas_limit"]*auctions["gas_price"]
auctions["timestamp"] = pd.to_datetime(auctions["timestamp"])
auctions["timestamp"] = auctions["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")) #reduce to seconds granularity instead for plotting reasons

# # get full user list
# all_users = list(set(auctions["sender"].apply(lambda x: x.replace('0x','\\x'))))
# all_users_string = "('" + "'),('".join(all_users) + "')"

# #for sanity check stuff only
# test_df = auctions[auctions["sender"]=="0xcbf3C9b38003fa550A7bfF5206366c1f62480bA7"]
# replaceHashKeysT = dict(zip(test_df["replaceHash"],test_df["tx_hash"])) #assign tx_hash based on replacements, just to keep consistency. 
# replaceHashKeysT.pop("none") #remove none key

#need to string together txs based on hash + replacementHash, but this is a nested dict due to multiple speedups. 
replaceHashKeys = dict(zip(auctions["replaceHash"],auctions["tx_hash"])) #assign tx_hash based on replacements, just to keep consistency. 
replaceHashKeys.pop("none") #remove none key

def recursive_tx_search(key):
    if key in replaceHashKeys:
        return recursive_tx_search(replaceHashKeys[key])
    else:
        return key

def try_replace_root_tx(x):
    try:
        # we need to recursively call the dictionary to find the root hash, since many actions will drop the original and replace it
        return recursive_tx_search(x)
    except:
        return x

auctions["tx_hash"] = auctions["tx_hash"].apply(lambda x: try_replace_root_tx(x))

###removing some user outliers
outliers = ["0xd387a6e4e84a6c86bd90c158c6028a58cc8ac459","0x35632b6976b5b6ec3f8d700fabb7e1e0499c1bfa"] #these guys are super whales
auctions = auctions[~auctions["sender"].isin(outliers)]

"""

feature engineering by auction (may be a way to do cumulative later)

"""
# @todo: replace with for loop later
# just concat each dataframe. Then could pivot table with count for projects, sum for states, averages for the rest of the metrics, 
# and a column for each project and their cluster position

# for project in auctions["projectId"].unique():

auctions = auctions[auctions["projectId"]==118]

###@feature: calculate basic auction stats
user_state_pivot = auctions.pivot_table(index=["sender"], columns="status",values="gas_eth", aggfunc="count")
user_state_pivot.fillna(0, inplace=True)

#filter out dropped, maybe come fix later but this is only like 6 txs out of thousands. 
user_state_pivot = user_state_pivot[~user_state_pivot[["cancel","confirmed","failed"]].eq(0).all(1)] 
user_state_pivot.drop(columns="pending", inplace=True)
user_number_submitted = auctions.pivot_table(index="sender", values="tx_hash", aggfunc=lambda x: len(x.unique()))
user_number_submitted.columns = ["number_submitted"]


###@feature: calculate avg gas per block difference between pending/speedup and confirmed. shift by 1 since it is pending for next block
gas_activity = auctions.pivot_table(index="sender", columns="blocknumber",values="gas_eth", aggfunc="mean") \
                .reindex(set(auctions["blocknumber"]), axis=1, fill_value=np.nan)
gas_activity = gas_activity.T.reset_index().sort_values(by="blocknumber",ascending=True).set_index("blocknumber").T

def fill_pending_values(x):
    first = x.first_valid_index()
    last = x.last_valid_index()
    x.loc[first:last] = x.loc[first:last].fillna(method="ffill")
    return x

gas_activity = gas_activity.apply(lambda x: fill_pending_values(x), axis=1)

gas_needed = auctions[auctions["status"]=="confirmed"].pivot_table(columns="blocknumber",values="gas_eth", aggfunc="mean") \
                    .reindex(set(auctions["blocknumber"]), axis=1, fill_value=np.nan)
gas_needed = gas_needed.T.reset_index().sort_values(by="blocknumber",ascending=True).set_index("blocknumber")
gas_needed = gas_needed.fillna(method="backfill").fillna(method="ffill")
gas_needed = gas_needed.T

for number in gas_needed.columns:
    gas_activity[number] = gas_activity[number] - gas_needed[number][0] #positive is extra gas, negative is missing gas

gas_activity["average_gas_behavior"] = gas_activity.mean(axis=1)
gas_activity["median_gas_behavior"] = gas_activity.iloc[:,:-1].median(axis=1)
gas_activity["stdev_gas_behavior"] = gas_activity.iloc[:,:-2].std(axis=1)
gas_activity["stdev_gas_behavior"].fillna(0,inplace=True) #no stdev for if only one block pending

###@feature: getting time diff per row. Not sure if there is a way to map this across all rows. 
get_first_pending = auctions[auctions["status"]=="pending"] #first submitted 
get_first_pending = get_first_pending.drop_duplicates(subset=["tx_hash","status"], keep="first")
auctions_time_data = pd.concat([get_first_pending,auctions[auctions["status"]=="speedup"]], axis=0)
time_action = auctions_time_data.pivot_table(index=["sender","tx_hash"], columns="blocknumber",values="status",aggfunc="count") \
                .reindex(set(auctions["blocknumber"]), axis=1, fill_value=np.nan)

sorting_blocknumber_columns = time_action.columns.sort_values(ascending=True)
time_action = time_action[sorting_blocknumber_columns]

def get_actions_diff(row):
    row = row.dropna().reset_index()
    actions_diff_nominal =list(row["blocknumber"].diff(1).fillna(0))
 
    #take the blocks with muliple actions and subtract one, then sum up. 
    zeros_to_add = sum([ actions - 1 if actions > 1 else 0 for actions in row[row.columns[1]]])
    actions_diff_nominal.extend(list(np.zeros(int(zeros_to_add))))
    actions_diff = np.mean(actions_diff_nominal)
    if (actions_diff==0) and (zeros_to_add==0):
        return 5000 #meaning they never took another action
    else:
        return actions_diff

row = time_action.iloc[1,:]
time_action["average_action_delay"] = time_action.apply(lambda x: get_actions_diff(x),axis=1)
time_action["total_actions"] = time_action.iloc[:,:-1].sum(axis=1)

#pivot a final time by sender, without tx_hash and using only average_action_delay.
users_actions = time_action.reset_index().pivot_table(index="sender",values=["total_actions","average_action_delay"], 
                                                      aggfunc={"total_actions":"sum","average_action_delay":"mean"})

#deal with some nans
user_state_featurized = pd.merge(user_number_submitted.reset_index(),user_state_pivot.reset_index(),on="sender",how="outer")
user_state_featurized = pd.merge(user_state_featurized,gas_activity[["average_gas_behavior","stdev_gas_behavior","median_gas_behavior"]].reset_index(),on="sender",how="outer")
user_state_featurized = pd.merge(user_state_featurized,users_actions[["average_action_delay","total_actions"]].reset_index(),on="sender",how="outer")

"""appending user wallet data"""
#there are 889 users out of 3385 who have an ENS registered on Ethereum. 
wh = pd.read_csv(r'C:/Users/Andrew/OneDrive - nyu.edu/Documents/Python Script Backup/artblock_auctions_analytics/datasets/dune_auction_participants.csv', index_col=0)
wh.dropna(how="all",inplace=True)

wh["user_address"] = wh["user_address"].apply(lambda x: x.replace("\\x","0x"))
wh = wh.rename(columns={'user_address':'sender'})
wh["time_since_first_tx"] = wh["time_since_first_tx"].apply(lambda x: "0" if x == "00:00:00" else x)
wh["time_since_first_tx"]=wh["time_since_first_tx"].apply(lambda x: int(x.split(" ")[0]))

user_state_featurized = pd.merge(user_state_featurized,wh,on="sender",how="outer")
user_state_featurized.set_index(["sender","ens"],inplace=True)
user_state_featurized.dropna(inplace=True) #shouldn't need this after

"""PCA and k-means"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=10)
principalComponents = pca.fit_transform(user_state_featurized) #replace with cosine or clust_df

PCA_components = pd.DataFrame(principalComponents)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
y_pred = kmeans.fit_predict(user_state_featurized) #can do features or PCA here. 

###@todo: need to do silhouette and other factors to assess best kmeans

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(user_state_featurized)
user_state_featurized[["tsne_0","tsne_1"]] = tsne_results
merged_components=user_state_featurized
merged_components["clusters"] = y_pred

fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(data=merged_components, x="tsne_0",y="tsne_1",hue="clusters", ax=ax)
ax.set(title="User Auction Behavior Groups")

# PCA_components["clusters"] = y_pred
# PCA_components[["sender","ens"]]= user_state_featurized.reset_index()[["sender","ens"]]
# PCA_components.set_index(["sender","ens"],inplace=True)
# PCA_to_merge = PCA_components[["clusters",0,1]]
# PCA_to_merge.columns = ["clusters","PCA_0","PCA_1"]
# merged_components = pd.concat([PCA_to_merge, user_state_featurized],axis=1)

# fig, ax = plt.subplots(figsize=(10,10))
# sns.scatterplot(data=merged_components, x="PCA_0",y="PCA_1",hue="clusters", ax=ax)
# ax.set(title="User Auction Behavior Groups")

#the following is more useful than pairplot for now
merged_components_melt = merged_components.set_index("clusters", append=True).melt(ignore_index=False)
merged_components_melt.reset_index(inplace=True, level=2)
g = sns.FacetGrid(merged_components_melt, col="variable", hue="clusters", 
                  sharey=False,sharex=False, col_wrap=5)
g.map(sns.kdeplot, "value", shade=True)

# @todo: add heatmap stuff corr matrix?
# this is probably good for comparing across auctions maybe, show their cluster across variables. 
# show main gas/action variables across auctions and also other variables? 

# # need to specify x and y cols better here before use
# sns.pairplot(data=merged_components, height=3, hue="clusters", kind="scatter")

"""histogram of an auction over time"""
# import seaborn as sns
# import matplotlib.pyplot as plt
# histogram_pivot = auctions.pivot_table(index=["blocknumber","tx_hash"],columns="status",values="gas_eth", aggfunc="mean")

# #how to deal with tracking over time... almost needs to be cumulative? which would work except for pending.  
# histogram_pivot_melted = histogram_pivot.melt(ignore_index=False)
# histogram_pivot_melted.columns=["status","gas_eth_value"]
# histogram_pivot_melted.dropna(inplace=True)
# histogram_pivot_melted.reset_index(inplace=True)

# color_map = {"cancel":"orange", "confirmed":"lightgreen", "pending":"yellow", "speedup":"blue", "failed":"red"}

# total_supply = 750 #this should be programmatic later
# confirmed = 0
# all_times = list(set(histogram_pivot_melted["blocknumber"]))
# all_times.sort()

# for time in all_times:
#     # print(time, confirmed)
#     temp_pivot = histogram_pivot_melted[histogram_pivot_melted["blocknumber"]<=time] 
#     #cumulative over time, but should drop old pending. other states should stay I think.
#     temp_pivot = temp_pivot[~((temp_pivot["blocknumber"]<time) & (temp_pivot["status"]=="pending"))]
#     confirmed+=len(temp_pivot[(temp_pivot["blocknumber"]==time) & (temp_pivot["status"]=="confirmed")])
    
#     fig = sns.displot(temp_pivot, x="gas_eth_value", hue="status",binwidth=0.02,height=7,palette=color_map)
#     fig.fig.suptitle("Auction State of LeWitt Generator Generator as of Block Number {}".format(time))
#     fig.set(xlim=(0,1),ylim=(0,250))
#     plt.text(0.9, 150, "Number minted {}/{}".format(confirmed,total_supply), horizontalalignment='left', size='medium', color='black', weight='semibold')
#     plt.tight_layout()
#     fig.savefig(r"C:\Users\Andrew\OneDrive - nyu.edu\Documents\Python Script Backup\datasets\genart\artblocks\round_{}.png".format(time), quality = 85)

# https://ezgif.com/maker/ezgif-2-a4ed6186-gif-equalized

# import glob 
# import moviepy.editor as mpy

# def generate_final():
#     gif_name = 'artblocks_dutch'
#     fps = 3
#     file_list = glob.glob(r"C:\Users\Andrew\OneDrive - nyu.edu\Documents\Python Script Backup\datasets\genart\artblocks\*")
#     clip = mpy.ImageSequenceClip(file_list, fps=fps)
#     clip.write_gif(r"C:\Users\Andrew\OneDrive - nyu.edu\Documents\Python Script Backup\datasets\genart\artblocks\{}.gif".format(gif_name), fps=fps)

# generate_final()
