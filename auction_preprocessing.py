# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:59:42 2021

@author: Andrew

SELECT "projectId", count("status") mints FROM artblocks_mints_expanded
WHERE "status"='confirmed'
GROUP BY 1
ORDER BY mints DESC
"""

import pandas as pd
import numpy as np
from scipy import stats
import datetime

"""

preprocessing auction data

"""
#will need to do something special for dutch versus normal auctions later... maybe can just be a dummy var flag for now. 
auctions = pd.read_csv(r'C:/Users/Andrew/OneDrive - nyu.edu/Documents/Python Script Backup/artblock_auctions_analytics/datasets/auctions_821.csv')
projects_keep = [118,133,110,131,143,140]
auctions = auctions[auctions["projectId"].isin(projects_keep)]

auctions["sender"] = auctions["sender"].apply(lambda x: x.lower())
auctions["gas_eth"] = auctions["gas_limit"]*auctions["gas_price"]
auctions["timestamp"] = pd.to_datetime(auctions["timestamp"])
auctions["timestamp"] = auctions["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")) #reduce to seconds granularity instead for plotting reasons

auctions = auctions.sort_values(by="timestamp",ascending=True)
auctions["blocknumber"] = auctions["blocknumber"].replace(to_replace=0, method='bfill') #deal with dropped txs that show as blocknumber 0

#need to string together txs based on hash + replacementHash, but this is a nested dict due to multiple speedups. 
replaceHashKeys = dict(zip(auctions["replaceHash"],auctions["tx_hash"])) #assign tx_hash based on replacements, just to keep consistency. 
replaceHashKeys.pop("none") #remove none key

def recursive_tx_search(key):
    if key in replaceHashKeys:
        return recursive_tx_search(replaceHashKeys[key])
    else:
        return key

auctions["tx_hash"] = auctions["tx_hash"].apply(lambda x: recursive_tx_search(x))

###removing some user outliers
outliers = ["0xd387a6e4e84a6c86bd90c158c6028a58cc8ac459","0x35632b6976b5b6ec3f8d700fabb7e1e0499c1bfa"] #these guys are super whales
auctions = auctions[~auctions["sender"].isin(outliers)]

#check for outliers in mint time, to remove those that were minted way before or after the peak auction period.
to_remove_indicies = []
for project in list(set(auctions["projectId"])):
    auction_spec = auctions[auctions["projectId"]==project]
    all_times = pd.Series(list(set(auction_spec.blocknumber)))
    to_remove_blocktimes = all_times[(np.abs(stats.zscore(all_times)) > 2.5)] #remove values more than 2 std dev of time away. likely not part of main auction time
    if len(to_remove_blocktimes)==0:
        break
    to_remove_indicies.extend(auction_spec.index[auction_spec['blocknumber'].isin(to_remove_blocktimes)].tolist())
auctions.drop(index=to_remove_indicies, inplace=True)

# get full user list for dune query
all_users = list(set(auctions["sender"].apply(lambda x: x.replace('0x','\\x'))))
all_users_string = "('" + "'),('".join(all_users) + "')"

"""

feature engineering by auction

"""
def fill_pending_values_gas(x):
    first = x.first_valid_index()
    last = x.last_valid_index()
    x.loc[first:last] = x.loc[first:last].fillna(method="ffill")
    return x

def get_actions_diff(row):
    row = row.dropna().reset_index()
    actions_diff_nominal =list(row["blocknumber"].diff(1).fillna(0))
 
    #take the blocks with muliple actions and subtract one, then sum up. 
    zeros_to_add = sum([ actions - 1 if actions > 1 else 0 for actions in row[row.columns[1]]])
    actions_diff_nominal.extend(list(np.zeros(int(zeros_to_add))))
    actions_diff = np.mean(actions_diff_nominal)
    if (actions_diff==0) and (zeros_to_add==0):
        return 2000 #meaning they never took another action
    else:
        return actions_diff

def preprocess_auction(df, projectId):
    """
    
    creates all the features for auction analysis. Could definitely be refactored further for each feature calc.
    
    """
    if projectId not in df["projectId"].unique():
        raise ValueError("projectId does not exist in dataframe")
        
    df = df[df["projectId"]==projectId]
    
    ##@feature: calculate basic auction stats
    user_state_pivot = df.pivot_table(index=["sender"], columns="status",values="gas_eth", aggfunc="count")
    user_state_pivot.fillna(0, inplace=True)
    user_state_pivot.drop(columns="pending", inplace=True)
    user_number_submitted = df.pivot_table(index="sender", values="tx_hash", aggfunc=lambda x: len(x.unique()))
    user_number_submitted.columns = ["number_submitted"]

    ##@feature: calculate avg gas per block difference between pending/speedup and confirmed. shift by 1 since it is pending for next block
    gas_activity = df.pivot_table(index="sender", columns="blocknumber",values="gas_eth", aggfunc="mean") \
                    .reindex(set(df["blocknumber"]), axis=1, fill_value=np.nan)
    gas_activity = gas_activity.T.reset_index().sort_values(by="blocknumber",ascending=True).set_index("blocknumber").T
    
    gas_activity = gas_activity.apply(lambda x: fill_pending_values_gas(x), axis=1)
    
    gas_needed = df[df["status"]=="confirmed"].pivot_table(columns="blocknumber",values="gas_eth", aggfunc="mean") \
                        .reindex(set(df["blocknumber"]), axis=1, fill_value=np.nan)
    gas_needed = gas_needed.T.reset_index().sort_values(by="blocknumber",ascending=True).set_index("blocknumber")
    gas_needed = gas_needed.fillna(method="backfill").fillna(method="ffill")
    gas_needed = gas_needed.T
    
    for number in gas_needed.columns:
        gas_activity[number] = gas_activity[number] - gas_needed[number][0] #positive is extra gas, negative is missing gas
    
    gas_activity["average_gas_behavior"] = gas_activity.mean(axis=1)
    gas_activity["median_gas_behavior"] = gas_activity.iloc[:,:-1].median(axis=1)
    gas_activity["stdev_gas_behavior"] = gas_activity.iloc[:,:-2].std(axis=1)
    gas_activity["stdev_gas_behavior"].fillna(0,inplace=True) #no stdev for if there was only one block pending
    
    ##@feature: getting time diff per row. Not sure if there is a way to map this across all rows. 
    get_first_pending = df[df["status"]=="pending"] #first submitted 
    get_first_pending = get_first_pending.drop_duplicates(subset=["tx_hash","status"], keep="first")
    auctions_time_data = pd.concat([get_first_pending,df[df["status"]=="speedup"]], axis=0)
    time_action = auctions_time_data.pivot_table(index=["sender","tx_hash"], columns="blocknumber",values="status",aggfunc="count") \
                    .reindex(set(df["blocknumber"]), axis=1, fill_value=np.nan)
    
    sorting_blocknumber_columns = time_action.columns.sort_values(ascending=True)
    time_action = time_action[sorting_blocknumber_columns]
    time_action["average_action_delay"] = time_action.apply(lambda x: get_actions_diff(x),axis=1)
    time_action["total_actions"] = time_action.iloc[:,:-1].sum(axis=1)
    users_actions = time_action.reset_index().pivot_table(index="sender",values=["total_actions","average_action_delay"], 
                                                          aggfunc={"total_actions":"sum","average_action_delay":"mean"})
    #get time of participation
    first_mint = get_first_pending["blocknumber"].min()
    get_first_pending["block_entry"] = get_first_pending["blocknumber"] - first_mint
    entry_pivot = get_first_pending.pivot_table(index="sender",values="block_entry",aggfunc="min")
    
    #merge all features together on outer join!
    user_state_featurized = pd.merge(user_number_submitted.reset_index(),user_state_pivot.reset_index(),on="sender",how="outer")
    user_state_featurized = pd.merge(user_state_featurized,gas_activity[["average_gas_behavior","stdev_gas_behavior","median_gas_behavior"]].reset_index(),on="sender",how="outer")
    user_state_featurized = pd.merge(user_state_featurized,users_actions[["average_action_delay","total_actions"]].reset_index(),on="sender",how="outer")
    user_state_featurized = pd.merge(user_state_featurized,entry_pivot["block_entry"].reset_index(),on="sender",how="outer")
    
    #for some transactions, it was never pending and went straight to final state. So block_entry, average_action_delay, and total_actions all currently show up as nan
    user_state_featurized.loc[user_state_featurized["block_entry"].isna(),"block_entry"] = user_state_featurized.loc[user_state_featurized["block_entry"].isna(),"sender"]\
        .apply(lambda x: df[df["sender"]==x]["blocknumber"].reset_index(drop=True).min() - first_mint) #convoluted line because user_state_featurized doesn't carry blocknumber column anymore 
    user_state_featurized = user_state_featurized[user_state_featurized["block_entry"]>=0] #don't include those who minted before the auction started.
    user_state_featurized.loc[user_state_featurized["average_action_delay"].isna(),"average_action_delay"] = 2000 
    user_state_featurized.loc[user_state_featurized["total_actions"].isna(),"total_actions"] = 1 

    user_state_featurized["block_entry"] = pd.to_numeric(user_state_featurized["block_entry"])
    user_state_featurized["projectId"] = projectId #create projectId column
    return user_state_featurized

# ##testing start
# projectId=140
# df = auctions[auctions["projectId"]==projectId]
# user_state_pivot = df.pivot_table(index=["sender","tx_hash"], columns="status",values="gas_eth", aggfunc="count")
# user_state_pivot.fillna(0, inplace=True) 
# user_state_pivot.reset_index(inplace=True)

# x='0x94a465183ff9a939295d3c8cc09b5ca27a63fb9c'
# tx_track = df[df["sender"]=='0x94a465183ff9a939295d3c8cc09b5ca27a63fb9c']
# ##testing end

auctions_all = []
for projectId in auctions["projectId"].unique():
    print("preprocessing data for auction #", projectId)
    auctions_all.append(preprocess_auction(auctions, projectId))
    
auctions_all_df = pd.concat(auctions_all)
auctions_all_df["dropped"].fillna(0,inplace=True) #some auctions had 0 dropped so then shows up as nan after concat
auctions_all_df.loc[auctions_all_df[["cancel","confirmed","failed","dropped"]].eq(0).all(1),"failed"] = 1 #after I checked many of them manually, all of the txs with missing end states are failed, though I'm not sure why they don't show up in mempool data. 

"""appending user wallet data"""
#there are 889 users out of 3385 who have an ENS registered on Ethereum. 
wh = pd.read_csv(r'C:/Users/Andrew/OneDrive - nyu.edu/Documents/Python Script Backup/artblock_auctions_analytics/datasets/dune_auction_participants.csv', index_col=0)
wh.dropna(how="all",inplace=True)

wh["user_address"] = wh["user_address"].apply(lambda x: x.replace("\\x","0x"))
wh = wh.rename(columns={'user_address':'sender'})
wh["time_since_first_tx"] = wh["time_since_first_tx"].apply(lambda x: "0" if x == "00:00:00" else x)
wh["time_since_first_tx"]=wh["time_since_first_tx"].apply(lambda x: int(x.split(" ")[0]))

auctions_all_df = pd.merge(auctions_all_df,wh,on="sender",how="left")
auctions_all_df.set_index(["sender","ens"],inplace=True)

"""

k-means + tsne

"""
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics

import warnings
warnings.simplefilter("ignore", UserWarning)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot

def run_clustering(df,cluster_algo,projectId=None,seed=42):
    """
    
    clusters and plots results for a given project    
    
    """
    if projectId == None:
        pass
    elif projectId not in df["projectId"].unique():
        raise ValueError("projectId does not exist in dataframe")
    else:
        df = df[df["projectId"]==projectId]
    df = df.drop(columns="projectId")
        
    tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=300, random_state=seed)
    tsne_results = tsne.fit_transform(df)
    df[["tsne_0","tsne_1"]] = tsne_results
        
    y_pred = cluster_algo.fit_predict(df[["tsne_0","tsne_1"]]) #can do features or PCA here. 
    merged_components=df
    merged_components["clusters"] = y_pred
    merged_components["clusters"] = merged_components["clusters"].apply(lambda x: str(x))
    merged_components.reset_index(inplace=True)
    print('Silhouette Score: ', metrics.silhouette_score(df[["tsne_0","tsne_1"]], cluster_algo.labels_))

    # #elbow point, where decrease faster is better. To plot this, comment out the other charts below.
    # inertia = []
    # for k in range(1, 8):
    #     kmeans = KMeans(n_clusters=k, random_state=1).fit(df[["tsne_0","tsne_1"]])
    #     inertia.append(np.sqrt(kmeans.inertia_))   
    # plt.plot(range(1, 8), inertia, marker='s');
    # plt.xlabel('$k$')
    # plt.ylabel('$J(C_k)$');
    
    # scatterplot of clusters based on t-SNE
    fig, ax = plt.subplots(figsize=(10,10))
    sns.scatterplot(data=merged_components, x="tsne_0",y="tsne_1",hue="clusters", ax=ax)
    ax.set(title="User Auction Behavior Groups Project {}".format(projectId))

    # Density plot
    merged_components_melt = merged_components.set_index(["sender","ens","clusters"]).melt(ignore_index=False)
    merged_components_melt.reset_index(inplace=True, level=2)
    g = sns.FacetGrid(merged_components_melt, col="variable", hue="clusters", 
                      sharey=False,sharex=False, col_wrap=5)
    g.map(sns.kdeplot, "value", shade=True)
    g.add_legend()
    
    # # boxplot of average_action_delay and average_gas_behavior
    # fig2, ax2 = plt.subplots(figsize=(10,10)) 
    # sns.boxplot(x=merged_components["clusters"],y=merged_components["average_action_delay"], ax=ax2)
    
    # fig3, ax3 = plt.subplots(figsize=(10,10)) 
    # sns.boxplot(x=merged_components["clusters"],y=merged_components["average_gas_behavior"], ax=ax3)

    # #plotly
    # fig = px.scatter(merged_components,x="tsne_0",y="tsne_1", color="clusters",hover_data=merged_components.columns)
    # plot(fig,filename="kmeans_{}.html".format(projectId))
    
    #get dict for return
    cluster_dict = dict(zip(merged_components.sender,merged_components.clusters))
    return cluster_dict

def try_cluster(x, cluster_dict):
    if x in cluster_dict:
        return cluster_dict[x]

auctions_temp = auctions_all_df.copy()
auctions_temp.reset_index(inplace=True)
bidders_progression_df = auctions_all_df.reset_index()[["sender","ens"]].drop_duplicates()

seed=20
cluster_algo = KMeans(n_clusters=3, random_state=seed) #3 and 4 both look fine
for projectId in auctions["projectId"].unique():
    print(projectId)
    project_cluster_dict = run_clustering(auctions_all_df,cluster_algo,projectId,seed)
    column_name = "cluster_for_{}".format(projectId)
    
    bidders_progression_df[column_name] = bidders_progression_df["sender"].apply(lambda x: try_cluster(x,project_cluster_dict))
    auctions_temp[column_name] = auctions_temp["sender"].apply(lambda x: try_cluster(x, project_cluster_dict))

#get percentage failed per cluster for each project
for projectId in projects_keep:
    percentage_cluster = auctions_temp[auctions_temp["projectId"]==projectId].pivot_table(index="cluster_for_{}".format(projectId),values=["number_submitted","failed","confirmed","cancel","dropped"],aggfunc="sum")
    percentage_cluster["percent_lost"] = (percentage_cluster["dropped"] + percentage_cluster["cancel"] + percentage_cluster["failed"])/percentage_cluster["number_submitted"]
    percentage_cluster["percent_confirmed"] = percentage_cluster["confirmed"]/percentage_cluster["number_submitted"]
    print(percentage_cluster)

# @todo: add heatmap corr matrix, show main gas/action variables across auctions and also other variables? 

# # need to specify x and y cols better here before next use. 
# sns.pairplot(data=merged_components, x=["tsne-0","tsne-1], y=[], height=3, hue="clusters", kind="scatter")

"""graveyard"""
###putting all epochs together does nothing lol
# final_auctions_df = auctions_all_df.pivot_table(index=auctions_all_df.index,values=auctions_all_df.columns,
#                                                 aggfunc={"number_submitted":"sum","cancel":"sum","confirmed":"sum","failed":"sum","speedup":"sum",
#                                                          "average_gas_behavior":"mean","stdev_gas_behavior":"mean","median_gas_behavior":"mean",
#                                                          "average_action_delay":"mean","total_actions":"mean","projectId":"count",
#                                                          "time_since_first_tx":"mean","total_gas_eth":"mean","number_of_txs":"mean"})

# final_df = run_kmeans(final_auctions_df)

###PCA not useful here
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# principalComponents = pca.fit_transform(user_state_featurized) #replace with cosine or clust_df
# PCA_components = pd.DataFrame(principalComponents)

# PCA_components["clusters"] = y_pred
# PCA_components[["sender","ens"]]= user_state_featurized.reset_index()[["sender","ens"]]
# PCA_components.set_index(["sender","ens"],inplace=True)
# PCA_to_merge = PCA_components[["clusters",0,1]]
# PCA_to_merge.columns = ["clusters","PCA_0","PCA_1"]
# merged_components = pd.concat([PCA_to_merge, user_state_featurized],axis=1)

# fig, ax = plt.subplots(figsize=(10,10))
# sns.scatterplot(data=merged_components, x="PCA_0",y="PCA_1",hue="clusters", ax=ax)
# ax.set(title="User Auction Behavior Groups")

###histogram of an auction over time, used for that one tweet only
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
