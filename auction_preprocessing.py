# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:59:42 2021

@author: Andrew
"""

import pandas as pd

"""preprocessing auction data into features. This is currently done per auction."""
#will need to do something special for dutch versus normal auctions later... maybe can just be a dummy var flag for now. 
auctions = pd.read_csv(r'C:/Users/Andrew/OneDrive - nyu.edu/Documents/Python Script Backup/blocknative_auctions/datasets/auction118.csv')
auctions["gas_eth"] = auctions["gas_limit"]*auctions["gas_price"]
auctions["timestamp"] = pd.to_datetime(auctions["timestamp"])
auctions["timestamp"] = auctions["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")) #reduce to seconds granularity instead for plotting reasons

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

user_state_pivot = auctions.pivot_table(index=["sender"], columns="status",values="gas_eth", aggfunc="count")
user_state_pivot.fillna(0, inplace=True)
#first time I forgot to filter for drops, so we have 4 missing final tx states here. Future shouldn't need this.
user_state_pivot = user_state_pivot[~user_state_pivot[["cancel","confirmed","failed"]].eq(0).all(1)] 
user_state_pivot.drop(columns="pending", inplace=True)
user_number_submitted = auctions.pivot_table(index="sender", values="tx_hash", aggfunc=lambda x: len(x.unique()))
user_number_submitted.columns = ["number_submitted"]

user_state_featurized = pd.merge(user_number_submitted.reset_index(),user_state_pivot.reset_index(),on="sender",how="outer")

##calculate avg gas per block difference between pending/speedup and confirmed. shift by 1 since it is pending for next block
gas_activity = auctions[(auctions["status"]=="pending") | (auctions["status"]=="speedup")].pivot_table(index="sender", columns="blocknumber",values="gas_eth", aggfunc="mean")
a = gas_activity.shift(1, axis=1) # this doesn't work since some blocknumbers didn't make it. So we need to keep past columns
gas_needed = auctions[auctions["status"]=="confirmed"].pivot_table(index="blocknumber",values="gas_eth", aggfunc="mean")
columns_to_keep = set(gas_needed.index) - set(gas_activity.columns)
gas_activity = gas_activity.loc[:,list(gas_needed.index)] #only keep up to the last block where there was a confirmation

temp = gas_activity.subtract(gas_needed, axis=1) #how to do this on uneven frames

# calc time between any actions, with 1000 as the value if they never did more than one action. 
# need to keep only first pending for each tx hash, as well as remove any failed/confirmed. Then can take column distance somehow, and average across users. 
time_action = auctions.pivot_table(index=["sender","tx_hash"], columns="blocknumber",values="status",aggfunc="count") 

# get full user list
all_users = list(set(auctions["sender"].apply(lambda x: x.replace('0x','\\x'))))
all_users_string = "'),('".join(all_users)

"""appending user wallet data"""
wh = pd.read_csv(r'C:/Users/Andrew/OneDrive - nyu.edu/Documents/Python Script Backup/blocknative_auctions/datasets/wallet_summaries.csv')

wh["user_address"] = wh["user_address"].apply(lambda x: x.replace("\\x","0x"))
wh["time_since_first_tx"] = wh["time_since_first_tx"].apply(lambda x: "0" if x == "00:00:00" else x)
wh["time_since_first_tx"]=wh["time_since_first_tx"].apply(lambda x: int(x.split(" ")[0]))

"""PCA and k-means"""
# project_details_mapping = 
# project_details_json = 
# combined_project_details = 
# merged next to auctions data? this should provide clarity in various charts


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
