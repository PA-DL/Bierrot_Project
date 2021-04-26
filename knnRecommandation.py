
import pandas as pd
import surprise
from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
import os 


os.chdir('C:/Users/MSI PA/Desktop/ESILV/Semestre 8/chatbot/socialmediadata-beeradvocate')

df = pd.read_csv("beermodif2.csv")


# create beerID for each beer
grouped_name = df.groupby('beer_name')
temp_df = grouped_name.count()

temp_df_idx = pd.DataFrame(temp_df. index)

temp_df_idx['beerID'] = temp_df_idx. index
dict_df=temp_df_idx[['beerID', 'beer_name']]

desc_dict = dict_df.set_index('beer_name').to_dict()
new_dict = desc_dict['beerID']

df['beerID'] = df.beer_name.map(new_dict)




# create userID for each user
grouped_user = df.groupby('review_profilename')

temp_df_user = grouped_user.count()

temp_df_user_idx = pd.DataFrame(temp_df_user. index)

temp_df_user_idx['userID']=temp_df_user_idx. index
dict_df_user=temp_df_user_idx[['userID', 'review_profilename']]

 

desc_dict_user = dict_df_user.set_index('review_profilename').to_dict()
new_dict_user = desc_dict_user["userID"]

df['userID'] = df.review_profilename.map(new_dict_user)



def read_item_names():
    "return two mappings to convert raw ids into beer names and beer names into raw ids."
    
    file_name = dict_df
    rid_to_name = {}
    name_to_rid = {}

    unique_beers = len(df.beer_name.unique())
    for i in range(unique_beers):
        line = file_name.iloc[i]
        rid_to_name[line[0]] = line[1]
        name_to_rid[line[1]] = line[0]
            
    return rid_to_name, name_to_rid




def get_rec(beer_name, k_):
    "Input Beer name and returns k recommendations based on item similarity Input: String, integer Output: String"
    
    output = []
    beer = str(beer_name)
    
    # Read the mappings raw id <-> beer name
    rid_to_name, name_to_rid = read_item_names()
    
    # Retrieve inner id of the Beer
    beer_input_raw_id = name_to_rid[beer]
    beer_input_inner_id = algo.trainset.to_inner_iid(beer_input_raw_id)
    
    K = k_
    
    # Retrieve inner ids of the nearest neighbors of the Beer
    beer_input_neighbors = algo.get_neighbors(beer_input_inner_id, k=K)
    
    # Convert inner ids of the neighbors into names.
    
    beer_input_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in beer_input_neighbors)
    beer_input_neighbors = (rid_to_name[rid] for rid in beer_input_neighbors)
    
    for beer_ in beer_input_neighbors:
        output. append(beer_)

    return output
    
    

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df[['userID', 'beerID', 'review_overall']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}


algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

response = get_rec('Goudale', 5)

print(response)

