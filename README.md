# Package
## Python module
* `scipy`
* `numpy`
* `gdown==4.7.3`
* `pandas`
* `os`
* `zipfile`

# Usage
## Collaborative Filtering
```python
from agent import Agent
agent = Agent(dataset_path='dataset', weight_path='weight', download_dataset=True, download_weight=True)

# Get k recommended animes by id
print(agent.find_similar_animes(id=813, k=10))

# Get k recommended animes by name
print(agent.find_similar_animes(name='Dragon Ball Z', k=10))

# Get DataFrame result
print(agent.find_similar_animes(name='Dragon Ball Z', k=10, return_df=True))

# Get top_k * num_animes recommend_animes using watched_episodes attribute by user_id, return id result
print(agent.find_anime_for_user_using_episode(id=0, top_k=5, num_animes=4))

# Get top_k * num_animes recommend_animes using watched_episodes attribute by user_id, return name result
print(agent.find_anime_for_user_using_episode(id=0, top_k=5, num_animes=4, return_name=True))

# Get top_k * num_animes recommend_animes using watched_episodes attribute by user_id, return DataFrame result
print(agent.find_anime_for_user_using_episode(id=0, top_k=5, num_animes=4, return_df=True))

# Get top_k * num_animes recommend_animes using rating attribute by user_id, return id result
print(agent.find_anime_for_user_using_rating(id=0, top_k=5, num_animes=4))

# Get top_k * num_animes recommend_animes using rating attribute by user_id, return name result
print(agent.find_anime_for_user_using_rating(id=0, top_k=5, num_animes=4, return_name=True))

# Get top_k * num_animes recommend_animes using rating attribute by user_id, return DataFrame result
print(agent.find_anime_for_user_using_rating(id=0, top_k=5, num_animes=4))
```
## FP-growth
```python
from agent import Agent
agent = Agent(dataset_path='dataset', weight_path='weight', download_dataset=True, download_weight=True)
agent.build_itemSetList(num_users=313670, num_animes=17172)
agent.build_fpgrowth(minSup=0.19, minConf=0.5)

# Get all rules of FP-growth algorithm by id
print(agent.rules_fpgrowth)

# Get all rules of FP-growth algorithm by name
for rule in agent.rules_fpgrowth:
    print(agent.anime_df.loc[agent.anime_df['MAL_ID'].isin(list(rule[0]))]['Name'].tolist(), end=' ')
    print('--->', end=' ')
    print(agent.anime_df.loc[agent.anime_df['MAL_ID'].isin(list(rule[1]))]['Name'].tolist())

# Get recommended animes using fp-growth algorithm by user_id, return id result
agent.find_anime_for_user_using_fpgrowth(id=12)

# Get recommended animes using fp-growth algorithm by user_id, return name result
agent.find_anime_for_user_using_fpgrowth(id=12, return_name=True)

# Get recommended animes using fp-growth algorithm by user_id, return DataFrame result
agent.find_anime_for_user_using_fpgrowth(id=12, return_df=True)
```

# Algorithm Tutorial
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14RzLFOnpWyvpsUsygTfF5HB29xyopL-x?usp=sharing)

# Dataset
## Cropped dataset
[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1CYjnad4Qmc5wx9BpXKcbHMbHE18iQNOa?usp=sharing)
