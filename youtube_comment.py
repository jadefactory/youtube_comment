import pandas as pd
from googleapiclient.discovery import build

import os
load_dotenv()


api_key = os.environ.get("API_KEY")
video_id = 'YxQQnn22RXI'
 
comments = list()
api_obj = build('youtube', 'v3', developerKey=api_key)
response = api_obj.commentThreads().list(part='snippet,replies', videoId=video_id, maxResults=100).execute()

while response:
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
 
        # if item['snippet']['totalReplyCount'] > 0:
        #     for reply_item in item['replies']['comments']:
        #         reply = reply_item['snippet']
        #         comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
 
    if 'nextPageToken' in response:
        response = api_obj.commentThreads().list(part='snippet,replies', videoId=video_id, pageToken=response['nextPageToken'], maxResults=100).execute()
    else:        
        break

df = pd.DataFrame(comments, columns=['comment', 'user_name', 'publish_date', 'likes'])
df = df.drop_duplicates(['comment'], keep='first')
df['comment'].values

import re

df['comment_refined'] = df['comment'].apply(lambda x: re.sub('[^가-힣\s]','',x))
df = df[df['comment_refined'].apply(lambda x: re.sub('[^가-힣]','', x)) != '']


from soynlp.word import WordExtractor

word_extractor = WordExtractor(min_frequency=100, min_cohesion_forward=0.05, min_right_branching_entropy=0.0)
word_extractor.train(df['comment_refined'].values)
words = word_extractor.extract()
words

from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus


cohesion_score = {word: score.cohesion_forward for word, score in words.items()}
tokenizer = LTokenizer(scores=cohesion_score)

df['tokenized'] = df['comment_refined'].apply(lambda x: tokenizer.tokenize(x, remove_r=True))
df

words = []
for i in df['tokenized'].values:
    for k in i:
        words.append(k)

words

from collections import Counter

count = Counter(words)
available_count = Counter({x : count[x] for x in count if len(x) > 1})
word_dict = dict(available_count)
word_dict

sorted_word_dict = sorted(word_dict.items(), key = lambda item: item[1], reverse = True)
sorted_word_dict[:10]

df_words = pd.DataFrame(sorted_word_dict, columns=['word', 'count'])
df_words['word'].to_csv('df_words.csv', index=False)

# Commented out IPython magic to ensure Python compatibility.
# 한글 표시 설정
import matplotlib as mpl
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina'
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()

from wordcloud import WordCloud

wordcloud = WordCloud(font_path='/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf', width=500, height=500, background_color="white").generate_from_frequencies(word_dict)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

