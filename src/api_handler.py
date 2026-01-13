import requests
from openai import AsyncOpenAI
import json
import asyncio

src_map = {
    'social security matters': 'https://www.ssa.gov/blog/en/',
    'nbc news': 'https://www.nbcnews.com/',
    'white house': 'https://trumpwhitehouse.archives.gov/',
    'us house of representatives document repository': 'https://docs.house.gov/',
    'council on foregin relations': 'https://www.cfr.org/',
    'food and nutrition service': 'https://www.fns.usda.gov/',
    'national library of medicine': 'https://pmc.ncbi.nlm.nih.gov/',
    'dietary guidelines for americans': 'https://www.dietaryguidelines.gov/',
    'harvard health publishing': 'https://www.health.harvard.edu/',
    'wikipedia': 'https://en.wikipedia.org/'
}


def fetch_video_snippet(video_id: str, yt_key: str) -> dict:
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": yt_key
    }

    response = requests.get(url, params=params)
    response.raise_for_status()  # raises error for 4xx / 5xx

    return response.json()


def fetch_video_comments(video_id: str, yt_key: str, top_k: int) -> dict:
    url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': min(top_k, 50),
        'order': 'relevance',
        'key': yt_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()

def parse_video_data(video_id: str, yt_key: str) -> dict:
    videoData = fetch_video_snippet(video_id, yt_key)

    if len(videoData['items'])==0:
        print('video not found or unavailable')
    
    snippet = videoData['items'][0]['snippet']

    videoInfo = {
        'id': video_id,
        'title': snippet['title'],
        'description': snippet['description'],
        'channelTitle': snippet['channelTitle'],
        'tags': snippet['tags'] if 'tags' in snippet.keys() else []
    }

    commentData = fetch_video_comments(video_id, yt_key, 2)
    comments = []
    # print(commentData['items'])
    for item in commentData['items']:
        tmp = item['snippet']['topLevelComment']['snippet']
        comments.append(
            {
                'author': tmp['authorDisplayName'],
                'text': tmp['textOriginal'],
                'likes': tmp['likeCount'],
                'publishedAt': tmp['publishedAt']
            }
        )

    return {
        'video': videoInfo,
        'comments': comments
    }

    

def promptBuilder(metaData: dict, base_prompt: str, type: str) -> str:
    prompt = base_prompt[:]

    if type=='creator':
        prompt += \
f'''\nNow classify the following metadata
channel name: {metaData['video']['channelTitle']}
Title: {metaData['video']['title']}
Tags: {metaData['video']['tags']}
Description: {metaData['video']['description']}
Your Answer:
(Respond only yes or no)'''
        
    elif type=='comments':
        prompt += \
f"""Now Classify the following comments:\n"""
        for i, comm in enumerate(metaData['comments']):
            prompt += f'{i+1}. {comm['text']}\n\n'

        prompt += \
"""Your Output (format must match exactly):
{
 "1": "...",
 "2": "...",
 "3": "...",
 ....
}"""
    elif type=='risk':
        prompt += \
f'''\nNow classify the following metadata
channel name: {metaData['video']['channelTitle']}
Title: {metaData['video']['title']}
Tags: {metaData['video']['tags']}
Description: {metaData['video']['description']}'''
        prompt += \
'''\nYour Output (must match the format exactly):
{
    "category": ...,
    "tactic": ...,
    "risk": ...
}'''
    elif type=='action':
        prompt += \
f'''\n* These are the sources that you must pick from: {src_map.keys()}\n
Now choose sources to verify for a content with the following metadata
channel name: {metaData['video']['channelTitle']}
Title: {metaData['video']['title']}
Tags: {metaData['video']['tags']}
Description: {metaData['video']['description']}'''
        prompt += \
'''\nYour Output (must match the format exactly):
{
    "sources": ["...", "...", ...]
}
'''

    return prompt


async def askGPT(prompt: str, client: AsyncOpenAI) -> str:
    response = await client.responses.create(
        model="gpt-5",
        reasoning={"effort": "low"},
        input=[
            {
                "role": "developer",
                "content": (
                    "You analyze content to determine whether a content is "
                    "AI-generated or not. Respond concisely with no explanation."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.output_text

async def call_gpts_concurrently(data, base_prompt_cr, base_prompt_cm, base_prompt_risk, base_prompt_act, client):

    prompt_cr = promptBuilder(data, base_prompt_cr, 'creator')
    prompt_cm = promptBuilder(data, base_prompt_cm, 'comments')
    prompt_risk = promptBuilder(data, base_prompt_risk, 'risk')
    prompt_act = promptBuilder(data, base_prompt_act, 'action')

    res_cr, res_cm, res_risk, res_src = await asyncio.gather(
        askGPT(prompt_cr, client),
        askGPT(prompt_cm, client),
        askGPT(prompt_risk, client),
        askGPT(prompt_act, client)
    )

    def safe_json_load(s: str, data: dict):
        try:
            res = json.loads(s)
            dt = {}
            for key in res.keys():
                dt[data['comments'][int(key)-1]['text']] = res[key]
            return dt
            
        except json.JSONDecodeError:
            return {}
    def safe_loads(s: str):
        res = {}
        try:
            res = json.loads(s)
        except:
            print(f'exception while loading: {s}')
            res = {}

        return res
    final_json = {
        "creator_tag": res_cr,
        "community_tags": safe_json_load(res_cm, data),
        "risk_evaluation": safe_loads(res_risk),
        "sources": safe_loads(res_src)
    }
    print(final_json)
    return final_json
    # return res_cr, res_cm

import pandas as pd
from datetime import datetime, timedelta


def get_responses_today(mail, worksheet):
    records = worksheet.get_all_records()
    df = pd.DataFrame(records)

    if df.empty:
        print('nothing here')
        return False

    # Google Forms timestamp column name
    timestamp_col = "Timestamp"
    # Convert to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Date range
    now = datetime.now()
    start_of_tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_of_today = now.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    try:
        filtered_df = df[
            (df[timestamp_col] >= start_of_today) &
            (df['Email'] == mail)
        ]
        return len(filtered_df)>0
    except:
        print('exception while filtering')
        return False




if __name__=='__main__':
    # with open('data/config.json') as jf:
    #     keys = json.load(jf)
    
    # yt_key = keys['YT_API_KEY']
    # gpt_key = keys['GPT_API_KEY']
    # video_id = 'SgSnz7kW-Ko'

    # client = AsyncOpenAI(api_key=gpt_key)
    # res = asyncio.run(get_aigc_tag(client, video_id, yt_key))

    # print(res)
    # responses = get_responses_yesterday_to_today('xyz@xyz.com')
    # print(f"Responses found: {len(responses)}")
    # print(responses)
    print('nothing')