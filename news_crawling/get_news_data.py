import requests
from bs4 import BeautifulSoup
import math
import pandas
import re
import datetime
from tqdm import tqdm
import json
import re
#from nltk.tokenize import sent_tokenize


def get_news_data(keyword, query_list):
    
    news_dic={}
        
    print("query list : ", query_list)

    url = "https://search.naver.com/search.naver?"

    params = {
        "where": 'news',
        # 페이지네이션 값
        "start": 0,
    }
    
    today = datetime.date.today()
    
    news_index = 0

    for query in tqdm(query_list):
        params["query"]=query

        raw = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, params=params)
        news_list_html = BeautifulSoup(raw.text, "html.parser")

        articles = news_list_html.select("ul.list_news > li")
        title = articles[0].select_one("a.news_tit").text

        for i in range(0, 10):
            title = articles[i].select_one("a.news_tit").text
            try:
                if articles[i].select("a.info")[1].text =="네이버뉴스":
                    news_link = articles[i].select("a.info")[1].get('href')
                    press = articles[i].select_one("a.info.press").text
                    
                    if press[-2:]=="선정":
                        press = press[:-2]

                    linked_news = requests.get(news_link, headers={'User-Agent': 'Mozilla/5.0'})
                    linked_news_html = BeautifulSoup(linked_news.text, "html.parser")

                    #content = linked_news_html.find_all("div",{"id":'articleBodyContents'})[0].text.replace("\n", "").split(". ")
                    content = linked_news_html.find_all("div",{"id":'articleBodyContents'})[0].text.replace("\n", "")
                    content = content.replace("// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}", "")
                    split=re.compile('(?<=[^0-9])\.')
                    content=split.split(content)
                    content = [re.sub(r'\[[^]]*\]', '', x) for x in content]
                    """
                    숫자.숫자"를 제외한 모든 온점에서 split하는 것으로 고도화"""

                    index_ = today.isoformat().replace("-", "")+str(news_index).zfill(2)

                    news_dic[index_]={"query":params["query"], "title":title, "url":news_link, "press":press, "content":content}
                    news_index +=1
            except:
                pass

    with open('news_data/{today}_{keyword}.json'.format(today=today.isoformat(), keyword=keyword), 'w', encoding='utf-8') as make_file:

        json.dump(news_dic, make_file, indent="\t")
        

    return news_dic
        
