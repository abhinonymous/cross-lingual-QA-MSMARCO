from googletrans import Translator
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from typing import List
import re
import json
import subprocess
import numpy as np
import string
import zmq
import pickle
class marcoSearch:
    def __init__(self, languages: List[str]) -> None:
        self.translator = Translator()
        self.languages = languages
        if 'en' not in self.languages:
            self.languages.append('en')

    def translate_string(self, list_of_strings: List[str] , destination_language: str) -> List[str]:
        if type(list_of_strings) != list:
            list_of_strings = [list_of_strings]
        translations = self.translator.translate(list_of_strings, dest = destination_language)
        translated_strings = [s.text for s in translations]
        #translated_strings = translations.text
        return(translated_strings)

    def search_single_query(self, query: str) -> dict:
        url_dict = {}
        for lang in self.languages:
            url_dict[lang] = []
            curr_query = self.translate_string(query, destination_language = lang)
            url_dict[lang].extend(search(query= curr_query[0], tld='com', lang=lang, num= 10, stop= 10))
        return(url_dict)

    def parse_page(self, url: str, translate: bool = True) -> List[str]:
        page = requests.get(url = url)
        soup = BeautifulSoup(page.text, 'html.parser')
        texts = [v.get_text() for v in soup.find_all('p')]
        texts = list(map(self.clean_text, texts))
        texts = list(filter(lambda x: x != '', texts))
        if translate:
            translated_texts = self.translate_string(texts, destination_language = 'en')
            return(translated_texts)
        else:
            return(texts)

    def search_query(self, query: str) -> tuple:
        print('Retrieving URLs')
        url_dict = self.search_single_query(query)
        docs = {}
        docs['en'] = [self.parse_page(url = u, translate = False) for u in url_dict['en']]
        #print(docs)
        print('Translating pages')
        for lang in url_dict:
            if lang != 'en':
                docs[lang] = []
                docs[lang].extend([self.parse_page(url = u) for u in url_dict[lang]])
        return((docs, url_dict))

    @staticmethod
    def clean_text(html_text: str) -> str:
        clean_text = re.sub('<.*>', '', html_text)
        exclusion_set = set(string.punctuation) - {'.', '?', ',', '!'}
        clean_text = [v for v in list(clean_text) if v not in exclusion_set]
        clean_text = ''.join(clean_text)
        return(clean_text)

    @staticmethod
    def get_marco_vectors(strings_list: List[str], api_key: str) -> List[dict]:
        standard_query = 'curl https://api.msturing.org/gen/encode -H "Ocp-Apim-Subscription-Key: {}" --data \'{"queries": ["JHU Hackathon starts now!", "Microsoft Bing Turing team is here to help you"]}\''.format(api_key)
        strings = '" , "'.join(strings_list)
        strings = '["' + strings[0:1000] + '"]'
        query = re.sub('\[.*\]', strings, standard_query)
        #query = query + strings + '}\ '.rstrip() + '\''
        #print(query)
        result = subprocess.check_output(query, shell=True)
        result_json = json.loads(result)
        return(result_json)

    def get_sorted_tuples(self, query: str) -> List[tuple]:
        docs, url_dict = self.search_query(query)
        query_vector = np.array(self.get_marco_vectors([query])[0]['vector'])
        output_list = []
        for lang in self.languages:
            count = 0
            for website in docs[lang]:
                curr_string = ' '.join(website)
                try:
                    vector = np.array(self.get_marco_vectors([curr_string])[0]['vector'])
                    score = np.dot(query_vector, vector)
                    score /= np.linalg.norm(vector)*np.linalg.norm(query_vector)
                except:
                    score = 0
                #print(score)
                cur_url = url_dict[lang]
                curr_tuple = (url_dict[lang][count], score, curr_string)
                output_list.append(curr_tuple)
                count += 1
        output_list.sort(key = lambda x: x[1], reverse = True)
        return(output_list)


def run(api_key="fb169a275fd64c10b13a9e21900f8f72"):
    #   d52c40fc4c2c4fcfb768ce18a7d1bafc
    query = 'st petersburg mayor'
    languages = ['en', 'ru']
    ms = marcoSearch(languages=languages)
    output_docs = ms.get_sorted_tuples(query = query)
    output_string = ''
    for tup in output_docs:
        output_string += tup[2]
    query = re.sub('\t', ' ', query)
    output_string = re.sub('\t', ' ', output_string)
    emit_tuple = (query, output_string)
    

    context = zmq.Context()
    
    #  Socket to talk to qa server
    print("Connecting to Question answering Server")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:8889")
    payload = pickle.dumps(emit_tuple)
    socket.send(payload)

    #  Get the reply.
    message = socket.recv()
    _, answer_str = pickle.loads(message)
    print("Received reply for query:%s \t %s ]" % (query, answer_str))


if __name__ == "__main__":
    run()

