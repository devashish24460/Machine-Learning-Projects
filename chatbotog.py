
import requests
import json
import os
import logging
import shutil
import time
from flask_cors import CORS
from flask import Flask, request, jsonify
from haystack.nodes import PromptNode, BM25Retriever, PromptTemplate, AnswerParser
from haystack.agents.memory import ConversationSummaryMemory
from haystack.utils import convert_files_to_docs, clean_wiki_text
from haystack.agents import AgentStep, Agent    
from haystack.pipelines import Pipeline
from haystack.utils import print_answers
from haystack.agents.base import Agent, ToolsManager
#from haystack.document_stores import InMemoryDocumentStore                                                                
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores import InMemoryDocumentStore, ElasticsearchDocumentStore
from haystack.nodes.retriever.base import BaseRetriever
import fitz



app = Flask(__name__)
CORS(app)
app.config["input"] = "D:\pdf"
app.config["host"] = "0.0.0.0"
app.config["port"] = "9200"







# document_store = InMemoryDocumentStore(use_bm25=True)
#document_store = ElasticsearchDocumentStore(host=app.config["host"], port=int(app.config["port"]), index='index',use_bm25=True)
# Assuming you get the index value from the request or some other source






@app.route('/get_index', methods=['GET','POST'])
def get_index():
    indexlist = os.listdir(app.config["input"])
    return json.dumps(
        {'status':'Susccess', 'all_indexes':indexlist})

@app.route('/delete_index', methods=['GET','POST'])
def delete_index():
    indexog = request.form['index']
    index = indexog.lower()
    index = index.replace(" ", "")
    parent_dir = app.config["input"]
    removepath = os.path.join(parent_dir, index)
    # os.rmdir(removepath)

    try:
        shutil.rmtree(removepath)
        print("Index '% s' has been removed successfully" % indexog)
    except OSError as error:
        print(removepath)
        print("Index '% s' can not be removed" % indexog)
    
    indexlist = os.listdir(app.config["input"])

    return json.dumps(
        {'status':'Susccess', 'all_indexes':indexlist})


@app.route('/update_document', methods=['POST'])
def update_document():
    """Return a the url of the index document."""
    if request.files:
        # index is the target document where queries need to sent.
        indexog = request.form['index']
        index = indexog.lower()
        index = index.replace(" ", "")
        # uploaded document for target source
        doc = request.files["doc"]
        
        document_store = InMemoryDocumentStore(use_bm25=True)

        #document_store = ElasticsearchDocumentStore(host=app.config["host"], port=int(app.config["port"]), index=index)



        print("reached here after saving in document store\n", document_store)

        directory = str(index)
        parent_dir = app.config["input"]
        path = os.path.join(parent_dir, directory)
        final_path = os.path.join(path, doc.filename)

        isExist = os.path.exists(path)
        if isExist == False:
            os.mkdir(path)
            doc.save(final_path)
            print('the final_path is',final_path)
        else:
            doc.save(final_path)

        docs = convert_files_to_docs(final_path, clean_func=clean_wiki_text, split_paragraphs=True)
        document_store.write_documents(docs)

        indexlist = os.listdir(app.config["input"])
        print('the indexlist is',indexlist)
        return json.dumps(
            {'status':'Susccess', 'all_indexes':indexlist})
    else:
        return json.dumps({'status':'Failed','message': 'No file uploaded', 'result': []})

logging.basicConfig(level=logging.DEBUG)

@app.route('/qna_pretrain', methods=['POST'])
def qna_pretrain():
    
    start_time = time.time()
    
    """Return the n answers."""

    question = request.form['question']
    # index is the target document where queries need to sent.
    # indexes = request.form['index']
   
    indexog = request.form['index']
    index = indexog.lower()
    indexes = index.replace(" ", "")

    # uploaded document for target source
    top_k = request.form['top_answer_count']

    document_store = InMemoryDocumentStore(use_bm25=True)

    directory = str(indexes)

    print("this is the index from input user", indexes)

    parent_dir = app.config["input"]
    print("this is parent dir",parent_dir)
    path = os.path.join(parent_dir, directory)
    print('the path is $$$$$$$$$$$$$$$',path)

    docs = convert_files_to_docs(path, split_paragraphs=True)

    print(docs)

    document_store.write_documents(docs)
    retriever = BM25Retriever(document_store=document_store, top_k= int(top_k))
    retriever.debug = True

    prompt_template = PromptTemplate(
        prompt="""
        In the following conversation, a human user interacts with an AI Agent. The human user poses {query}, and the AI Agent reads the documents refered by the indexes to provide detailed answers.
        If the human user starts casual conversation, the AI Agent must respond politely
        The index refers to documents in elastic search document store under special catoegories.
        Provide Detailed answer to the question truthfully based solely on the given documents under given index.
        The AI Agent may politely request the human user for additional information, clarification, or context
        
        Documents:{join(documents)}
        Question:{query}
        Answer:
        """,    
        output_parser=AnswerParser(),
    )
    index = indexes

    prompt_node = PromptNode(
        model_name_or_path="gpt-3.5-turbo", api_key="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", default_prompt_template=prompt_template
    )

    
    generative_pipeline = Pipeline()
    generative_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    generative_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])





















    ##get the retrieved context

    response = generative_pipeline.run(question)
    retrievelog = response["_debug"]

    # print(retrievelog)

    retrieved_content = ''
    alldocs = retrievelog['retriever']['output']['documents']
    print("got here with all docs \n",alldocs)
    for doc in alldocs:
        retrieved_content += doc.content
    
    print("this is retrievecontent \n",retrieved_content)

    finalanswer = response['answers'][0]

    end_time = time.time()
    response_time_ms = int((end_time - start_time) * 1000)

    # prompttxt = finalanswer
    botanswer = finalanswer.answer
    prompttext = finalanswer.meta['prompt']
    # print("this is ans \n", ans)
    print("this is meta \n", prompttext)
    # answer_dict = finalanswer.to_dict()
    # print("this is answer dict \n", answer_dict)


    def estimate_tokens(text):
        word_count = len(text.split())
        char_count = len(text)
        tokens_count_word_est = word_count / 0.75
        tokens_count_char_est = char_count / 4.0
        
        output = max(tokens_count_word_est, tokens_count_char_est)
        return int(output)

    finres = [x.to_dict() for x in response["answers"]]
    print("this is finres\n",finres)

    # Replace answer with botanswer
    return jsonify({'status':'success','Question':'question', 'Answer': finalanswer})


if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0', port=7778)

             