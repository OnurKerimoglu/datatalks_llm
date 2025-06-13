import os

from dotenv import load_dotenv
from openai import OpenAI
import requests

from src import minsearch


prompt_template = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database. 
Use only the facts from the CONTEXT when answering the question.
If the CONTEXT doesn't contain the answer, output NONE.

Question: {question}

CONTEXT:
{context}
"""


class RAGwithMinsearch:
    def __init__(
            self,
            retriever='minisearch',
            model='gpt-4o'):
        
        self.model = model
        
        documents = self.get_documents()
        if retriever == 'minisearch':
            self.index = self.index_documents_with_minisearch(documents)
        else:
            raise ValueError(f"Invalid retriever: {retriever}")
        
        if model in ['gpt-4o']:
            print('Creating llm client: OpenAI gpt-4o')
            load_dotenv()
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Invalid model: {model}")

    def get_documents(self):
        print ("Getting documents...")
        docs_url = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/01-intro/documents.json'
        docs_response = requests.get(docs_url)
        docs_raw = docs_response.json()
        documents = []
        for course_dict in docs_raw:
            for doc in course_dict['documents']:
                doc['course'] = course_dict['course']
                documents.append(doc)
        return documents

    def index_documents_with_minisearch(self, documents):
        print('Indexing with minisearch...')
        index = minsearch.Index(
            text_fields=["question", "text", "section"],
            keyword_fields=["course"]
        )
        index.fit(documents)
        return index

    def search(self, query, index, course_filter=None):
        print(f'Retrieving relevant content for question: {query}')
        boost = {'question': 3.0, 'section': 0.5}
        if course_filter is None:
            print('Not course filters applied')
            filter_dict={}
        else:
            print(f'Applying course filter: {course_filter}')
            filter_dict={'course': course_filter}
        results = index.search(
            query=query,
            filter_dict=filter_dict,
            boost_dict=boost,
            num_results=5
        )
        return results

    def create_prompt(self, query, results):
        context = ""
        for doc in results:
            doc_context = f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
            context = context + doc_context
        context.strip()
        prompt = prompt_template.format(question=query, context=context).strip()
        # print(f'prompt: {prompt} ')
        return context, prompt

    def refine_print_response(self, response, context):
        resp_text = response.choices[0].message.content
        if resp_text == 'NONE':
            refined_response = 'I couldn\'t find the answer to that question within the provided context.\n'
            refined_response += 'Context:\n\n'
            refined_response += context
        else:
            refined_response = 'Answer:\n\n'
            refined_response += resp_text
        print(refined_response)
        # return refined_response
    
    def answer(
            self,
            question,
            course_filter=None):
        results = self.search(question, self.index, course_filter)
        context, prompt = self.create_prompt(question, results)
        if self.model in ['gpt-4o']:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'user', 
                        'content': prompt}
                ]
            )
            self.refine_print_response(response, context)
            # return refined_response
        else:
            raise ValueError(f"Unknown model: {self.model}")


if __name__ == '__main__':
    rag = RAGwithMinsearch(
        retriever='minisearch',
        model='gpt-4o'
    )
    rag.answer(
    question='The course has already started. Can I still join?',
    course_filter = 'data-engineering-zoomcamp'
    )
