from src.rag_with_minsearch import RAGwithMinsearch

def example_1(rag):
    rag.answer(
        question='The course has already started. Can I still join?',
        course_filter = 'data-engineering-zoomcamp'
    )

def example_2(rag):
    rag.answer(
        question='How can I run spark in standalone mode on windows?',
        course_filter = 'data-engineering-zoomcamp'
    )

def example_3(rag):
    rag.answer(
        question='How can I run spark in standalone mode on windows?',
        course_filter = 'machine-learning-zoomcamp'
    )

if __name__ == '__main__':
    rag = RAGwithMinsearch(
        retriever='minisearch',
        model='gpt-4o'
    )
    example_3(rag)
