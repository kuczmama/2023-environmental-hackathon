import logging
from PyPDF2 import PdfReader

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http

document_store = InMemoryDocumentStore(use_bm25=True)
from haystack.utils import fetch_archive_from_http

doc_dir = "/Users/mark/develop/answrly-accounting/data/fasab"

# reader = PdfReader('2022_ FASAB_ Handbook.pdf')
# file = open(doc_dir + "/fasab.txt", "a")
# print(f"enumerating {len(reader.pages)} pages")
# for i, page in enumerate(reader.pages):
#     text = page.extract_text()
#     if text:
#         file.write(text)
# file.close()

fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
    output_dir=doc_dir,
)
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)
from haystack.nodes import BM25Retriever

retriever = BM25Retriever(document_store=document_store)
from haystack.nodes import FARMReader

reader = FARMReader(
    model_name_or_path="google/tapas-base-finetuned-wtq", use_gpu=True, from_tf=True
)
from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)
prediction = pipe.run(
    query="What is Activity-based Costing?",
    params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
)
from pprint import pprint

pprint(prediction)
from haystack.utils import print_answers

print_answers(
    prediction, details="minimum"  ## Choose from `minimum`, `medium`, and `all`
)
