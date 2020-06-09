from pymed import PubMed
import mlflow
import click
from collections import defaultdict
import json
import ast


@click.command(help="Pull records from Pubmed for given search term"
                    "Search term should follow the query format")
@click.option("--search_term", default='test',
              help="https://pubmed.ncbi.nlm.nih.gov/advanced/")
@click.option("--max_records", default=10000,
              help="Limit the data size to run comfortably.")
@click.option("--save_json", default='pmed.json',
              help="Name of the output JSON file")
@click.option("--inputfile", default=1,
              help="Whether to use the input.txt")
def querysave(search_term, max_records, save_json, inputfile):
    if inputfile == 1:
        with open("input.txt", "r") as f:
            para = ast.literal_eval(f.read())
        search_term = para['search_term']
        max_records = para['max_records']
        save_json = para['save_json']
    with mlflow.start_run() as mlrun:
        pubmed = PubMed(tool="AlphabetH", email="pcding@outlook.com")
        query = search_term
        results = pubmed.query(query, max_results=max_records)
        pp = defaultdict(lambda: defaultdict(dict))
        for art in results:
            pmed = art.pubmed_id
            try:
                pp[pmed]['title'] = art.title
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['abstract'] = art.abstract
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['abstract'] = pp[pmed]['abstract'] + art.conclusions
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['abstract'] = pp[pmed]['abstract'] + art.methods
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['abstract'] = pp[pmed]['abstract'] + art.results
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['keywords'] = art.keywords
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['authors']= art.authors
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['journal'] = art.journal
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['pubdate'] = str(art.publication_date.year)
            except(AttributeError, TypeError):
                pass
            try:
                pp[pmed]['conclusions'] = art.conclusions
            except(AttributeError, TypeError):
                pass
        with open(save_json, 'w') as fp:
            json.dump(pp, fp)


if __name__ == '__main__':
    querysave()
