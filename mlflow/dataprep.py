import mlflow
import click
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import subprocess
import ast


@click.command(help="Pull records from Pubmed for given search term"
                    "Search term should follow the query format")
@click.option("--json_in", default='pmed.json')
@click.option("--save_df", default='dfprocessed.p')
@click.option("--inputfile", default=1,
              help="Whether to use the input.txt")
def dfprep(json_in, save_df, inputfile):
    if inputfile == 1:
        with open("input.txt", "r") as f:
            para = ast.literal_eval(f.read())
        json_in = para['json_in']
        save_df = para['save_df']
    with mlflow.start_run() as mlrun:
        print(subprocess.getoutput("python -m spacy download en_core_web_sm"))
        artpd = pd.read_json(json_in, orient='index', convert_dates=False, convert_axes=False)
        artpda = artpd[artpd.abstract.notnull()]
        artpda.index = pd.Series(artpda.index).apply(lambda x: x[0:8])
        artpdak = artpda[artpda.keywords.str.len() > 0]
        dataf = pd.DataFrame(index=artpdak.index, columns=['SRC', 'TRG', 'keywords', 'Extracted', 'abskey'])
        dataf.loc[:, 'SRC'] = artpdak.title + ' ' + artpdak.abstract
        dataf.loc[:, 'keywords'] = artpdak.keywords
        svoc = spacy.load("en_core_web_sm")
        matcher = PhraseMatcher(svoc.vocab, attr="LOWER")
        for pmid in dataf.index:
            t0 = dataf.loc[pmid]
            patterns = [svoc.make_doc(str(name)) for name in t0.keywords]
            matcher.add("Names", None, *patterns)
            doc = svoc(t0.SRC)
            t1 = ['O']*(len(doc))
            matched = []
            matn = 0
            for _, start, end in matcher(doc):
                t1[start] = 'B'
                t1[start+1:end] = 'I'*(end-start-1)
                if str(doc[start:end]).lower() not in matched:
                    matn = matn+1
                    matched.append(str(doc[start:end]).lower())
            abskw = []
            for x in t0.keywords:
                if x.lower() not in matched:
                    abskw.append(x)
            dataf.loc[pmid, 'TRG'] = ' '.join([t for t in t1])
            dataf.loc[pmid, 'Extracted'] = matn
            dataf.loc[pmid, 'abskey'] = abskw
            matcher.remove("Names")
        dataf.to_pickle(save_df)


if __name__ == '__main__':
    dfprep()
