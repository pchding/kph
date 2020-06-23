from pymed import PubMed
import mlflow
from collections import defaultdict
import json
import ast
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import torch
from torch import nn
import time
import torchtext
import numpy as np
from torchcrf import CRF
import subprocess
import mlflow.pytorch
from elasticsearch import Elasticsearch
from elasticsearch import helpers


@click.option("--search_term", default='test',
              help="https://pubmed.ncbi.nlm.nih.gov/advanced/")
@click.option("--max_records", default=10000,
              help="Limit the data size to run comfortably.")
@click.option("--json_out", default='pmed.json',
              help="Name of the output JSON file")
@click.option("--embvec", default=1)
@click.option("--embvecache", default='/home/pding/Documents/glove/')
@click.option("--val_ratio", default=0.2)
@click.option("--rnnsize", default=128)
@click.option("--batchsize", default=310)
@click.option("--lr", default=0.01)
@click.option("--weight_decay", default=1e-5)
@click.option("--n_epochs", default=15)
@click.option("--model_save", default='model0.pt')
@click.option("--es", default=1)
@click.option("--inputfile", default=1,
              help="Whether to use the input.txt")
def mainpipe(inputfile, search_term, max_records, json_out, embvec, embvecache, val_ratio, rnnsize, batchsize,lr, weight_decay, n_epochs, model_save, es):
    if inputfile == 1:
        with open("input.txt", "r") as f:
            para = ast.literal_eval(f.read())
        search_term = para['search_term']
        max_records = para['max_records']
        embvec = para['embvec']
        embvecache = para['embvecache']
        val_ratio = para['val_ratio']
        rnnsize = para['rnnsize']
        batchsize = para['batchsize']
        lr = para['lr']
        weight_decay = para['weight_decay']
        n_epochs = para['n_epochs']
        model_save = para['model_save']
    if embvec == 1:
        embvec = torchtext.vocab.GloVe(name='840B', dim=300, cache=embvecache)
        use_pretrained = True
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
        print(subprocess.getoutput("python -m spacy download en_core_web_sm"))
        artpd = pd.DataFrame.from_dict(pp, orient='index')
        artpda = artpd[artpd.abstract.notnull()].copy()
        artpda = artpda[artpd.title.notnull()]
#        artpda.index = pd.Series(artpda.index).apply(lambda x: x[0:8])
        artpdak = artpda[artpda.keywords.str.len() > 0].copy()
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
        datatrain = dataf[dataf['Extracted']>=3].copy()
        datatest = dataf[dataf['Extracted']<3].copy()
        # separate train and validate
        dtrain = datatrain.loc[:,['SRC','TRG']]
        dtraink = datatrain.loc[:,['SRC','TRG','keywords']]
        seed = 250
        idx = np.arange(datatrain.shape[0])
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.shuffle(idx)
        val_size = int(len(idx) * val_ratio)
        df_train = dtrain.iloc[idx[val_size:], :]
        df_val = dtrain.iloc[idx[:val_size], :]
        df_val_k = dtraink.iloc[idx[:val_size], :]
        df_test = datatest.loc[:,['SRC','TRG']]
        dtraink = datatrain.loc[:,['SRC','TRG','keywords']]
        df_val_k = dtraink.iloc[idx[:val_size], :]
        # Load original dataset
        datai = artpda.copy()
        datai = datai[datai.abstract.notnull()]
        datai = datai[datai.title.notnull()]
        datai = datai.replace('\n',' ', regex=True)
        datai = datai.replace('\t',' ', regex=True)
        dataiu = datai.loc[datai.keywords.str.len() ==0]
        dataik = datai.loc[datai.keywords.str.len() >0]
        dataiu['SRC'] = dataiu.title + ' '+ dataiu.abstract
        tokenizertrg = lambda x: x.split()

        def tokenizersrc(text):  # create a tokenizer function
            return [tok.text for tok in svoc.tokenizer(text)]
        def safe_value(field_val):
            return field_val if not pd.isna(field_val) else "Other"
        def safe_year(field_val):
            return field_val if not pd.isna(field_val) else 1900
        TEXT = torchtext.data.Field(init_token='<bos>', eos_token='<eos>', sequential=True, lower=False)
        LABEL = torchtext.data.Field(init_token='<bos>', eos_token='<eos>', sequential=True, unk_token=None)
        fields = [('text', TEXT), ('label', LABEL)]
        device = 'cuda'
        train_examples = read_data(df_train, fields, tokenizersrc, tokenizertrg)
        valid_examples = read_data(df_val, fields, tokenizersrc, tokenizertrg)
        # Load the pre-trained embeddings that come with the torchtext library.
        if use_pretrained:
            print('We are using pre-trained word embeddings.')
            TEXT.build_vocab(train_examples, vectors=embvec)
        else: 
            print('We are training word embeddings from scratch.')
            TEXT.build_vocab(train_examples, max_size=5000)
        LABEL.build_vocab(train_examples)
        # Create one of the models defined above.
        #self.model = RNNTagger(self.TEXT, self.LABEL, emb_dim=300, rnn_size=128, update_pretrained=False)
        model0 = RNNCRFTagger(TEXT, LABEL, rnnsize, emb_dim=300,  update_pretrained=False)

        model0.to(device)
        optimizer = torch.optim.Adam(model0.parameters(), lr=lr, weight_decay=weight_decay)
        train(train_examples, valid_examples, embvec, TEXT, LABEL, device, model0, batchsize, optimizer,n_epochs)
        out2 = evaltest2(df_val, df_val_k, model0, tokenizersrc, fields, device)
        ttp3 = kphperct(df_val_k, out2,svoc)
        mlflow.log_param("epochs", n_epochs)
        mlflow.pytorch.save_model(model0, model_save)
        mlflow.log_metric("extraction_rate", ttp3.mean())
        augout = evaltest2(dataiu,model0, tokenizersrc, fields, device)
        klist = kphext2(dataiu.SRC,augout,svoc)
        for i in range(len(dataiu.index)):
            dataiu.iloc[i,2].extend(list(set(klist[i])))
        output = pd.concat([dataik,dataiu], join="inner")
        output.to_json('/home/pding/OneDrive/kph/MSaug.json', orient='index')
        if es == 1:
            output['journal'] = output['journal'].apply(safe_value)
            output['conclusions'] = output['conclusions'].apply(safe_value)
            output['pubdate'] = output['pubdate'].apply(safe_year)
            output['PMID'] = output.index
            test_server = [{'host':'127.0.0.1','port':9200}]
            es = Elasticsearch(test_server,http_compress=True)
            use_these_keys = ['PMID', 'title', 'abstract', 'keywords','authors','pubdate']

            def filterKeys(document):
                return {key: document[key] for key in use_these_keys }
            
            def doc_generator(df):
                df_iter = df.iterrows()
                for index, document in df_iter:
                    try:
                        yield {
                            "_index": 'ms',
                            "_source": filterKeys(document),
                        }
                    except StopIteration:
                        return
            helpers.bulk(es, doc_generator(output))
        print(ttp3.mean())



class RNNCRFTagger(nn.Module):
    
    def __init__(self, text_field, label_field, rnn_size, emb_dim, update_pretrained=False):
        super().__init__()
        
        voc_size = len(text_field.vocab)
        self.n_labels = len(label_field.vocab)       
        
        self.embedding = nn.Embedding(voc_size, emb_dim)
        if text_field.vocab.vectors is not None:
            self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
                                                       requires_grad=update_pretrained)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=rnn_size, 
                          bidirectional=True, num_layers=1)

        self.top_layer = nn.Linear(2*rnn_size, self.n_labels)
 
        self.pad_word_id = text_field.vocab.stoi[text_field.pad_token]
        self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]
    
        self.crf = CRF(self.n_labels)
        
    def compute_outputs(self, sentences):
        embedded = self.embedding(sentences)
        rnn_out, _ = self.rnn(embedded)
        out = self.top_layer(rnn_out)
        return out
                
    def forward(self, sentences, labels):
        # Compute the outputs of the lower layers, which will be used as emission
        # scores for the CRF.
        scores = self.compute_outputs(sentences)
        mask0 = sentences != self.pad_word_id
        mask = mask0.byte()
        # We return the loss value. The CRF returns the log likelihood, but we return 
        # the *negative* log likelihood as the loss value.            
        # PyTorch's optimizers *minimize* the loss, while we want to *maximize* the
        # log likelihood.
        return -self.crf(scores, labels, mask=mask)

    def predict(self, sentences):
        # Compute the emission scores, as above.
        scores = self.compute_outputs(sentences)
        mask0 = sentences != self.pad_word_id
        mask = mask0.byte()
        # Apply the Viterbi algorithm to get the predictions. This implementation returns
        # the result as a list of lists (not a tensor), corresponding to a matrix
        # of shape (n_sentences, max_len).
        return self.crf.decode(scores, mask=mask)

def train(train_examples, valid_examples, embvec, TEXT, LABEL, device, model, batch_size, optimizer, n_epochs):


    # Count the number of words and sentences.
    n_tokens_train = 0
    n_sentences_train = 0
    for ex in train_examples:
        n_tokens_train += len(ex.text) + 2
        n_sentences_train += 1
    n_tokens_valid = 0       
    for ex in valid_examples:
        n_tokens_valid += len(ex.text)


    
    n_batches = np.ceil(n_sentences_train / batch_size)

    mean_n_tokens = n_tokens_train / n_batches

    train_iterator = torchtext.data.BucketIterator(
        train_examples,
        device=device,
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False,
        train=True,
        sort=True)

    valid_iterator = torchtext.data.BucketIterator(
        valid_examples,
        device=device,
        batch_size=64,
        sort_key=lambda x: len(x.text),
        repeat=False,
        train=False,
        sort=True)

    train_batches = list(train_iterator)
    valid_batches = list(valid_iterator)

    n_labels = len(LABEL.vocab)

    history = defaultdict(list)  

   

    for i in range(1, n_epochs + 1):

        t0 = time.time()

        loss_sum = 0

        model.train()
        for batch in train_batches:

            # Compute the output and loss.
            loss = model(batch.text, batch.label) / mean_n_tokens

            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_loss = loss_sum / n_batches
        history['train_loss'].append(train_loss)
        if i % 1 == 0:
            print(f'Epoch {i}: train loss = {train_loss:.4f}')
    mlflow.log_metric("train_loss", history['train_loss'])


def evaltest2(df_val, df_val_k, model, tokenizersrc,fields,device):
    # This method applies the trained model to a list of sentences.
    examples = []
    for sen in df_val.SRC:
        words = tokenizersrc(sen)
        labels = ['O']*len(words) # placeholder
        examples.append(torchtext.data.Example.fromlist([words, labels], fields))
    dataset = torchtext.data.Dataset(examples, fields)

    iterator = torchtext.data.Iterator(
        dataset,
        device=device,
        batch_size=300,
        repeat=False,
        train=False,
        sort=False)

    # Apply the trained model to all batches.
    out = []
    model.eval()
    for batch in iterator:
        # Call the model's predict method. This returns a list of NumPy matrix
        # containing the integer-encoded tags for each sentence.
        predicted = model.predict(batch.text)

        # Convert the integer-encoded tags to tag strings.
        #for tokens, pred_sen in zip(sentences, predicted):
        for pred_sen in predicted:
            out.append([LABEL.vocab.itos[pred_id] for  pred_id in  pred_sen[1:-1]])
    return out



def kphext2(sentences,tags,svoc):
    kph = []
    for i in range(len(sentences)):
        s0 = svoc.tokenizer(sentences[i])
        s1 = [tok.text for tok in s0]
        t1 = tags[i]
        k1 = []
        for j in range(len(s1)):
            start = j
            if t1[j] == 'B':
                sti = 0
                stop = j+1
                while sti == 0:
                    try:
                        kt = str(t1[stop])
                        if kt == 'I':
                            stop = stop+1
                        else:
                            k2 = str(s0[start:stop])
                            k1.append(k2)
                            sti = 1
                    except(IndexError):
                        k2 = str(s0[start:stop])
                        k1.append(k2)
                        sti = 1
                k2 = str(s1[j])
        kph.append(k1)
    return kph


def read_data(df_train, datafields, tokenizersrc, tokenizertrg):
    examples = []
    words = []
    labels = []
    for pmid in df_train.index:
        words = tokenizersrc(df_train.loc[pmid,'SRC'])
        labels = tokenizertrg(df_train.loc[pmid,'TRG'])
        examples.append(torchtext.data.Example.fromlist([words, labels], datafields))
    return torchtext.data.Dataset(examples, datafields)

def tagperct(df_val,out):
    tp = np.empty(len(out))
    for i in range(len(df_val.index)):
        trg = tokenizertrg(df_val.iloc[i,1])
        total = 0
        for x in trg:
            if x != 'O':
                total = total+1
        matched = 0
        for j in range(total):
            if trg[j] != 'O':
                if trg[j]== out[i][j]:
                    matched = matched + 1
        p = matched/total
        tp[i] = p
    return tp


def kphperct(df_val_k,out,svoc):
    tp = np.empty(len(out))
    for i in range(len(df_val_k.index)):
        ktrg = df_val_k.iloc[i,2]
        pred = kphext2([df_val_k.iloc[i,0]],[out[i]],svoc)
        k = 0
        for kp in ktrg:
            if str(kp).lower() in [str(x).lower() for x in pred[0]]:
                k = k+1
        tp[i] =  k/df_val_k.iloc[i,3]
    return tp


if __name__ == '__main__':
    mainpipe()
