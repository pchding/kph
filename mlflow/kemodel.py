import mlflow
import torch
from torch import nn
import time
import torchtext
import click
import numpy as np
import pandas as pd
import spacy
from torchcrf import CRF
import ast
import subprocess
import mlflow.pytorch


@click.command(help="Convert JSON to PD. Tag key phrases")
@click.option("--df_in", default='dfprocessed.p')
@click.option("--embvec", default=1)
@click.option("--embvecache", default='/home/pding/Documents/glove/')
@click.option("--val_ratio", default=0.2)
@click.option("--rnnsize", default=128)
@click.option("--batchsize", default=310)
@click.option("--lr", default=0.01)
@click.option("--weight_decay", default=1e-5)
@click.option("--n_epochs", default=15)
@click.option("--model_save", default='model0.pt')
@click.option("--inputfile", default=1,
              help="Whether to use the input.txt")
def ketraintest(inputfile, df_in, embvec, embvecache, val_ratio, rnnsize, batchsize,lr, weight_decay, n_epochs, model_save):
    if inputfile == 1:
        with open("input.txt", "r") as f:
            para = ast.literal_eval(f.read())
        df_in = para['df_in']
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
    subprocess.getoutput("python -m spacy download en_core_web_sm")
    svoc = spacy.load("en_core_web_sm")
    datao = pd.read_pickle(df_in)
    datatrain = datao[datao['Extracted']>=3]
    datatest = datao[datao['Extracted']<3]
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
    tokenizertrg = lambda x: x.split()

    def tokenizersrc(text):  # create a tokenizer function
        return [tok.text for tok in svoc.tokenizer(text)]
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
    with mlflow.start_run() as mlrun:
        train(train_examples, valid_examples, embvec, TEXT, LABEL, device, model0, batchsize, optimizer,n_epochs)
        out2 = evaltest2(df_val, df_val_k, model0, tokenizersrc, fields, device)
        ttp3 = kphperct(df_val_k, out2,svoc)
        mlflow.log_param("epochs", n_epochs)
        mlflow.pytorch.save_model(model0, model_save)
        mlflow.log_metric("extraction_rate", ttp3.mean())
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
        batch_size=1,
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
        for tokens, pred_sen in zip(batch.text.view(1,-1), predicted):
            out.append([LABEL.vocab.itos[pred_id] for _, pred_id in zip(tokens, pred_sen[1:])])
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

                            sti =1

                    except(IndexError):

                        k2 = s0[start:stop]

                        k1.append(k2)

                        sti =1

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
    ketraintest()
