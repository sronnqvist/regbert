from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import TransfoXLModel, TransfoXLTokenizer
from xl_extension import TransfoXLForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import confusion_matrix
import collections
import torch
import csv
import os
import numpy as np
import scipy


### Load data
DATA_PATH="/home/samuel/data/CORE-final"
csv.field_size_limit(1000000)


def read_tsv(filename, label_index=0, sentence_index=1):
    with open(filename) as tsv_file:
        for line in csv.reader(tsv_file, delimiter='\t'):
            try:
                yield (line[label_index], line[sentence_index])
            except:
                print(line)
                raise

data = {}
for dataset in ['train', 'dev', 'test']:
    print("Preparing", dataset)
    data[dataset] = [x for x in read_tsv(os.path.join(DATA_PATH, dataset+'.tsv'), sentence_index=2)]

"""
label_list = set()
for d in data.values():
    label_list = label_list.union([label for label, _ in d])

label_list = list(label_list)
"""

top_level_labels = set("NA OP IN ID HI IP LY SP OTHER".split())
def normalize_label(label):
    return ' '.join(sorted([l if l in top_level_labels else l.lower() for l in label.split()]))


label_counter = collections.defaultdict(lambda: 0)
for d in data.values():
    for label, _ in d:
        label_counter[normalize_label(label)] += 1

label_list = [l for l,c in label_counter.items()]# if c > 1]
label2idx = {l:i for i,l in enumerate(label_list)}

print("Labels:", ', '.join(list(label2idx.keys())))

### Tokenize
print("Loading BERT tokenizer...")
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
from tokenizers import BertWordPieceTokenizer
#tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True, )
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

MAX_LEN = 1000
BATCH_SIZE = 1
# Tokenize all of the sentences and map the tokens to thier word IDs.
data_loader = {}

for dataset in data:
    print("  Tokenizing", dataset)
    input_ids = []
    attn_masks = []
    labels = []
    for i, (label, sent) in enumerate(data[dataset][:1000]):
        l = normalize_label(label)
        if l not in label2idx:
            print("        Skipping label", label)
            continue
        if i % 1000 == 0:
            print("    sample %d" % (i))
        tok = tokenizer.encode_plus(sent, add_space_before_punct_symbol=True)#, add_special_tokens = True, max_length=MAX_LEN)#["input_ids"]
        sent_ids = tok['input_ids']
        attn_mask = tok['attention_mask']
        #token_types = tok['token_type_ids']
        sent_ids = sent_ids[:MAX_LEN] + [0]*(MAX_LEN-len(sent_ids))
        attn_mask = attn_mask[:MAX_LEN]+[0]*(MAX_LEN-len(attn_mask))
        input_ids.append(sent_ids)
        attn_masks.append(attn_mask)
        labels.append(label2idx[l])
    input_ids = torch.tensor(input_ids).to('cuda')
    attn_masks = torch.tensor(attn_masks).to('cuda')
    labels = torch.tensor(labels).to('cuda')
    prep_data = TensorDataset(input_ids, attn_masks, labels)
    sampler = SequentialSampler(prep_data)
    data_loader[dataset] = DataLoader(prep_data, sampler=sampler, batch_size=BATCH_SIZE)


### Training
print("Loading BERT model...")
"""model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = len(label_list), # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)"""
#model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
model = TransfoXLForSequenceClassification.from_pretrained('transfo-xl-wt103', num_labels=len(label2idx))

device = torch.device("cuda")
model.cuda()

#model.from_pretrained('output')

optimizer = AdamW(model.parameters(),
                  lr = 5e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


epochs = 4
total_steps = len(data['train']) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


def evaluate(model, dataset='dev'):
    ### Validation
    model.eval()
    eval_correct, eval_all = 0, 0
    all_preds = []
    all_true = []
    # Evaluate data for one epoch
    for batch in data_loader[dataset]:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids)
            #                token_type_ids=None,
            #                attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        eval_correct += np.sum(pred_flat == labels_flat)
        eval_all += len(labels_flat)
        all_preds += list(pred_flat)
        all_true += list(labels_flat)

        #print(np.max(scipy.special.softmax(logits),axis=1))

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_correct/eval_all*100))
    print(confusion_matrix(all_true, all_preds))


print("Training...")
for ep in range(epochs):
    print("Epoch", ep)
    total_loss = 0
    model.train() #Mode
    # For each batch of training data...
    for step, batch in enumerate(data_loader['train']):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(data_loader['train'])))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(b_input_ids)
        #            token_type_ids=None,
        #            attention_mask=b_input_mask,
        #            labels=b_labels)
        loss = outputs[0]
        total_loss += loss.sum().item()
        # Perform a backward pass to calculate the gradients.
        _ = loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    evaluate(model)
    avg_train_loss = total_loss / len(data_loader['train'])
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

#evaluate(model)
"""
output_dir = 'output3'
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
"""
