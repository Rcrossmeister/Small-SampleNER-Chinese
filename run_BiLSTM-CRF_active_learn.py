import json
import torch
import argparse
import torch.nn as nn
import numpy as np
import math
import random
from torch import optim
from pathlib import Path
from model import NERModel,BERTNERModel
from dataset_loader import DatasetLoader
from progressbar import ProgressBar
from ner_metrics import SeqEntityScore
from data_processor import CluenerProcessor
from lr_scheduler import ReduceLROnPlateau
import lossnet as lossnet
from sampler import SubsetSequentialSampler
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from utils_ner import get_entities
from common import (init_logger,
                    logger,
                    json_to_text,
                    load_model,
                    AverageMeter,
                    seed_everything)




def train(args,model,processor):
    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=True,
                                 vocab = processor.vocab,label2id = args.label2id)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    best_f1 = 0
    for epoch in range(1, 1 + args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(train_loader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        print(" ")
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_log, class_info = evaluate(args,model,processor)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        scheduler.epoch_step(logs['eval_f1'], epoch)
        if logs['eval_f1'] > best_f1:
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to disk.")
            best_f1 = logs['eval_f1']
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'arch': args.arch, 'state_dict': model_stat_dict}
            model_path = args.output_dir / 'best-model.bin'
            model_path = args.output_dir / args.save_model_name
            torch.save(state, str(model_path))
            print("Eval Entity Score: ")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)

def evaluate(args,model,processor):
    if not args.do_test:
        eval_dataset = load_and_cache_examples(args,processor, data_type='dev')
    else:
        eval_dataset = load_and_cache_examples(args,processor, data_type='test')
    
    eval_dataloader = DatasetLoader(data=eval_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=False,
                                 vocab=processor.vocab, label2id=args.label2id)
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            eval_loss.update(val=loss.item(), n=input_ids.size(0))
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=tags, label_paths=target)
            pbar(step=step)
    print(" ")
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info

def predict(args,model,processor):
    # model_path = args.output_dir / 'best-model.bin'
    model_path = args.output_dir / args.save_model_name
    model = load_model(model, model_path=str(model_path))
    test_data = []
    with open(str(args.data_dir / "test.json"), 'r') as f:
        idx = 0
        for line in f:
            json_d = {}
            line = json.loads(line.strip())
            text = line['text']
            words = list(text)
            labels = ['O'] * len(words)
            json_d['id'] = idx
            json_d['context'] = " ".join(words)
            json_d['tag'] = " ".join(labels)
            json_d['raw_context'] = "".join(words)
            idx += 1
            test_data.append(json_d)
    pbar = ProgressBar(n_total=len(test_data))
    results = []
    for step, line in enumerate(test_data):
        token_a = line['context'].split(" ")
        input_ids = [processor.vocab.to_index(w) for w in token_a]
        input_mask = [1] * len(token_a)
        input_lens = [len(token_a)]
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            input_lens = torch.tensor([input_lens], dtype=torch.long)
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            features = model.forward_loss(input_ids, input_mask, input_lens, input_tags=None)
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
        label_entities = get_entities(tags[0], args.id2label)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(tags[0])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step=step)
    print(" ")
    output_predic_file = str(args.output_dir / "test_prediction.json")
    output_submit_file = str(args.output_dir / "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(str(args.data_dir / 'test.json'), 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['id'] = x['id']
        json_d['label'] = {}
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)

def load_and_cache_examples(args,processor, data_type='train'):
    # Load data features from cache or dataset file
    cached_examples_file = args.data_dir / 'cached_crf-{}_{}_{}'.format(
        data_type,
        args.arch,
        str(args.task_name))
    # if cached_examples_file.exists():
    #     logger.info("Loading features from cached file %s", cached_examples_file)
    #     examples = torch.load(cached_examples_file)
    #     print(examples)
    # else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if data_type == 'train':
        examples = processor.get_train_examples()
        # print(examples)
    elif data_type == 'dev':
        examples = processor.get_dev_examples()
    elif data_type == 'test':
        examples = processor.get_test_examples()
    # logger.info("Saving features into cached file %s", cached_examples_file)
    # torch.save(examples, str(cached_examples_file))
    return examples

def active_method_train():
    pass

def active_method_eval():
    pass

def active_train(args,model,processor,active_train_dataloader,count_query):

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    best_f1 = 0
    for epoch in range(1, 1 + args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        pbar = ProgressBar(n_total=len(active_train_dataloader), desc='Training')
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(active_train_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        print(" ")
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_log, class_info = evaluate(args,model,processor)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        scheduler.epoch_step(logs['eval_f1'], epoch)
        if logs['eval_f1'] > best_f1:
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to disk.")
            best_f1 = logs['eval_f1']
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'arch': args.arch, 'state_dict': model_stat_dict}
            model_path = args.output_dir / (f'count_query_{count_query}_'+args.save_model_name)
            torch.save(state, str(model_path))
            print("Eval Entity Score: ")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
   
    loss = torch.sum(torch.clamp(margin - one * input, min=0))
    loss = loss / input.size(0)
    return loss

def loss_train(args,model,processor,loss_module,active_train_dataloader,count_query):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    best_f1 = 0
    WEIGHT = 1#loss_train临时变量
    for epoch in range(1, 1 + args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        pbar = ProgressBar(n_total=len(active_train_dataloader), desc='Training')
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(active_train_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features_list,features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags,Flag_loss_train=True)
            print(loss.size())
            loss = loss.view(loss.size(0))
            backbone_loss = torch.sum(loss) / loss.size(0)

            pred_loss = loss_module(features_list)
            pred_loss = pred_loss.view(pred_loss.size(0))
            module_loss = LossPredLoss(pred_loss, loss)
            final_loss = backbone_loss +WEIGHT*module_loss
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': final_loss.item()})
            train_loss.update(final_loss.item(), n=1)
        print(" ")
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_log, class_info = evaluate(args,model,processor)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        scheduler.epoch_step(logs['eval_f1'], epoch)
        if logs['eval_f1'] > best_f1:
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to disk.")
            best_f1 = logs['eval_f1']
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'arch': args.arch, 'state_dict': model_stat_dict}
            model_path = args.output_dir / 'best-model.bin'
            torch.save(state, str(model_path))
            print("Eval Entity Score: ")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)

def active_evaluate(args,model,processor,unlabeled_dataloader):
    pbar = ProgressBar(n_total=len(unlabeled_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    raw_logits = []
    turncate_list = []
    with torch.no_grad():
        for step, batch in enumerate(unlabeled_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            eval_loss.update(val=loss.item(), n=input_ids.size(0))
            tag, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=tag, label_paths=target)
            pbar(step=step)
            logits = F.softmax(features, dim=2)
            logits = logits.detach().cpu().numpy()
            logits = np.max(logits,axis=2)
            for logit,length in zip(logits,input_lens):
                raw_logits.append(logit)
                turncate_list.append(length)
            # print(raw_logits)
    # print(raw_logits[0].shape)
    # print(raw_logits[0])          
    print(" ")
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info,raw_logits,turncate_list

def loss_get_uncertainty(args,model,processor,unlabeled_dataloader,loss_module,):
    pbar = ProgressBar(n_total=len(unlabeled_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    uncertainty = torch.tensor([]).to(args.device)
    with torch.no_grad():
        for step, batch in enumerate(unlabeled_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)

            features_list,features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags,Flag_loss_train=True)
            pred_loss = loss_module(features_list) 
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0) 
            pbar(step=step)
    return uncertainty

def random_sample(args,model,processor,eval_dataloader):
    return 

def least_confidence_sample(args,model,processor,eval_dataloader):
    uncertainty = []
    _,_,raw_prediction, turncate_list = active_evaluate(args,model,processor,eval_dataloader) 
    for i, sentence in enumerate(raw_prediction):
        uncertainty.append(np.sum(sentence[:turncate_list[i]])/turncate_list[i])
    # print(uncertainty[0])
    return uncertainty



def loss(args,model,processor,loss_module,eval_dataloader):
    uncertainty = loss_get_uncertainty(args,model,processor,eval_dataloader,loss_module) 
    return uncertainty.cpu()


def active_learn(args,model,processor,selected_method,num_initial_ratio =0.2,num_each_query_add_ratio = 0.2,num_query=5,train_only_new_data=False):
    assert selected_method
    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    
    total_len = len(train_dataset)
    indices = list(range(len(train_dataset)))
    random.seed(123)
    random.shuffle(indices)
    labeled_set = indices[:math.floor(num_initial_ratio*total_len)]
    unlabeled_set = indices[math.floor(num_initial_ratio*total_len):]
    print(f'labeled_set : {len(labeled_set)}')
    print(f'unlabeled_set : {len(unlabeled_set)}')
    if selected_method =="loss":
        loss_module = lossnet.LossNet(feature_sizes=[128,768,768,33],num_channels=[50, 50, 50, 50]).to(args.device)
        active_method = loss
    elif selected_method =='lc':
        active_method = least_confidence_sample
    elif selected_method =='random':
        active_method = random_sample
    for count_query in range(num_query):
        print(f"--- Current query : {count_query} ----")
        if train_only_new_data: # here just 
            if count_query == 0:
                train_subset = labeled_set
            else: 
                train_subset = labeled_set[math.floor((num_initial_ratio+num_each_query_add_ratio*(count_query-1))*total_len):math.floor((num_initial_ratio+num_each_query_add_ratio*(count_query))*total_len)]
        else:
            train_subset = labeled_set
        print(f'train_set : {len(train_subset)}')
        active_train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=True,
                                 vocab = processor.vocab,label2id = args.label2id,sampler ="SubsetSequentialSampler",sampler_list =train_subset)
        
        if selected_method =='loss':
            loss_train(args,model,processor,loss_module,active_train_loader,count_query)
        else:
            active_train(args,model,processor,active_train_loader,count_query)
        

        subset = unlabeled_set[:total_len]
        active_eval_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=True,
                                 vocab = processor.vocab,label2id = args.label2id,sampler ="RandomSampler",sampler_list = subset)
        

        if selected_method =='loss':
            uncertainty = active_method(args, model,processor,loss_module,active_eval_loader)
        elif selected_method =="random":
            argsort_list = range(len(unlabeled_set)) #just unchanged
        else:
            uncertainty = active_method(args, model,processor,active_eval_loader)

        
        if selected_method !="random":
            argsort_list = np.argsort(uncertainty)
        
        print(f'argsort_list : {len(argsort_list)}')
        labeled_set += list(torch.tensor(subset)[argsort_list][:math.floor(num_each_query_add_ratio*total_len)].numpy())
        unlabeled_set = list(torch.tensor(subset)[argsort_list][math.floor(num_each_query_add_ratio*total_len):].numpy())
        print(f'labeled_set : {len(labeled_set)}')
        print(f'unlabeled_set : {len(unlabeled_set)}')


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default='./dataset/cluener', type=str)
    parser.add_argument("--output_dir", default='./output', action='store_true')
    parser.add_argument("--save_model_name",default='best-model.bin', type=str)

    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument("--do_eval", default=False, action='store_true')
    parser.add_argument("--do_test", default=False, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true')

    parser.add_argument("--markup", default='bios', type=str, choices=['bios', 'bio'])
    parser.add_argument("--arch",default='bilstm_crf',type=str)
    parser.add_argument("--learning_rate",default=0.001,type=float)
    parser.add_argument("--seed",default=1234,type=int)
    parser.add_argument("--gpu",default='0',type=str)
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--batch_size",default=64,type=int)
    parser.add_argument("--embedding_size",default=128,type=int)
    parser.add_argument("--hidden_size",default=384,type=int)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--task_name", type=str, default='ner')



    parser.add_argument("--do_active_train", default=False, action='store_true')
    parser.add_argument("--active_method", default="random", type=str)
    parser

    args = parser.parse_args()
    # args.data_dir = config.data_dir
    print(args.data_dir)
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)

    if not args.output_dir.exists():
        args.output_dir.mkdir()
    args.output_dir = args.output_dir / '{}'.format(args.arch)
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    init_logger(log_file=str(args.output_dir / '{}-{}.log'.format(args.arch, args.task_name)))
    seed_everything(args.seed)
    if args.gpu!='':
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    label2id_path = args.data_dir / 'label2id.json'

    with open(label2id_path, 'r') as f:
        label2id = json.load(f)

    args.id2label = {i: label for i, label in enumerate(label2id)}
    args.label2id = label2id
    processor = CluenerProcessor(args,data_dir=args.data_dir)
    processor.get_vocab()
    model = NERModel(vocab_size=len(processor.vocab), embedding_size=args.embedding_size,
                     hidden_size=args.hidden_size,device=args.device,label2id=args.label2id)
    
    # model = BERTNERModel(vocab_size=len(processor.vocab), embedding_size=args.embedding_size,
    #                  hidden_size=args.hidden_size,device=args.device,label2id=args.label2id)
    model.to(args.device)
    if args.do_train:
        train(args,model,processor)
    if args.do_eval:
        model_path = args.output_dir / args.save_model_name
        model = load_model(model, model_path=str(model_path))
        evaluate(args,model,processor)
    if args.do_test:
        model_path = args.output_dir / args.save_model_name
        model = load_model(model, model_path=str(model_path))
        eval_log,_=evaluate(args,model,processor)
        logs = dict( **eval_log)
        show_info = f'\npredict result: - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
    if args.do_predict:
        eval_log, _ = evaluate(args,model,processor)
    if args.do_active_train:
        active_learn(args,model,processor,selected_method=args.active_method)
if __name__ == "__main__":
    main()
