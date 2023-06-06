import json

def bio2json(bio_data):
    """
    change single bio type's sentence and label to json {"text":"XXX","label(just tag)":{entity:[start:end]}}  |  e.g. {"text": "我会邀请许惠佑先生届时一同来访。", "label": {"PER": {"许惠佑": [[4, 6]]}}}
    """
    text = ''
    entities = {}
    entity_type = ''
    start = -1
    end = -1

    for i in range(len(bio_data)):
        word, label = bio_data[i][0],bio_data[i][2:]
        if label.startswith('B-'):
            if start != -1:
                if entity_type not in entities:
                    entities[entity_type] = {}
                if text[start:end]:
                    if text[start:end] not in entities[entity_type]:
                        entities[entity_type][text[start:end]] = []
                    entities[entity_type][text[start:end]].append([start, end-1])
            entity_type = label[2:]
            start = i
            end = i + 1
        elif label.startswith('I-'):
            if entity_type != label[2:]:
                if entity_type not in entities:
                    entities[entity_type] = {}
                if text[start:end]:
                    if text[start:end] not in entities[entity_type]:
                        entities[entity_type][text[start:end]] = []
                    entities[entity_type][text[start:end]].append([start, end-1])
                entity_type = label[2:]
                start = i
            else:
                end = i + 1
        else:
            if start != -1:
                if entity_type not in entities:
                    entities[entity_type] = {}
                if text[start:end]:
                    if text[start:end] not in entities[entity_type]:
                        entities[entity_type][text[start:end]] = []
                    entities[entity_type][text[start:end]].append([start, end-1])
                start = -1
                end = -1
                entity_type = ''

        text += word
        
    if start != -1:
        if entity_type not in entities:
            entities[entity_type] = {}
        if text[start:end]:
            if text[start:end] not in entities[entity_type]:
                entities[entity_type][text[start:end]] = []
            entities[entity_type][text[start:end]].append([start, end-1])

    labels = {}
    for entity_type in entities:
        for entity in entities[entity_type]:
            if entity_type not in labels:
                labels[entity_type] = {}
            labels[entity_type][entity] = entities[entity_type][entity]
    
    return json.dumps({"text": text, "label": labels}, ensure_ascii=False)

def read_bio_save_json(src_path):
    """
    read XXX.txt to save XXX.json and label2id.json {"label":id} | e.g. {"O": 0, "B-LOC": 1, "B-ORG": 2, "B-PER": 3, "I-LOC": 4, "I-ORG": 5, "I-PER": 6, "<START>": 7, "<STOP>": 8}
    """
    json_buffer = []
    temp_buffer = []
    label2id = {}
    label2id_buffer =set()
    
    with open(src_path,'r',encoding='utf-8') as f:
        for (idx,item) in enumerate(f.readlines()):
            item = item.strip()
            if item!="":
                temp_buffer.append(item)
                #print(idx,item)
                label2id_buffer.add(item[2:])
            else:
                json_buffer.append(bio2json(temp_buffer))
                temp_buffer=[]

    label2id_buffer = list(label2id_buffer)
    label2id_buffer.sort()
    label2id_buffer.remove('O')
    label2id_buffer.insert(0,'O')
    label2id = {item:idx for idx,item in enumerate(label2id_buffer)}
    
    label2id["<START>"] = len(label2id)
    label2id["<STOP>"] = len(label2id)

    dot_idx = len(src_path)-src_path[::-1].index('.')-1
    des_path = src_path[:dot_idx]+'.json'
    
    with open(des_path,'w',encoding='utf-8') as f:
        for item in json_buffer:
            f.write(str(item)+"\n")
            
            
    
    label2id_path = des_path.rsplit('/',1)[0]+'/label2id.json'
    with open(label2id_path,'w',encoding='utf-8') as f:
        json.dump(label2id, f)
        
    return des_path,label2id_path

if __name__ == "__main__":
    for item in ['./test.txt','./val.txt','./train.txt']:
        read_bio_save_json(item)
