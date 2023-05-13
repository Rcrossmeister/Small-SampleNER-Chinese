import json

def bio2json(bio_data):
    text = ''
    entities = {}
    entity_type = ''
    start = -1
    end = -1

    for i in range(len(bio_data)):
        word, label = bio_data[i].split()
        if label.startswith('B-'):
            if start != -1:
                if entity_type not in entities:
                    entities[entity_type] = {}
                if text[start:end] not in entities[entity_type]:
                    entities[entity_type][text[start:end]] = []
                entities[entity_type][text[start:end]].append([start, end])
            entity_type = label[2:]
            start = i
            end = i + 1
        elif label.startswith('I-'):
            if entity_type != label[2:]:
                if entity_type not in entities:
                    entities[entity_type] = {}
                if text[start:end] not in entities[entity_type]:
                    entities[entity_type][text[start:end]] = []
                entities[entity_type][text[start:end]].append([start, end])
                entity_type = label[2:]
                start = i
            else:
                end = i + 1
        else:
            if start != -1:
                if entity_type not in entities:
                    entities[entity_type] = {}
                if text[start:end] not in entities[entity_type]:
                    entities[entity_type][text[start:end]] = []
                entities[entity_type][text[start:end]].append([start, end])
                start = -1
                end = -1
                entity_type = ''

        text += word

    if start != -1:
        if entity_type not in entities:
            entities[entity_type] = {}
        if text[start:end] not in entities[entity_type]:
            entities[entity_type][text[start:end]] = []
        entities[entity_type][text[start:end]].append([start, end])

    labels = {}
    for entity_type in entities:
        for entity in entities[entity_type]:
            if entity_type not in labels:
                labels[entity_type] = {}
            labels[entity_type][entity] = entities[entity_type][entity]


    return json.dumps({"text": text, "label": labels}, ensure_ascii=False)

def read_bio_save_json(src_path):
    json_buffer = []
    temp_buffer = []
    label2id = {}
    label2id_buffer =set()
    with open(src_path,'r',encoding='utf-8') as f:
        for item in f.readlines():
            item = item.strip()
            if item!="":
                temp_buffer.append(item)
                label2id_buffer.add(item.split()[1])
            else:
                json_buffer.append(bio2json(temp_buffer))
                temp_buffer=[]

    label2id_buffer = list(label2id_buffer)
    label2id_buffer.sort()
    label2id_buffer.remove('O')
    label2id_buffer.insert(0,'O')
    label2id = {idx:item for idx,item in enumerate(label2id_buffer)}

    dot_idx = len(src_path)-src_path[::-1].index('.')-1
    des_path = src_path[:dot_idx]+'.json'
    
    with open(des_path,'w',encoding='utf-8') as f:
        for item in json_buffer:
            f.write(item+'\n')
    
    label2id_path = des_path.rsplit('/',1)[0]+'/label2id.json'
    with open(label2id_path,'w',encoding='utf-8') as f:
        json.dump(label2id, f)
        
    return des_path,label2id_path
