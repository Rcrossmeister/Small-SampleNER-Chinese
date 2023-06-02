from TextPreprocess_English import TextPreprocess_English
import argparse
import random

def PreProcess(args):
    text_list = []
    label_list = []
    raw_data_path = args.raw_data
    obj_data_path = args.obj_path + '/Edited_Data.txt'

    with open(raw_data_path) as raw_data_file:
        raw_text = raw_data_file.readlines()
        raw_data_file.close()

    for element in raw_text:
        last_tab_index = element.rfind('\t')
        raw_t = element[:last_tab_index]
        label = element[last_tab_index+1:]

        tp = TextPreprocess_English(raw_t)
        t = tp.TextPreprocess()

        text_list.append(t)
        label_list.append(label)

    with open(obj_data_path, 'w') as obj_data_file:
        for edited_text, label in zip(text_list, label_list):

            line = edited_text + "\t" + label
            obj_data_file.write(line)

        if not args.shuffle:
            obj_data_file.close()
        else:
            return obj_data_file

def Shuffle(args):

    shuffle_data = args.shuffle_data
    obj_file = args.obj_path

    with open(shuffle_data) as shuffle_file:
        text = shuffle_file.readlines()
        shuffle_file.close()

    random.shuffle(text)
    number = len(text)

    with open(obj_file + '/train.txt', 'w') as train_file:
        train_file.writelines(text[:int(0.8*number)])
        train_file.close()

    with open(obj_file + '/dev.txt', 'w') as dev_file:
        dev_file.writelines(text[int(0.8*number):int(0.9*number)])
        dev_file.close()

    with open(obj_file + '/test.txt', 'w') as test_file:
        test_file.writelines(text[int(0.9*number):])
        test_file.close()

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--preprocess", default=False, action='store_true')
    parser.add_argument("--shuffle", default=False, action='store_true')
    parser.add_argument("--raw_data", default='', action='store_true')
    parser.add_argument("--shuffle_data", default='/home/hzj/NLP1/SentimentAnalysis/IMDB_0.5/Final_IMDB_0.5.txt',
                        action='store_true')
    parser.add_argument("--obj_path", default='/home/hzj/NLP1/SentimentAnalysis/IMDB_0.5/', action='store_true')

    args = parser.parse_args()

    if args.preprocess:
        PreProcess(args)
        print('PreProcess Finished!')
    if args.shuffle:
        Shuffle(args)
        print('Shuffle Finish!')

if __name__ == '__main__':
    main()