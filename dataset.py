import pandas as pd
import json
import numpy as np

def load_dataset(dataset_name):
    if dataset_name == 'cities':
        csv_file_path = './dataset/cities.csv'
        data = pd.read_csv(csv_file_path)
        # Filter the statements based on the label and get the respective lists
        lista = data[data['label'] == 1]['statement'].tolist()
        listb = data[data['label'] == 0]['statement'].tolist()
        
        list_true = []
        list_false = []
        for i in range(len(lista)):
            list_true.append(lista[i])
        label_true = np.ones(len(list_true))
        
        for i in range(len(listb)):
            list_false.append(listb[i])
        label_false = np.zeros(len(list_false))

        prompts = []
        list_total = list_true + list_false
        label_total = np.concatenate((label_true, label_false))

        for item in list_total:
            prompt = get_prompt(dataset_name, item)
            prompts.append(prompt)

        return prompts, label_total

def get_prompt(dataset_name, prompt):
    if dataset_name == 'cities':
        return prompt + " Judge the statement is True or False."