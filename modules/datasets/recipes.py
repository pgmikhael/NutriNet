from modules.utils.shared import register_object
import numpy as np
import os, pickle
import warnings, traceback
import json
from tqdm import tqdm
from modules.datasets.base import Abstract_Dataset
from collections import Counter, OrderedDict, defaultdict
import pdb
from random import Random
from typing import List

METADATA_FILENAMES = {"recipe_generation": "recipes_with_nutritional_info.json" } #_normalized_nutrition.json"}

@register_object("recipes", 'dataset')
class Recipes(Abstract_Dataset):
    """
    Abstract Dataset Object for all Datasets. All datasets have some metadata
    property associated with them, a create_dataset method, a task, a check
    label, a skip label and get label function. The __getitem__ class is the
    most important method as it returns the item to be batched instead of
    loading any large data into memory.
    """
    def create_dataset(self, split_group: str) -> list:
        """
        Creates and returns the dataset from the sample ids and labels in the json.
        This will include in the dataset any metadata that exists in the json file as well.

        :param split_group: ['train'|'dev'|'test']
        :return: dataset
        """
        if self.args.assign_data_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        # normalize:

        nutr = np.stack([ np.array(list(recipe['nutr_values_per100g'].values())) for recipe in self.metadata_json if recipe['split'] == 'train'])
        self.args.nutritional_mean = np.mean(nutr, axis = 0)
        self.args.nutritional_std = np.std(nutr, axis = 0)
        self.args.nutritional_max = np.max(nutr, axis = 0)

        self.args.norm_nutritional_medians = np.median( nutr / self.args.nutritional_max, axis = 0)

        dataset = []
        for recipe in tqdm(self.metadata_json):
            sample_id, split = recipe['id'], recipe['split']

            if not split == split_group:
                continue
            
            if self.skip_sample(recipe, None):
                continue 

            # ingredients
            ingredients = self.get_input_dict(recipe, 'ingredients').replace(',', '')
            instructions = self.get_input_dict(recipe, 'instructions')

            y = self.get_label(recipe)

            sample = {
                    'name': recipe['title'],
                    'sample_id': sample_id,
                    'ingredients': ingredients,
                    'instructions': instructions,
                    'nutrition': y, # maybe need to edit format of this
                    'webpage' : recipe['url']
                }

            dataset.append(sample)

        return dataset
    
    def skip_sample(self, recipe: dict, image_dict: dict) -> bool:
        """
        Tests whether or not to skip a sample. Used to skip samples that will break the pipeline,
        for example if a label is missing.

        :param well: this is the metadata dictionary for a specific well (sample)
        :param image_dict:
        :return:
        """
        if recipe['id'] is None:
            return True
        
        if recipe['ingredients'] is None or len(recipe['ingredients']) == 0:
            return True
        
        if recipe['instructions'] is None or len(recipe['instructions']) == 0:
            return True
        
        if recipe['nutr_values_per100g'] is None or len(recipe['nutr_values_per100g']) == 0:
            return True

        if not self.args.assign_data_splits and recipe['split'] is None:
            return True
    
        return False 
    
    def assign_splits(self,meta: List[dict]) -> None:
        """
        Assigns splits to each sample in the metadata_json
        :param meta: the parsed metadata json
        :return: None
        """
        recipe_ids = sorted(list(set([m['id'] for m in meta if not m['id'] is None  ])))
        recipe_ids2split = {}
        for recipe in recipe_ids:
            recipe_ids2split[recipe] = np.random.choice(['train','dev','test'], p = self.args.split_probs)
        for idx in range(len(meta)):
            if meta[idx]['id'] is not None:
                meta[idx]['split'] = recipe_ids2split[meta[idx]['id'] ]
            else:
                meta[idx]['split'] = 'none'

    def get_input_dict(self, case: dict, key) -> str:
        """
        Generic method that can be used to get most data from metadata json
        :param case: the specific sample dictionary from the metadata json
        :return: ingredients string
        """
        keys = ""
        for text in case[key]:
            keys += list(text.values())[0] + ' '
        return keys

    def get_label(self, case: dict) -> float:
        """
        Gets and returns the label for the sample_id in the case dict
        :param case: the specific sample dictionary from the metadata json
        :return: label of case's sample
        """
        return np.array([v for v in case['nutr_values_per100g'].values()])/self.args.nutritional_max
    
    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        summary = ''
        summary += 'DATASET SIZE: {}\n'.format(len(self.dataset))
        # if self.args.predict_multi_target:
        #     class_dist = np.sum([d['y'] for d in self.dataset], axis = 0)
        # else:
        #     class_dist = Counter([d['y'] for d in self.dataset])
        #     class_dist = {k: v for k, v in sorted(class_dist.items(), key=lambda item: item[0])}
        # summary += 'CLASS DIST: {}'.format(class_dist)
        return summary
    
    def get_majority_baseline(self, train_data) -> dict:
        """
        Gets the stats for a basic model that just predicts the majority baseline.
        :param train_data: training data from which to get these stats
        :return: dict of stats
        """
        pass
    
    @property
    def LOAD_FAIL_MSG(self):
        return "Failed to load input: {}\nException: {}"

    @property
    def DATASET_ITEM_KEYS(self):
        return ['y']
    
    @property
    def supported_tasks(self):
        return ['recipe_generation']

    @property
    def task(self):
        return "recipe_generation"

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.task]


    def get_summary_statement(self, split_group):
        '''
        Prints a summary statement
        '''
        print('\n')
        print("{} DATASET {} CREATED.\n{}".format(split_group.upper(), self.args.dataset.upper(), self.SUMMARY_STATEMENT))


    @staticmethod
    def set_args(args):
        """
        Sets any args particular to the dataset.
        Warning: this will reset the args for all subsequent steps. Args is global to the system and mutating it
        can have side effects.
        """
        args.num_classes = 6

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input = self.input_loader.load_input(sample)
        try:
            
            item = {
                    'x': input['input'],
                    'y': sample['nutrition'],
                    'sample_id': sample['sample_id']
                    }
            
            for key in ['x_mask', 'x_length']: # 0s where sequence is padded, length of sequence
                if key in input:
                    item[key] = input[key]
                
            for key in self.DATASET_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item 

        except Exception:
            warnings.warn(self.LOAD_FAIL_MSG.format(sample['sample_id'], traceback.print_exc()))

@register_object("recipes_binary", 'dataset')
class RecipesBinary(Recipes):
    def get_label(self, case: dict) -> float:
        """
        Gets and returns the label for the sample_id in the case dict
        :param case: the specific sample dictionary from the metadata json
        :return: label of case's sample
        """
        return np.array([v for v in case['nutr_values_per100g'].values()])/self.args.nutritional_max > self.args.norm_nutritional_medians
