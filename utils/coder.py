import torch
import pickle
from clip import clip
import torch.nn.functional as F
import json
from tqdm import tqdm

def get_templates(dataset, text_type, class_name=None, att=None, ana=None, syno=None, ovo=None, other_class_name=None, task_type='zero-shot'):
    if text_type == "name":
        templates = get_name_templates(dataset, class_name)
    elif text_type == 'syno':
        templates = get_name_templates(dataset, syno)
    elif text_type == "att":
        templates = get_att_templates(dataset, class_name, att, task_type)
    elif text_type == "ana":
        templates = get_ana_templates(dataset, class_name, ana, task_type)
    elif text_type == "ovo":
        templates = get_ovo_templates(dataset, class_name, ovo, other_class_name)

    return templates

def get_name_templates(dataset, class_name):
    if dataset == 'eurosat':
        templates = [
            f'a photo of {class_name}, satellite domain.'
        ]
    elif dataset == 'food-101':
        templates = [
            f'a photo of {class_name}, a type of food.'
        ]
    elif dataset == 'oxford_flowers':
        templates = [
            f'a photo of a {class_name}, a type of flower.'
        ]
    elif dataset == 'Aircraft':
        templates = [
            f'a photo of a {class_name}, a type of aircraft.'
        ]
    elif dataset == 'oxford_pets' or dataset == 'oxford-pet':
        templates = [
            f'a photo of a {class_name}, a type of pet.'
        ]
    elif dataset == 'ucf101':
        templates = [
            f'a photo of a person doing {class_name}.'
        ]
    else:
        templates = [
            f'a photo of {class_name}'
        ]   
    return templates


def get_att_templates(dataset, class_name, att, task_type):
    if dataset == 'eurosat':
        templates = [
            f"A satellite photo of " + class_name + " which has " + att + "."
        ]
    elif dataset == 'oxford-pet' and task_type == 'zero-shot':
        templates = [
            "A pet photo of " + class_name + " which has " + att + "."
        ]
    elif dataset == 'food-101':
        templates = [
            "A food photo of " + class_name + " which has " + att + "."
        ]
    elif dataset == 'DTD' and task_type == 'zero-shot':
        templates = [
            'The texture of ' + class_name + ' is characterized by its ' + att + ' feature.'
        ]
    else:
        templates = [
            class_name + " which has " + att
        ]
        
    return templates

def get_ana_templates(dataset, class_name, analogous_class, task_type):
    if dataset == 'eurosat':
        templates = [
            f"A satellite photo of " + f"a {class_name} similar to {analogous_class}."
        ]
    elif dataset == 'oxford-pet' and task_type == 'zero-shot':
        templates = [
            "A pet photo of " + f"a {class_name} similar to {analogous_class}."
        ]
    elif dataset == 'food-101':
        templates = [
            "A food photo of " + f"a {class_name} similar to {analogous_class}."
        ]
    elif dataset == 'DTD' and task_type == 'zero-shot':
        templates = [
            f'The texture of {class_name} bears a likeness to that of {analogous_class} .'
        ]
    else:
        templates = [
            f"a {class_name} similar to {analogous_class}."
        ]

    return templates

def get_ovo_templates(dataset, class_name1, ovo_text, class_name2):
    if dataset == 'eurosat':
        templates = [
            f"The unique utilization of {ovo_text} in satellite photos makes {class_name1} different from {class_name2}.",
            f"Differences in {ovo_text} representation separate {class_name1} in satellite photos from {class_name2}.",
            f"Contrasting {class_name1} and {class_name2} in satellite photos reveals notable differences in the representation of {ovo_text}.",
            f"{class_name1} and {class_name2} showcase distinct visual features when it comes to {ovo_text} in satellite imagery.",
            f"The comparison of {class_name1} and {class_name2} in satellite photos underscores variations in the depiction of {ovo_text}.",
            f"In the context of satellite imagery, differences emerge between {class_name1} and {class_name2} in how they capture {ovo_text}.",
            f"Analyzing satellite photos highlights disparities in the portrayal of {ovo_text} between {class_name1} and {class_name2}.",
            f"{class_name1} and {class_name2} exhibit contrasting visual interpretations when representing {ovo_text} in satellite imagery."
        ]

    elif dataset == 'oxford-pet':
        templates = [
            f'A {class_name1}, a species of pet, can be distinguished from a {class_name2}, a species of pet, by the characteristics of {ovo_text}', 
            f'Because of {ovo_text}, a {class_name1}, a kind of pet, is different from a {class_name2}, a kind of pet.', 
            f'Due to their {ovo_text}, a {class_name1}, a pet type, displays a different demeanor compared to a {class_name2}, another pet type.', 
            f'Because of their {ovo_text}, a {class_name1}, a kind of pet, differs significantly from a {class_name2}, a kind of pet.'
        ]

    elif dataset == 'food-101':
        templates = [
            f'While both are popular dishes, {class_name1} is notably different from {class_name2} in that {ovo_text}.',
            f'While both are popular snacks, {class_name1} is notably different from {class_name2} in that {ovo_text}.',
            f'While both are popular dessert, {class_name1} is notably different from {class_name2} in that {ovo_text}.',
            f'In terms of food aspects, a food photo of {class_name1}, which has {ovo_text}, differs from {class_name2}.'
        ]

    elif dataset == 'places-365':
        templates = [
            f'{class_name1}, a type of places, can be distinguished from {class_name2}, a type of places, by the characteristics of {ovo_text}',
            f'Because of {ovo_text}, {class_name1}, a type of places, is different from {class_name2}, a type of places.'
        ]

    elif dataset == 'CUB200':
        templates = [
            f'{class_name1}, a type of bird, can be distinguished from {class_name2} by the characteristics of {ovo_text}', 
            f'Because of {ovo_text}, {class_name1}, a type of bird, is different from {class_name2}.', 
            f"{class_name1}, a type of bird, exhibits a unique {ovo_text} that sets it apart from {class_name2}.",
            f"{class_name1}, a type of bird, uses {ovo_text} in a way that differs from {class_name2}, making it distinctive.", 
            f"The unique expression of {ovo_text} is what separates {class_name1}, a type of bird, from {class_name2}.", 
            f"In handling {ovo_text}, {class_name1}, a type of bird, differs distinctly from {class_name2}.",
            f"{class_name1}, a type of bird, stands out in {ovo_text} compared to {class_name2}.", 
            f"The way {class_name1}, a type of bird, approaches {ovo_text} distinguishes it from {class_name2}.", 
            f"In {ovo_text} expression, {class_name1}, a type of bird, deviates from {class_name2}.", 
            f"Unique {ovo_text} utilization makes {class_name1}, a type of bird, different from {class_name2}.",
            f"Differences in {ovo_text} usage separate {class_name1}, a type of bird, from {class_name2}." 
        ]
    
    elif dataset == 'DTD':
        templates = [
            f"{class_name1}, a type of texture, exhibits a unique {ovo_text} that sets it apart from {class_name2}." 
            f"Because of {ovo_text}, {class_name1} is different from {class_name2}.",  
            f"In handling {ovo_text}, {class_name1} differs distinctly from {class_name2}.", 
            f"In {ovo_text} expression, {class_name1} deviates from {class_name2}.", 
        ]
        
    else:
        templates = [
            f"Because of {ovo_text}, {class_name1} is different from {class_name2}.", 
            f"{class_name1} is characterized by a distinct {ovo_text}, while {class_name2} isn't.", 
            f"The distinctive way in which {class_name1} handles {ovo_text} separates it significantly from {class_name2}.", 
            f"{class_name1} stands out in {ovo_text} compared to {class_name2}.", 
            f"The way {class_name1} approaches {ovo_text} distinguishes it from {class_name2}.", 
            f"In {ovo_text} expression, {class_name1} deviates from {class_name2}.",
            f"Differences in {ovo_text} usage separate {class_name1} from {class_name2}." 
        ]

    return templates

def heuristic_classifier(images_coder, text_nums):
    idx = 0
    cls_score_list = []

    for att_ana_num, name_syno_num in text_nums:
        max_name_score = torch.max(images_coder[:, idx+att_ana_num:idx+att_ana_num+name_syno_num], dim=1)[0]
        cls_score = torch.cat((images_coder[:, idx:idx+att_ana_num], max_name_score.unsqueeze(1)), dim=1)
        cls_score_list.append(torch.mean(cls_score, dim=1).unsqueeze(1))
        idx += (att_ana_num+name_syno_num)
        
    total_cls_score = torch.cat(cls_score_list, dim=1)
    total_cls_score = 100 * total_cls_score

    return total_cls_score

def get_class_names(dataset):
    if dataset == 'ImageNet-V2':
        dataset = "ImageNet"
            
    file_path = f'configs/class_name/{dataset}.txt'  
    category_list = []

    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip()
            category_list.append(cleaned_line)

    return category_list

def modify_top_gap(img_feats, indices, class_names, coder):
    img_number = img_feats.shape[0]
    top_num = indices.shape[1]

    mod_idx_list = []
    for i in range(img_number):
        img_feat = img_feats[i].cuda()
        top5_cls = [class_names[indices[i,j].item()] for j in range(top_num)]
        top5_cls_idx = [indices[i,j].item() for j in range(top_num)]

        scores = []
        for cls in top5_cls:
            score = []
            other_clses = [class_name for class_name in top5_cls if class_name != cls]

            for other_cls in other_clses:
                if other_cls != cls:  
                    try:
                        cls_ovo_coder = coder.get_ovo_CODER(img_feat, cls, other_cls)
                        other_cls_ovo_coder = coder.get_ovo_CODER(img_feat, other_cls, cls)
                        score.append(torch.mean(cls_ovo_coder, dim=0) - torch.mean(other_cls_ovo_coder, dim=0))
                    except:
                        score.append(0)
            scores.append(sum(score) / len(score))

        zipped_lists = zip(scores, top5_cls_idx)
        sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
        scores, mod_idx = zip(*sorted_lists)
        mod_idx_list.append(mod_idx[0])

    mod_idx = torch.tensor(mod_idx_list)

    return mod_idx

def get_thres(dataset):
    thres_dict = {
        'ImageNet': 0.1,
        'ImageNet-V2': 0.05,
        'places-365': 0.1,
        'caltech-101': 0.3,
        'DTD': 0.6,
        'CUB200': 0.2,
        'eurosat': 0.3, 
        'food-101': 0.1,
        'oxford-pet': 0.1
    }

    return thres_dict[dataset]


class CODER():
    def __init__(self, dataset:str, text_type_list:list, clip_model, classnames, task_type='zero-shot') -> None:
        self.dataset = dataset
        self.class_name_lists = classnames
        self.clip_model = clip_model
        self.text_type_list = text_type_list
        self.task_type = task_type

        if task_type == 'few-shot':
            self.update_class_names()

        if "name" in text_type_list:
            self.name_texts = self.get_name_texts()
        if "att" in text_type_list:
            self.att_texts = self.get_att_texts()
        if "ana" in text_type_list:
            self.ana_texts = self.get_ana_texts()
        if "syno" in text_type_list:
            self.syno_texts = self.get_syno_texts()
        if "ovo" in text_type_list:
            self.ovo_texts = self.get_ovo_texts()

        
        self.concate_general_texts()

    def get_class_names(self):
        if self.dataset == 'ImageNet-V2':
            dataset = "ImageNet"
        else:
            dataset = self.dataset
                
        file_path = f'configs/class_name/{dataset}.txt'  
        category_list = []

        with open(file_path, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                category_list.append(cleaned_line)
        return category_list

    def get_name_texts(self):
        print("============================= get name texts =============================")
        name_texts = []
        with torch.no_grad():
            for class_name in tqdm(self.class_name_lists):
                class_name = class_name.replace('_', ' ')
                text_inputs = clip.tokenize(get_templates(self.dataset, 'name', class_name=class_name), truncate=True).cuda()
                text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                name_texts.append(text_features)

        name_texts = torch.cat(name_texts).cuda()
            
        return name_texts
    
    def get_att_texts(self):
        print("============================= get attribute texts =============================")
        if self.dataset == 'ImageNet-V2':
            dataset = "ImageNet"
        else:
            dataset = self.dataset

        data_path = f'expert_knowledge/attribute/{dataset}.pkl'
        f = open(data_path, 'rb')
        attribute = pickle.load(f)

        attribute_texts = [[] for _ in range(len(attribute))]
        cls_attribute_num = []

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            for i in tqdm(range(len(attribute))):
                class_name = self.class_name_lists[i]
                cls_attribute_num.append(len(attribute_texts[i]))
                input_list = []
                for j in range(len(attribute[i])):
                    input_list = input_list + get_templates(self.dataset, "att", class_name=class_name, att=attribute[i][j], task_type=self.task_type)
                for input in input_list:
                    text_feautre = clip.tokenize(input, truncate=True).to(device)
                    text_feature = self.clip_model.encode_text(text_feautre)
                    text_features = F.normalize(text_feature)
                    attribute_texts[i].append(text_features)
                if len(attribute_texts[i]):
                    attribute_texts[i] = torch.cat(attribute_texts[i])
                else:
                    attribute_texts[i] = None


        return attribute_texts

    def get_ana_texts(self):
        print("============================= get analogous texts =============================")
        if self.dataset == 'ImageNet-V2':
            f = open(f"expert_knowledge/simile_class/ImageNet.pkl", 'rb')
        else:
            f = open(f"expert_knowledge/simile_class/{self.dataset}.pkl", 'rb')
        analogous_classes = pickle.load(f)

        analgous_texts = [[] for _ in range(len(analogous_classes))]
        cls_analogous_num = []

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            for i in tqdm(range(len(analogous_classes))):
                class_name = self.class_name_lists[i]
                cls_analogous_num.append(len(analogous_classes[i]))
                input_list = []
                for j in range(len(analogous_classes[i])):
                    input_list = input_list + get_templates(self.dataset, "ana", class_name=class_name, ana=analogous_classes[i][j], task_type=self.task_type)
                for input in input_list:
                    text_inputs = clip.tokenize(input, truncate=True).to(device)
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = F.normalize(text_features)
                    analgous_texts[i].append(text_features)
                if len(analgous_texts[i]):
                    analgous_texts[i] = torch.cat(analgous_texts[i])
                else:
                    analgous_texts[i] = None

        return analgous_texts
    
    def get_syno_texts(self):
        print("============================= get synonym texts =============================")
        if self.dataset == 'ImageNet-V2':
            dataset = "ImageNet"
        else:
            dataset = self.dataset

        f = open(f"expert_knowledge/synonym/{dataset}.json", 'r')
        synonyms = json.load(f)

        synonym_texts = [[] for _ in range(len(synonyms))]

        with torch.no_grad():
            for i in tqdm(range(len(synonyms))):
                if len(synonyms[i]["syno"]) == 0:
                    synonym_texts[i] = None
                
                else:
                    input_list = []
                    for synonym in synonyms[i]["syno"]:
                        input_list = input_list + get_templates(dataset, 'syno', synonym)
                    text_inputs = clip.tokenize(input_list, truncate=True).cuda()
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = F.normalize(text_features)
                
                    synonym_texts[i] = text_features
        
        return synonym_texts

    def get_ovo_texts(self):
        print("============================= get ovo texts =============================")
        if self.dataset == 'ImageNet-V2':
            dataset = "ImageNet"
        else:
            dataset = self.dataset

        f = open(f"expert_knowledge/ovo_attribute/{dataset}.json", 'r')
        ovo_dict = json.load(f)
        ovo_texts = {}

        with torch.no_grad():
            for cls1, hard_cls_dict in tqdm(ovo_dict.items()):
                if cls1 not in ovo_texts.keys():
                    ovo_texts[cls1] = {}
                
                for cls2, ovos in hard_cls_dict.items():
                    ovo_texts[cls1][cls2] = []

                    input_list = []
                    for ovo in ovos:
                        template_ovo = get_templates(dataset, 'ovo', ovo=ovo, class_name=cls1, other_class_name=cls2)
                        self.ovo_template_num = len(template_ovo)
                        input_list = input_list + template_ovo

                    text_inputs = clip.tokenize(input_list, truncate=True).cuda()
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = F.normalize(text_features)

                    ovo_texts[cls1][cls2] = text_features.cpu()

        return ovo_texts

    def concate_general_texts(self):
        total_texts_list = []
        total_general_texts_num_list = []
        
        for i in range(len(self.class_name_lists)):
            att_num, ana_num, name_num, syno_num = 0, 0, 0, 0
            text_list = []
            if "att" in self.text_type_list:
                if self.att_texts[i] is not None:
                    text_list.append(self.att_texts[i])
                    att_num = self.att_texts[i].shape[0]
            if "ana" in self.text_type_list:
                if self.ana_texts[i] is not None:
                    text_list.append(self.ana_texts[i])
                    ana_num = self.ana_texts[i].shape[0]
            if "name" in self.text_type_list:
                text_list.append(self.name_texts[i].unsqueeze(0))
                name_num = 1
            if "syno" in self.text_type_list:
                if self.syno_texts[i] is not None:
                    text_list.append(self.syno_texts[i])
                    syno_num = self.syno_texts[i].shape[0]
            
            total_texts_list.append(torch.cat(text_list, dim=0))
            total_general_texts_num_list.append((att_num+ana_num, name_num+syno_num))
        
        self.total_general_texts = torch.cat(total_texts_list, dim=0)
        self.total_general_texts_num_list = total_general_texts_num_list

    def get_general_CODER(self, images):
        images = images.cuda()
        if images.dtype == torch.float:
            images = F.normalize(images, p=2, dim=1)
        elif images.dtype == torch.float16:
            images /= images.norm(dim=-1, keepdim=True)

        self.total_general_texts = F.normalize(self.total_general_texts, p=2, dim=1)

        images_coder = images.float() @ self.total_general_texts.T.float()
 
        return images_coder, self.total_general_texts_num_list

    def get_ovo_CODER(self, images, cls1, cls2):
        images = images.cuda()
        if images.dtype == torch.float:
            images = F.normalize(images, p=2, dim=1)
        elif images.dtype == torch.float16:
            images /= images.norm(dim=-1, keepdim=True)

        images_coder = images @ self.ovo_texts[cls1][cls2].T.cuda()

        return images_coder
    
    def get_ovo_template_num(self):
        return self.ovo_template_num
    
    def update_class_names(self):
        if self.dataset == 'eurosat':
            file_path = f'configs/class_name/{self.dataset}1.txt'  
            category_list = []

            with open(file_path, 'r') as file:
                for line in file:
                    cleaned_line = line.strip()
                    category_list.append(cleaned_line)
            
            self.class_name_lists = category_list
    
    def update_total_general_texts(self, new_texts):
        self.total_general_texts = F.normalize(new_texts, p=2, dim=1)


def main():
    text_type_list = ["proto"]
    CODER(text_type_list)

if __name__ == "__main__":
    main()