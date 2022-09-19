from torch import nn
import pdb
import torch
from copy import deepcopy
import torch.nn.functional as F

from constant import constants


class ObjNameCoordStateEncodeNameTokenMix(nn.Module):
    def __init__(self, args, large_language_model_token_encoder_wte):
        super(ObjNameCoordStateEncodeNameTokenMix, self).__init__()
        self.output_dim = args.llm_hidden_size
        self.hidden_dim = args.hidden_size

        self.large_language_model_token_encoder_wte = deepcopy(large_language_model_token_encoder_wte)
        
        self.class_fc = nn.Linear(self.output_dim, int(self.hidden_dim / 2))
        self.coord_embedding = nn.Sequential(nn.Linear(3, int(self.hidden_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2)))

        self.combine = nn.Sequential(nn.ReLU(), nn.Linear(self.hidden_dim, self.output_dim))

        
    def forward(self, input_obs_node_gpt2_token, input_obs_node_gpt2_token_mask, input_obs_char_obj_rel_gpt2_token):
        obs_node_class_name_feat_tem = self.large_language_model_token_encoder_wte(input_obs_node_gpt2_token.long())
        obs_node_class_name_feat_tem = (obs_node_class_name_feat_tem * input_obs_node_gpt2_token_mask[:,:,:,None]).sum(2) / (1e-9 + input_obs_node_gpt2_token_mask.sum(2)[:,:,None])
        class_embedding = self.class_fc(obs_node_class_name_feat_tem)

        coord_embedding = self.coord_embedding(input_obs_char_obj_rel_gpt2_token)
        
        inp = torch.cat([class_embedding, coord_embedding], dim=2)

        return self.combine(inp)


class SimpleAttention(nn.Module):
    def __init__(self, n_features, n_hidden, key=True, query=False):
        super().__init__()
        self.key = key
        self.query = query
        if self.key:
            self.make_key = nn.Linear(n_features, n_features)
        if self.query:
            self.make_query = nn.Linear(n_features, n_features)
        self.n_out = n_hidden


    def forward(self, features, hidden, mask=None):
        if self.key:
            key = self.make_key(features)
        else:
            key = features

        if self.query:
            query = self.make_query(hidden)
        else:
            query = hidden

        scores = (key * query).sum(dim=2)
        
        if mask is not None:
            mask_values = (torch.min(scores, -1)[0].view(-1, 1).expand_as(scores)) * mask
            scores = scores * (1-mask) + mask_values
        
        return scores

import numpy as np
import pdb
import torch
import torch.nn as nn
from copy import deepcopy

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()

        self.base = GoalAttentionModel(args)

    def forward(self, data):
        value = self.base(data)
        return value

class GoalAttentionModel(nn.Module):
    def __init__(self, args):
        super(GoalAttentionModel, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.args = args
        self.hidden_size = args.hidden_size

        self.max_node_length = args.max_node_length

        self.actions = args.actions
        
        ## pretrained language model
        self.llm_hidden_size = args.llm_hidden_size
    
        model = args.model
        tokenizer = args.tokenizer

        self.llm = model.from_pretrained('gpt2')
        model_tem = model.from_pretrained('gpt2')

        self.tokenizer = tokenizer.from_pretrained('gpt2')
        
        self.large_language_model_token_encoder_wte = deepcopy(model_tem.transformer.wte)

        ## encoders
        self.single_object_encoding_name_token = ObjNameCoordStateEncodeNameTokenMix(self.args, self.large_language_model_token_encoder_wte)
        self.large_language_model_token_encoder_wte_goal = deepcopy(model_tem.transformer.wte)
        self.large_language_model_token_encoder_wte_history = deepcopy(model_tem.transformer.wte)

        ## object / action decoders
        self.action_decoder_hidden = nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        self.verb_decoder = nn.Sequential(nn.ReLU(), nn.Linear(self.llm_hidden_size, len(self.actions)))
        self.object_attention = SimpleAttention(self.llm_hidden_size, self.hidden_size, key=False, query=False)

        self.train()


    def forward(self, inputs):

        input_obs_node_gpt2_token, input_obs_node_gpt2_token_mask, input_obs_char_obj_rel_gpt2_token, \
                history_action_gpt2_token, history_action_gpt2_token_mask, goal_gpt2_token, goal_gpt2_token_mask = inputs
        
        B = input_obs_node_gpt2_token.shape[0]
        
        ## encode goal
        goal_embedding = self.large_language_model_token_encoder_wte_goal(goal_gpt2_token.long())
        goal_embedding = goal_embedding.view(B, -1, self.llm_hidden_size)
        goal_embedding_mask = goal_gpt2_token_mask.view(B, -1)

        ## encode observation
        input_node_embedding = self.single_object_encoding_name_token(input_obs_node_gpt2_token, input_obs_node_gpt2_token_mask, input_obs_char_obj_rel_gpt2_token)
        input_node_mask = input_obs_node_gpt2_token_mask.sum(2)>0
    
        ## encode history
        history_embedding = self.large_language_model_token_encoder_wte_history(history_action_gpt2_token.long())
        history_embedding = history_embedding.view(B, -1, self.llm_hidden_size)
        history_embedding_mask = history_action_gpt2_token_mask.view(B, -1)

        ## joint embedding
        joint_embedding = torch.cat([goal_embedding, history_embedding, input_node_embedding], dim=1)
        joint_mask = torch.cat([goal_embedding_mask, history_embedding_mask, input_node_mask], dim=1)
        
        ## pre-trained language model
        pretrained_language_output = self.large_language_model(inputs_embeds=joint_embedding, attention_mask=joint_mask, output_hidden_states=True)
        
        language_ouput_embedding = pretrained_language_output['hidden_states'][-1]
        joint_mask = joint_mask.unsqueeze(-1)
        language_ouput_embedding = language_ouput_embedding * joint_mask
        context_embedding = language_ouput_embedding.sum(1) / (1e-9 + joint_mask.sum(1))
        
        obs_node_embedding = language_ouput_embedding[:,-self.max_node_length:,:]
        action_hidden = self.action_decoder_hidden(context_embedding)

        ## predict verb / object
        verb = self.verb_decoder(action_hidden)
        obj = self.object_attention(obs_node_embedding, action_hidden.unsqueeze(dim=1), mask=1-input_node_mask.float())
        
        return verb, obj

class trainer:
        def __init__(self, args) -> None:
            self.args = args
            self.model = MyModel(args)
            self.num_agents = args.num_agents
        
        def load_data(self, path, files):
            obs = {
                'past_actions': torch.zeros([self.num_agents, 100, 10], dtype=torch.float),
                'past_actions_mask': torch.zeros([self.num_agents, 100, 10], dtype=torch.int),
                'agent_pos':torch.zeros([self.num_agents, 3]),
                'object_graph': []
            }
            obs = {
                'past_actions': [],
                'past_actions_mask': [],
                'agent_pos': [],
                'object_name': [],
                'object_pos': [],
                'object_graph_mask':[],
                'goal': None,
                'trajectory':None,
                'hint':None
            }
            for f in files:
                p = path + '/' + f
                file = open(p,"rb")
                data = pickle.load(file)
                file.close()
                past_act = []
                for k,v in data['trajectory'].items():
                    if not('actions' in v.keys()):
                        continue
                    past_act.append(v['actions'])
                    obs['past_actions'].append(deepcopy(past_act))
                    for inst in v['obs']['object_graph']:
                        temp_name = self.parse_object_name(inst['name'])
                        obs['object_pos'].append(temp_name)
                        obs['object_name'].append(inst['position'])
                    obs['agent_pos'].append(v['obs']['agent_pos'])
            
            ic(obs)


        def parse_object_name(self, str):
            ic(str)
            re.sub('[0-9]','',str)
            name = str.split('_')


if __name__ == '__main__':
    import pickle
    from icecream import ic
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import re

    class args:
        def __init__(self) -> None:
            const = constants('collect')
            self.hidden_size = 128
            self.max_node_length = 50
            self.llm_hidden_size = 768
            self.model = GPT2LMHeadModel
            self.tokenizer = GPT2Tokenizer
            self.actions = const.actions
            self.output_dim = 128
            self.num_agents = 2

    path ='C:/Users/YangYuxiang/Desktop/proj/proj/task'
    files = ['log_with_seed_3.pkl']

    t = trainer(args())
    t.load_data(path, files)
    

    