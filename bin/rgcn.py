import torch, sys, json
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from transformers import Trainer, TrainingArguments
from tqdm import trange, tqdm

config = json.load(open('config.json'))
sys.path.append(config['root_dir'])

from data.gcn_data.data_utils import *

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_type)
        
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

def train(train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations, negative_sample)

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding)

    return loss

def valid(valid_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, hits=[1, 3, 10])

    return mrr

def test(test_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, test_triplets, all_triplets, hits=[1, 3, 10])

    return mrr


entity2id, relation2id, train_triplets, valid_triplets = load_data(config['gcn_data_root'])
all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets)))
test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)
valid_triplets = torch.LongTensor(valid_triplets)

def main():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)

    best_mrr = 0

    
    model = RGCN(len(entity2id), len(relation2id), num_bases=4, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(model)

    if use_cuda:
        model.cuda()

    for epoch in trange(1, (10001), desc='Epochs', position=0):

        model.train()
        optimizer.zero_grad()

        loss = train(train_triplets, model, use_cuda, batch_size=30000, split_size=0.6, 
            negative_sample=1, reg_ratio = 0.01, num_entities=len(entity2id), num_relations=len(relation2id))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 500 == 0:

            tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))

            if use_cuda:
                model.cpu()

            model.eval()
            valid_mrr = valid(valid_triplets, model, test_graph, all_triplets)
            
            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                            'best_mrr_model.pth')

            if use_cuda:
                model.cuda()
    
    if use_cuda:
        model.cpu()

entities = json.load(open(os.path.join(config['gcn_data_root'], 'entities.json')))
id2entity = {v: k for k, v in entities.items()}
relations = json.load(open(os.path.join(config['gcn_data_root'], 'relations.json')))
id2relation = {v: k for k, v in relations.items()}

if __name__ == '__main__':
    # use_2entitys_to_get_relation('san francisco', 'apple')
    main()