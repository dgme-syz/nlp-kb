import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from tqdm import trange, tqdm



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

class GlobalConfig:
    def __init__(self, path):
        self.entity2id, self.relation2id, self.train_triplets, self.valid_triplets = load_data(path)
        self.all_triplets = torch.LongTensor(np.concatenate((self.train_triplets, self.valid_triplets)))
        self.test_graph = build_test_graph(len(self.entity2id), len(self.relation2id), self.train_triplets)
        self.valid_triplets = torch.LongTensor(self.valid_triplets)

def main(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
    best_mrr = 0
    info = GlobalConfig(args.data_path)
    model = RGCN(len(info.entity2id), len(info.relation2id), num_bases=4, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(model)
    if use_cuda:
        model.cuda()
    for epoch in trange(1, (args.epoch + 1), desc='Epochs', position=0):
        model.train()
        optimizer.zero_grad()
        loss = train(info.train_triplets, model, use_cuda, batch_size=args.batch_size, split_size=0.6, 
            negative_sample=1, reg_ratio = 0.01, num_entities=len(info.entity2id), num_relations=len(info.relation2id))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch % args.eval_step == 0:
            tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))
            if use_cuda:
                model.cpu()
            model.eval()
            valid_mrr = valid(info.valid_triplets, model, info.test_graph, info.all_triplets)
            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                            os.path.join(args.save_folder, 'best_mrr_model.pth'))
            if use_cuda:
                model.cuda()
    
    if use_cuda:
        model.cpu()

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--epoch', type=int, default=100, help='epoch')
    paser.add_argument('--batch_size', type=int, default=30000, help='batch_size')
    paser.add_argument('--save_folder', type=str, default='./rgcn_model', help='save folder')
    paser.add_argument('--eval_step', type=int, default=20, help='every x steps calc mrr')
    paser.add_argument('--data_path', type=str, default='./data/gcn_data', help='data path')
    args = paser.parse_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    main(args)