import queue, json, os, torch
from bin.rgcn import GlobalConfig, RGCN

from config import gcn_data_root as data_root


class InterFace:
    def __init__(self):
        pass
    def update(self, data_dir: dir):
        self.info = GlobalConfig(data_dir)
        self.id2entity = {v: k for k, v in self.info.entity2id.items()}
        self.id2relation = {v: k for k, v in self.info.relation2id.items()}
        self.entities = self.info.entity2id
        self.relations = self.info.relation2id

    def use_2entitys_to_get_test_triples(self, entity1, entity2):
        triples = []
        for i in range(len(self.relations)):
            triples.append([self.entities[entity1], i, self.entities[entity2]])
        return triples


    def use_2entitys_to_get_relation(self, entity1, entity2, threshold=2.):
        triples = self.use_2entitys_to_get_test_triples(entity1, entity2)
        return self.get_kth_relation(triples, threshold)
        
    def get_kth_relation(self, triples, threshold=2.):
        model = RGCN(len(self.entities), len(self.relations), num_bases=4, dropout=0.2)
        model.load_state_dict(torch.load('./rgcn_model/best_mrr_model.pth')['state_dict'])
        model.eval()
        
        ans = []
        for tri in triples:
            trix = torch.LongTensor([tri])
            embed = model(self.info.test_graph.entity, self.info.test_graph.edge_index, self.info.test_graph.edge_type)
            score = model.distmult(embed, trix)
            if score.item() > threshold:
                ans.append(([self.id2entity[tri[0]], self.id2relation[tri[1]], self.id2entity[tri[2]]], score.item()))
        ans.sort(key=lambda x: -x[1])
        # print(ans)
        return ans

    def near_entitys(self, entity, lim_egde=2):
        # 得到只经过 lim_egde 条边可以到达的实体
        li = [0 for _ in range(len(self.entities))]
        li[self.entities[entity]] = 1
        # 由 test_graph.edge_index 获取邻接表
        adj = [[] for _ in range(len(self.entities))]
        for i in range(self.info.test_graph.edge_index.shape[1]):
            adj[self.info.test_graph.edge_index[0][i].item()].append(self.info.test_graph.edge_index[1][i].item())
        # bfs
        res = [self.entities[entity]]
        q = queue.Queue()
        q.put(self.entities[entity])
        for _ in range(lim_egde):
            o = []
            while not q.empty():
                u = q.get()
                for v in adj[u]:
                    if li[v] == 0:
                        li[v] = 1
                        res.append(v)
                        o.append(v)
            for x in o:
                q.put(x)
        
        return res

    def use_entity1_plus_relation_to_get_relation(self, entity1, relation,
            threshold=2., lim_edge=2):
        entity2s = self.near_entitys(entity1, lim_edge)
        triples = []
        for x in entity2s:
            triples.append([self.entities[entity1], self.relations[relation], x])
        return self.get_kth_relation(triples, threshold)

    def use_entity2_plus_relation_to_get_relation(self, entity2, relation,
            threshold=2., lim_edge=2):
        entity1s = self.near_entitys(entity2, lim_edge)
        triples = []
        for x in entity1s:
            triples.append([x, self.relations[relation], self.entities[entity2]])
        return self.get_kth_relation(triples, threshold)

if __name__ == '__main__':
    pass