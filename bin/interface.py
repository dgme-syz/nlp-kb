import queue
from rgcn import *
  
def use_2entitys_to_get_test_triples(entity1, entity2):
    triples = []
    for i in range(len(relations)):
        triples.append([entities[entity1], i, entities[entity2]])
    return triples


def use_2entitys_to_get_relation(entity1, entity2):
    triples = use_2entitys_to_get_test_triples(entity1, entity2)
    return get_kth_relation(triples)
    
def get_kth_relation(triples, threshold=2.):
    model = RGCN(len(entities), len(relations), num_bases=4, dropout=0.2)
    model.load_state_dict(torch.load('best_mrr_model.pth')['state_dict'])
    model.eval()
    
    ans = []
    for tri in triples:
        trix = torch.LongTensor([tri])
        embed = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type)
        score = model.distmult(embed, trix)
        if score.item() > threshold:
            ans.append(([id2entity[tri[0]], id2relation[tri[1]], id2entity[tri[2]]], score.item()))
    ans.sort(key=lambda x: -x[1])
    print(ans)
    return ans

def near_entitys(entity, lim_egde=2):
    # 得到只经过 lim_egde 条边可以到达的实体
    li = [0 for _ in range(len(entities))]
    li[entities[entity]] = 1
    # 由 test_graph.edge_index 获取邻接表
    adj = [[] for _ in range(len(entities))]
    for i in range(test_graph.edge_index.shape[1]):
        adj[test_graph.edge_index[0][i].item()].append(test_graph.edge_index[1][i].item())
    # bfs
    res = [entities[entity]]
    q = queue.Queue()
    q.put(entities[entity])
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

def use_entity1_plus_relation_to_get_relation(entity1, relation):
    entity2s = near_entitys(entity1)
    triples = []
    for x in entity2s:
        triples.append([entities[entity1], relations[relation], x])
    return get_kth_relation(triples)

def use_entity2_plus_relation_to_get_relation(entity2, relation):
    entity1s = near_entitys(entity2)
    triples = []
    for x in entity1s:
        triples.append([x, relations[relation], entities[entity2]])
    return get_kth_relation(triples)

if __name__ == '__main__':
    # use_2entitys_to_get_relation('Snap', 'DefenceAgainsttheDark Arts')
    # near_entitys('03964744', 3)
    use_entity1_plus_relation_to_get_relation('Harry', 'per:friend_of')