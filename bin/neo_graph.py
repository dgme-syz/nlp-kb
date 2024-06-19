from neo4j import GraphDatabase
from pyvis.network import Network
import networkx as nx
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class Neo:
    def __init__(self, **kwargs) -> None:
        self.url = kwargs.get('url', 'bolt://localhost:7687')
        self.username = kwargs.get('username', 'neo4j')
        self.password = kwargs.get('password', 'neo4j')
        self.driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
    def make_graph(self, txt_dir: str) -> None:
        with self.driver.session() as session:
            for filename in os.listdir(txt_dir):
                if filename.endswith('.txt'):
                    filepath = os.path.join(txt_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    print(f"Processing neo4j instructions {filename} ...")
                    for line in tqdm(lines):
                        # CREATE (subj13)-[:father]->(subj550)
                        if line.startswith('CREATE'):
                            session.run(line)
                    print(f"Finished processing neo4j instructions {filename}.")
    def visualize_and_clear_graph(self, queries: list[str]) -> str:
        query = "MATCH (a)-[r]->(b) RETURN a, r, b"
        with self.driver.session() as session:
            results = session.run(query)
            G = nx.Graph()
            for record in tqdm(results):
                node_a = record['a'].id
                node_b = record['b'].id
                G.add_node(node_a, label=record['a']['name'])
                G.add_node(node_b, label=record['b']['name'])
                G.add_edge(node_a, node_b, label=record['r'].type)

        net = Network(notebook=True)
        net.from_nx(G)
        filepath = './neos/graph.html'
        net.save_graph(filepath)
        # 删除带有 TEMP 标签的节点和关系
        return filepath