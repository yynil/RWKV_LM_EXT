if __name__ == '__main__':
    import os
    import json
    input_jsonl = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output_relations_filtered.jsonl')
    import networkx as nx
    from networkx import Graph
    G = Graph()
    with open(input_jsonl,'r') as f:
        for line in f:
            relation = json.loads(line)
            head = relation['head']
            tail = relation['tail']
            head_type = relation['head_type'] if 'head_type' in relation else relation['type'] if 'type' in relation else None
            tail_type = relation['tail_type'] if 'tail_type' in relation else relation['type'] if 'type' in relation else None
            head_identifier = f'{head_type}_{head}'
            tail_identifier = f'{tail_type}_{tail}'
            relation_label = relation['relation']
            # 如果节点不存在，则添加节点
            if head_identifier not in G:
                G.add_node(head_identifier)
            if tail_identifier not in G:
                G.add_node(tail_identifier)
            
            # 添加有向边，边的属性为关系标签
            #if edge not in G.edges:
            if not G.has_edge(head_identifier, tail_identifier):
                G.add_edge(head_identifier, tail_identifier, label=relation_label)
    from graspologic.partition import hierarchical_leiden
    community_cluster_size = 15
    community_mapping = hierarchical_leiden(G,max_cluster_size=community_cluster_size,random_seed=86)
    # print(community_mapping)
    sub_graph_nodes = {}
    for community_node in community_mapping:
        node = community_node.node
        cluster = community_node.cluster
        parent_cluster = community_node.parent_cluster
        level = community_node.level
        is_final_cluster = community_node.is_final_cluster
        if cluster not in sub_graph_nodes:
            sub_graph_nodes[cluster] = {'nodes':[],
                                        'level':level,
                                        'parent_cluster':parent_cluster,
                                        'is_final_cluster':is_final_cluster}
        sub_graph_nodes[cluster]['nodes'].append(node)
    print(sub_graph_nodes)
    #randomly select a level 2 cluster
    import random
    level_2_clusters = [k for k,v in sub_graph_nodes.items() if v['level'] == 2]
    selected_cluster = random.choice(level_2_clusters)
    parent_cluster = sub_graph_nodes[selected_cluster]['parent_cluster']
    parent_parent_cluster = sub_graph_nodes[parent_cluster]['parent_cluster']


    #get the subgraph of the selected cluster
    selected_nodes = sub_graph_nodes[selected_cluster]['nodes']
    selected_nodes.extend(sub_graph_nodes[parent_cluster]['nodes'])
    selected_nodes.extend(sub_graph_nodes[parent_parent_cluster]['nodes'])
    sub_graph = G.subgraph(selected_nodes)
    #export to svg using pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout,write_dot
    import pygraphviz as pgv
    A = nx.nx_agraph.to_agraph(sub_graph)
    A.layout(prog='dot')
    A.draw('sub_graph.svg',format='svg')
    del A
    print(f'dump to {os.path.join(os.path.dirname(os.path.realpath(__file__)),"sub_graph.svg")}')


