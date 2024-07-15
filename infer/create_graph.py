import os
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(parent_path)
print(f'add path: {parent_path} to sys.path')
from infer.states_generator import StatesGenerator
import jieba
import addressparser
if __name__ == '__main__':    
    model_file = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'

    # kg_states_file = '/media/yueyulin/data_4t/models/states_tuning/instructKGC_scattered/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    # type_states_file = '/media/yueyulin/data_4t/models/states_tuning/kg_type/20240702-105004/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    unit_extractor_states_file = '/media/yueyulin/data_4t/models/states_tuning/units_extractor/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    sg = StatesGenerator(model_file,tokenizer_file)
    sg.load_states(unit_extractor_states_file,'unit_extractor')
    unit_instruction = '你是一个单位提取专家。请从input中抽取出数字和单位，请按照JSON字符串的格式回答，无法提取则不输出。'
    import networkx as nx
    import os
    import json
    from kg_schema import whole_schema,all_types
    schema = whole_schema['地理地区']
    allow_relations = schema[1]
    print(f'allow_relations: {allow_relations}')
    allow_head_tail_types = {}
    for h_t_type in schema[0]:
        h,r,t = h_t_type.split('_')
        heads = h.split('/')
        tails = t.split('/')
        for head in heads:
            for tail in tails:
                if head not in allow_head_tail_types:
                    allow_head_tail_types[head] = {}
                if r not in allow_head_tail_types[head]:
                    allow_head_tail_types[head][r] = []
                allow_head_tail_types[head][r].append(tail)
    print(f'allow_head_tail_types: {allow_head_tail_types}')
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output_relations.jsonl')
    relations = []
    with open(input_file,'r') as f:
        for line in f:
            r = json.loads(line)
            if 'relation' in r and 'head' in r and 'tail' in r and r['relation'] in schema[1]:
                if 'head_type' not in r:
                    r['head_type'] = schema[0]
                if 'tail_type' not in r:
                    if 'type' in r:
                        r['tail_type'] = r['type']
                    else:
                        r['tail_type'] = schema[0]
                if r['head_type'] in allow_head_tail_types and \
                    r['relation'] in allow_head_tail_types[r['head_type']] \
                        and r['tail_type'] in allow_head_tail_types[r['head_type']][r['relation']]:
                    
                    if r['tail_type'] == '地理地区':
                        df = addressparser.transform([r['tail']])
                        tail_province = df['省'][0]
                        tail_city = df['市'][0]
                        tail_district = df['区'][0]
                        tail_region = df['地名'][0]
                        final_name = tail_province+tail_city+tail_district+tail_region
                        r['tail'] = final_name

                    if r['head_type'] == '地理地区':
                        df = addressparser.transform([r['head']])
                        province = df['省'][0]
                        city = df['市'][0]
                        district = df['区'][0]
                        region = df['地名'][0]
                        final_name = province+city+district+region
                        r['head'] = final_name
                        # regions = list(jieba.cut(r['head']))
                        # if len(regions) > 1:
                        #     r['head'] = regions[-1]
                        #     for i in range(len(regions)-1):
                        #         relations.append({'head':regions[i+1],'tail':regions[i],'relation':'位于','head_type':'地理地区','tail_type':'地理地区'})
                    
                        # regions = list(jieba.cut(r['tail']))
                        # if len(regions) > 1:
                        #     r['tail'] = regions[-1]
                        #     for i in range(len(regions)-1):
                        #         relations.append({'head':regions[i+1],'tail':regions[i],'relation':'位于','head_type':'地理地区','tail_type':'地理地区'})
                    relations.append(r)
                    if r['tail_type'] == '度量':
                        try:
                            unit_output = sg.generate(r['tail'],unit_instruction,'unit_extractor',top_k=0,top_p=0,gen_count=128)
                            result = json.loads(unit_output)
                            number = result['number']
                            unit = result['unit']
                            r['tail'] = number
                            relations.append({'head':r['tail'],'tail':unit,'relation':'单位','head_type':'度量','tail_type':'单位'})
                        except:
                            print(f'error in parsing unit {r}')
                            continue
                    
            else:
                print(f'{r} is broken')
    print(f'final relations count: {len(relations)}')

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'output_relations_filtered.jsonl'),'w') as f:
        for r in relations:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    def create_graph_from_relations(relations):
        # 创建一个有向图
        G = nx.DiGraph()
        
        # 遍历关系数组，添加节点和边
        for relation in relations:
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
        
        return G
    graph = create_graph_from_relations(relations)
    # import matplotlib.pyplot as plt
    # from matplotlib import rcParams
    # rcParams['font.family'] = 'WenQuanYi Zen Hei'
    # rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(20, 16))
    # pos = nx.spring_layout(graph)
    # nx.draw_networkx_nodes(graph, pos)
    # nx.draw_networkx_edges(graph, pos,arrowstyle='->',arrowsize=5)
    # nx.draw_networkx_labels(graph, pos,font_family='WenQuanYi Zen Hei')
    # nx.draw_networkx_edge_labels(graph, pos,edge_labels=nx.get_edge_attributes(graph,'relation'),font_family='WenQuanYi Zen Hei')
    # plt.savefig('geo.png')
    # plt.close()
    # print(f'graph nodes: {graph.nodes}')
    print(f'Graph has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')
    from networkx.drawing.nx_agraph import graphviz_layout,write_dot
    import pygraphviz as pgv
    A = nx.nx_agraph.to_agraph(graph)
    A.layout(prog='dot')
    A.draw('geo.svg',format='svg')
    del A
    print(f'dump to {os.path.join(os.path.dirname(os.path.realpath(__file__)),"geo.svg")}')
