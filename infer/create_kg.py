import os
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(parent_path)
print(f'add path: {parent_path} to sys.path')
from infer.states_generator import StatesGenerator


if __name__ == '__main__':
    model_file = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'

    kg_states_file = '/media/yueyulin/data_4t/models/states_tuning/instructKGC_scattered/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    # type_states_file = '/media/yueyulin/data_4t/models/states_tuning/kg_type/20240702-105004/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    sg = StatesGenerator(model_file,tokenizer_file)
    sg.load_states(kg_states_file,'kg')
    # sg.load_states(type_states_file,'type')

    test_txt_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'geo.txt')
    with open(test_txt_file,'r') as f:
        lines = f.readlines()
    import json
    from kg_schema import whole_schema,all_types
    kg_instruction = '你是一个图谱实体知识结构化专家。请从input中抽取出符合schema定义的实体实例和其属性，不存在的属性不输出，属性存在多值就返回列表。请按照JSON字符串的格式回答。'
    # type_instruction = '请根据input中的文本内容，选择schema中的最符合input内容的类别输出，用JSON格式输出。'
    # for s in sentences:
    #     print(s)
    #     input_text = {'input':s,'schema':all_types}
    #     input_text = json.dumps(input_text,ensure_ascii=False)
    #     type_output = sg.generate(input_text,type_instruction,'type',top_k=0,top_p=0,gen_count=128)
    #     print(type_output)
    #     try:
    #         result = json.loads(type_output)['result']
    #         print(f'infer {s} as {result}')
    #         if isinstance(result,list):
    #             result = result[0]
    #         if result != current_type:
    #             if current_type is not None:
    #                 typed_sentences.append((current_type,current_sentence))
    #             current_type = result
    #             current_sentence = s
    #         else:
    #             current_sentence += s
    #     except:
    #         print('error')
    # if current_type is not None:
    #     typed_sentences.append((current_type,current_sentence))
    # print(typed_sentences)

    # for type,sentence in typed_sentences:
    #     type = '事件'
    #     print(f'processing {type} {sentence}')
    #     schema = whole_schema[type]
    #     input_text = json.dumps({'input':sentence,'schema':schema},ensure_ascii=False)
    #     kg_output = sg.generate(input_text,kg_instruction,'kg',top_k=0,top_p=0,gen_count=2048)
    #     print(kg_output)
    #     kg_output = json.loads(kg_output)
    #     print(kg_output['result'])
    #     print('---------------------------------')
    schema = whole_schema['地理地区']
    relations = []
    for txt in lines:
        kg_output = sg.generate(json.dumps({'input':txt,'schema':all_types},ensure_ascii=False),kg_instruction,'kg',top_k=0,top_p=0,gen_count=2048)
        print(kg_output)
        relations.extend(json.loads(kg_output)['result'])
    print(relations)

    def extract_entity_and_type(relation):
        typed_entities = {}
        for r in relation:
            head = r['head']
            tail = r['tail']
            head_type = r['head_type']
            tail_type = r['tail_type']
            if head_type not in typed_entities:
                typed_entities[head_type] = []
            if head not in typed_entities[head_type]:
                typed_entities[head_type].append(head)
            if tail_type not in typed_entities:
                typed_entities[tail_type] = []
            if tail not in typed_entities[tail_type]:
                typed_entities[tail_type].append(tail)
        return typed_entities



    all_typed_entities = extract_entity_and_type(relations)

    print(all_typed_entities)
    
    import networkx as nx

    def create_graph_from_relations(relations):
        # 创建一个有向图
        G = nx.DiGraph()
        
        # 遍历关系数组，添加节点和边
        for relation in relations:
            head = relation['head']
            tail = relation['tail']
            head_type = relation['head_type']
            tail_type = relation['tail_type']
            head_identifier = f'{head_type}_{head}'
            tail_identifier = f'{tail_type}_{tail}'
            relation_label = relation['relation']
            
            # 如果节点不存在，则添加节点
            if head_identifier not in G:
                G.add_node(head_identifier)
            if tail_identifier not in G:
                G.add_node(tail_identifier)
            
            # 添加有向边，边的属性为关系标签
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

    from networkx.drawing.nx_agraph import graphviz_layout,write_dot
    import pygraphviz as pgv
    A = nx.nx_agraph.to_agraph(graph)
    A.layout(prog='dot')
    A.draw('geo.svg',format='svg')
    del A