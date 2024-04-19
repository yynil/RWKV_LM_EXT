# RWKV_LM_EXT
This project is to extend RWKV LM's capabilities including sequence classification/embedding/peft/cross encoder/bi encoder/multi modalities, etc.


## src/model_ext.py

We extends two types of model based on RWKV(5,6)。

- RwkvForClassification

This class is used to do sequence classification.
```mermaid
graph LR
    A(idx) --> B[embeddings]
    B --> C[Apply rwkv blocks]
    C --> D[Found the eos id's embeddings]
    D --> E[Score the embeddings]
    E --> F(Return scores)
```

- RwkvForSequenceEmbedding

This class is used to do sequence embedding.
```mermaid
graph LR
    A(idx) --> B[embeddings]
    B --> C[Apply rwkv blocks]
    C --> D[Apply pooling method]
    D --> E(Return embeddings)
```