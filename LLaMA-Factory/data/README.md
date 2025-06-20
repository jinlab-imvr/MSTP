To train with a custom dataset, you can organize your data according to the following structure.
```text
GraSP
|
|___augmented_train
|    |
|    |___frames
|         |
|         |___CASE001
|              |___00000_0.jpg
|              |___00000_1.jpg
|              |___00000_2.jpg
|              |___00001_0.jpg
|              |...
|         |___CASE002
|         |...
|___train
|    |
|    |___frames
|         |
|         |___CASE001
|              |___00000.jpg
|              |___00001.jpg
|              |___00002.jpg
|              |___00003.jpg
|              |...
|         |___CASE002
|         |...
|___test
|    |
|    |___frames
|         |
|         |___CASE041
|              |___00000.jpg
|              |___00001.jpg
|              |___00002.jpg
|              |___00003.jpg
|              |...
|         |___CASE047
|         |...
|
```
The label json file format for training is as follows
Multimodal image datasets require an images column containing the paths to the input images.
The number of images should be identical to the <image> tokens in the conversations.
```text
[
    {
        "images": [
            "GraSP/augmented_train/frames/CASE003/02605_0.jpg"
        ],
        "keywords": [
            "phase recognition",
            "step recognition"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<image>ligation of the deep dorsal venous complex, tying the suture"
            },
            {
                "from": "gpt",
                "value": "Change"
            }
        ]
    }
]
```
Regarding the above dataset, the dataset description in dataset_info.json should be:
```text
    "GraSP_chain1_1s": {
    "file_name": "unified_labels_image_train_phase_step_chain_1_1s.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
```
