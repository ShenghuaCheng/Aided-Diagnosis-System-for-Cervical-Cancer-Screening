# Dataset

---

Every experiment need a config file, and every config file need a dataset config. 
Dataset config describe the data using in the training, evaluation or testing. 
It is manipulating by `core/data/datasets/**_dataset.py`

The format of dataset config should be like [demo](./demo.xlsx).

- There should be three tables(train, val and test) in file.
- Every table contains configs of dataset, including attributes show below:

| flg_use | group_name | subset_name | subset_label | subset_mpp | index_files | mask_index | group_nb | subset_ratio | subset_sample | subset_total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| whether to use this line in train | given name of a group | given name of subset in the group | label of this line | mpp of images this line indicates | index files in which recording all images absolute path of this subset| index file of mask. empty will generate a blank mask for neg images | total images set to a **group** | ratio of this line in the group belonging to | actually number read in | total number of images in index files of this subset |

- When attribute `with_mask` in config is set to `False`, the `mask_index` in dataset config file will be mute.
