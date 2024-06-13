# Run Benchmarks

Prepare benchmark data first:
- see `prepare_benchmark_data.ipynb`

Train models using benchmark data of different levels:

- First, change the path in the following file to your own：
  - `bash` files in [`./bash/`](./bash/) dir: ``**_path``, e.g., ``data_path``.
  - `data_loader.py` and `plm_models.py`: ``**_checkpoint``, e.g., ``esm2_8b_checkpoint``.

- See `./bash/train_**.sh` to run `train_main.py` for traing models.
  - If **finetune**, you need to **add a line to the `bash` script** with the parameter ``--finetune``; if not finetune, remove it.

- After `train_main.py`, you have to test the model again with `test_main.py`.
  - Because the test in `train_main.py` is only for the last epoch of models, not the best epoch.
  - See `./bash/test_**.sh` to run `test_main.py`


Define the max_len for different levels.：

||pep_max_len|tcr_max_len|
|:-:|:-:|:-:|
|level-I|15|19|
|level-IIA|44 (10+34)|19|
|level-IIB|10|19|
|level-III|10|19|
|level-IV|10|121|


To-do list:

- [ ] Add NoRN Dataset class in `data_loader.py`.
