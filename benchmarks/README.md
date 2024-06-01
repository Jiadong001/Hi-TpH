注意事项：

- 将以下文件中的路径改为自己的：
  - `bash` 脚本（都放在了 [`./bash/`](./bash/) 文件夹）中的各种 path
  - `data_loader.py` 中的 ``protbert_bfd_checkpoint`` 和 ``esm2_8b_checkpoint``
  - `plm_models.py` 中的 ``protbert_bfd_checkpoint`` 和 ``esm2_8b_checkpoint``

- 如果 **finetune**，需要在 `bash` 脚本中**加一行参数** ``--finetune``；如果不 finetune，去掉。
  - 这与之前代码有所不同

- 在 `train_main.py` 结束后还要再用 `test_main.py` 再测试一次
  - 因为 `train_main.py` 的测试只针对最后一轮模型，而不是最优模型
  - 具体方式参考 [`./bash/test_baseline.sh`](./bash/test_baseline.sh)


规定下不同 level 的 max_len：

||pep_max_len|tcr_max_len|
|:-:|:-:|:-:|
|level-I|15|19|
|level-IIA|43 (9+34)|19|
|level-IIB|9|19|
|level-III|9|19|
|level-IV|9|121|



To-do list:

- [ ] 补充 `data_loader.py` 的 NoRN Dataset class