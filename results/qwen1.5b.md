## Overall

| model | mode | n | accuracy % | partial credit % | peak ctx (mean/max) | prompt tok/episode | gen tok/episode | LLM calls | sec/episode | resident GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen1.5b-base | fha | 18 | 11.1 | 31.8 | 634/2561 | 7540 | 164 | 10.2 | 4.6 | 1.51 |
| qwen1.5b-base | sssm | 18 | 0.0 | 1.0 | 913/1323 | 31689 | 6687 | 56.9 | 155.7 | 1.09 |

## By task family and horizon

| family | horizon | model | mode | n | accuracy % | partial % | peak ctx (mean/max) | prompt tok | gen tok | calls | sec | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cart | 4 | qwen1.5b-base | fha | 2 | 100.0 | 100.0 | 382/382 | 1462 | 40 | 5.0 | 2.0 | 1.51 |
| cart | 4 | qwen1.5b-base | sssm | 2 | 0.0 | 0.0 | 843/929 | 16584 | 2430 | 36.0 | 60.5 | 1.09 |
| cart | 8 | qwen1.5b-base | fha | 2 | 0.0 | 22.3 | 420/541 | 2044 | 48 | 6.0 | 1.5 | 1.51 |
| cart | 8 | qwen1.5b-base | sssm | 2 | 0.0 | 2.4 | 846/946 | 5962 | 1112 | 13.0 | 27.3 | 1.09 |
| cart | 16 | qwen1.5b-base | fha | 2 | 0.0 | 0.0 | 301/301 | 766 | 25 | 3.0 | 0.7 | 1.51 |
| cart | 16 | qwen1.5b-base | sssm | 2 | 0.0 | 7.1 | 706/714 | 27528 | 2955 | 55.5 | 80.1 | 1.09 |
| ledger | 4 | qwen1.5b-base | fha | 2 | 0.0 | 19.8 | 733/1194 | 6420 | 298 | 10.0 | 8.0 | 1.51 |
| ledger | 4 | qwen1.5b-base | sssm | 2 | 0.0 | 0.0 | 712/771 | 16968 | 2494 | 36.0 | 61.6 | 1.09 |
| ledger | 8 | qwen1.5b-base | fha | 2 | 0.0 | 0.0 | 296/296 | 498 | 62 | 2.0 | 1.4 | 1.51 |
| ledger | 8 | qwen1.5b-base | sssm | 2 | 0.0 | 0.0 | 786/902 | 30817 | 4274 | 60.0 | 107.7 | 1.09 |
| ledger | 16 | qwen1.5b-base | fha | 2 | 0.0 | 0.0 | 1418/2561 | 37390 | 532 | 28.0 | 14.6 | 1.51 |
| ledger | 16 | qwen1.5b-base | sssm | 2 | 0.0 | 0.0 | 1292/1323 | 76655 | 19362 | 108.0 | 439.9 | 1.09 |
| pointer | 4 | qwen1.5b-base | fha | 2 | 0.0 | 42.4 | 788/1165 | 6984 | 158 | 12.0 | 4.8 | 1.51 |
| pointer | 4 | qwen1.5b-base | sssm | 2 | 0.0 | 0.0 | 1040/1044 | 19357 | 6120 | 36.0 | 136.9 | 1.09 |
| pointer | 8 | qwen1.5b-base | fha | 2 | 0.0 | 47.6 | 530/531 | 3362 | 104 | 9.0 | 3.0 | 1.51 |
| pointer | 8 | qwen1.5b-base | sssm | 2 | 0.0 | 0.0 | 1037/1038 | 32186 | 11743 | 60.0 | 259.5 | 1.09 |
| pointer | 16 | qwen1.5b-base | fha | 2 | 0.0 | 53.9 | 838/841 | 8938 | 204 | 17.0 | 5.8 | 1.51 |
| pointer | 16 | qwen1.5b-base | sssm | 2 | 0.0 | 0.0 | 956/1042 | 59144 | 9692 | 108.0 | 228.1 | 1.09 |
