# Geki-teikokukagekidan-kai
Explore Jev in browser used. Playing with [檄! 帝国華撃団（改）](https://www.youtube.com/watch?v=6_FXujqJbQE).


* **About 檄! 帝国華撃団（改）**

> 檄! 帝国華撃団（改）·UTAHIME DREAM All Stars【
>
>>  Ibuki Yumesaki (CV:Rena Yamasaki),
>>
>>    Maika Sakuragi (CV:Anna Suzuki),
>>
>>    Kiyomi Mashiro (CV:Arisa Sonohara),
>>
>>    HiREN (CV:Kaya),
>>
>>    Hikari Mizutsuki (CV:Karin Isobe),
>>
>>    Rin Takagi (CV:Tomomi Jiena Sumi),
>>
>>    Himawari Hagiwara (CV:Reo Kurachi),
>>
>>    SAKURAKO (CV:Yume Takeuchi) 
>>
>    】
>
> Kiseki nanka iranai

## Reference

* [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
* [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)


## Setup

* **browser use**

  * deploy

    ```bash
    # build image
    docker build \
      --build-arg PROJECT_DIR=componet/jevbrowser \
      -t jev-browse:dev \
      -f componet/jevbrowser/Dockerfile .

    # run browser server
    docker run -d -p 8766:8766 \
      --env-file .env \
      jev-browse:dev
    ```

* **RLCD**

  * server
    ```bash
    source ~/.venv/bin/activate
    # use Qwen/Qwen3-4B-Instruct-2507-FP8
    MODEL_ID='Qwen/Qwen3-4B-Instruct-2507-FP8' python -m componet.rlcd.server.main --port 7860
    ```

  * deploy

    ```bash
    # build image
    docker build \
      --network=host \
      --build-arg PROJECT_DIR=componet/rlcd \
      -t rlcd:dev \
      -f componet/rlcd/Dockerfile .

    # run RLCD server
    MSYS_NO_PATHCONV=1 docker run -d --gpus=all \
      -p 7860:7860 \
      -v "$HOME/.cache/huggingface:/home/user/.cache/huggingface" \
      -e HF_HOME="/home/user/.cache/huggingface" \
      -e MODEL_ID="Qwen/Qwen3-4B-Instruct-2507-FP8" \
      rlcd:dev
    ```

  * client

    ```bash
    # cache server is ready
    curl http://localhost:7860/api/presets

    # send request
    curl -X POST http://localhost:7860/api/run-rlcd \
      -H 'Content-Type: application/json' \
      --data-binary '@examples/request.json'
    ```
