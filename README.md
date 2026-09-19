# Geki-teikokukagekidan-kai
Explore Jev in browser used. Playing with [檄! 帝国華撃団（改）](https://www.youtube.com/watch?v=6_FXujqJbQE).


* **About 檄! 帝国華撃団（改）**

> 檄! 帝国華撃団（改）·UTAHIME DREAM All Stars【
>    Ibuki Yumesaki (CV:Rena Yamasaki),
>    Maika Sakuragi (CV:Anna Suzuki),
>    Kiyomi Mashiro (CV:Arisa Sonohara),
>    HiREN (CV:Kaya),
>    Hikari Mizutsuki (CV:Karin Isobe),
>    Rin Takagi (CV:Tomomi Jiena Sumi),
>    Himawari Hagiwara (CV:Reo Kurachi),
>    SAKURAKO (CV:Yume Takeuchi) 
>    】
> Kiseki nanka iranai

## Reference

* [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
* [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)


```bash
docker build \
  --build-arg PROJECT_DIR=componet/jevbrowser \
  -t jev-browse:dev \
  -f componet/jevbrowser/Dockerfile .

```