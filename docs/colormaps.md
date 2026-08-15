# PixieNN colormaps

These maps are generated from Matplotlib and are available through `--color-map`.

Confidence coloring uses a continuous map:

```bash
pixienn --color-map=viridis --color-by-confidence model.yml image.jpg
pixienn --color-map=viridis --color-by-confidence --stretch-confidence model.yml image.jpg
```

With `--stretch-confidence`, the lowest and highest confidence values in the image map to 0.0 and 1.0.

## Continuous maps

`Continuous` means the map accepts a normalized scalar value. `Sequential / ordered` means it is designed to communicate low-to-high magnitude and is appropriate for confidence. This is a semantic visual-design property, not a claim that every RGB channel is mathematically monotonic.

### `Blues`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7fbff;border:1px solid #999;vertical-align:middle"></span> | `#f7fbff` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#e3eef9;border:1px solid #999;vertical-align:middle"></span> | `#e3eef9` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#d0e1f2;border:1px solid #999;vertical-align:middle"></span> | `#d0e1f2` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#b7d4ea;border:1px solid #999;vertical-align:middle"></span> | `#b7d4ea` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#94c4df;border:1px solid #999;vertical-align:middle"></span> | `#94c4df` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#6aaed6;border:1px solid #999;vertical-align:middle"></span> | `#6aaed6` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#4a98c9;border:1px solid #999;vertical-align:middle"></span> | `#4a98c9` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#2e7ebc;border:1px solid #999;vertical-align:middle"></span> | `#2e7ebc` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#1764ab;border:1px solid #999;vertical-align:middle"></span> | `#1764ab` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#084a91;border:1px solid #999;vertical-align:middle"></span> | `#084a91` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#08306b;border:1px solid #999;vertical-align:middle"></span> | `#08306b` |

### `BuGn`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7fcfd;border:1px solid #999;vertical-align:middle"></span> | `#f7fcfd` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#e9f7fa;border:1px solid #999;vertical-align:middle"></span> | `#e9f7fa` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#d6f0ee;border:1px solid #999;vertical-align:middle"></span> | `#d6f0ee` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#b8e4db;border:1px solid #999;vertical-align:middle"></span> | `#b8e4db` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#8fd4c2;border:1px solid #999;vertical-align:middle"></span> | `#8fd4c2` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#65c2a3;border:1px solid #999;vertical-align:middle"></span> | `#65c2a3` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#48b27f;border:1px solid #999;vertical-align:middle"></span> | `#48b27f` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#2f9858;border:1px solid #999;vertical-align:middle"></span> | `#2f9858` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#157f3b;border:1px solid #999;vertical-align:middle"></span> | `#157f3b` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#006428;border:1px solid #999;vertical-align:middle"></span> | `#006428` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#00441b;border:1px solid #999;vertical-align:middle"></span> | `#00441b` |

### `BuPu`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7fcfd;border:1px solid #999;vertical-align:middle"></span> | `#f7fcfd` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#e5eff6;border:1px solid #999;vertical-align:middle"></span> | `#e5eff6` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#ccddec;border:1px solid #999;vertical-align:middle"></span> | `#ccddec` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#b2cae1;border:1px solid #999;vertical-align:middle"></span> | `#b2cae1` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#9ab4d6;border:1px solid #999;vertical-align:middle"></span> | `#9ab4d6` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#8c95c6;border:1px solid #999;vertical-align:middle"></span> | `#8c95c6` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#8c74b5;border:1px solid #999;vertical-align:middle"></span> | `#8c74b5` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#8a51a5;border:1px solid #999;vertical-align:middle"></span> | `#8a51a5` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#852d90;border:1px solid #999;vertical-align:middle"></span> | `#852d90` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#760c71;border:1px solid #999;vertical-align:middle"></span> | `#760c71` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#4d004b;border:1px solid #999;vertical-align:middle"></span> | `#4d004b` |

### `GnBu`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7fcf0;border:1px solid #999;vertical-align:middle"></span> | `#f7fcf0` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#e5f5e0;border:1px solid #999;vertical-align:middle"></span> | `#e5f5e0` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#d4eece;border:1px solid #999;vertical-align:middle"></span> | `#d4eece` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#bee6bf;border:1px solid #999;vertical-align:middle"></span> | `#bee6bf` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#9fdab8;border:1px solid #999;vertical-align:middle"></span> | `#9fdab8` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#7accc4;border:1px solid #999;vertical-align:middle"></span> | `#7accc4` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#57b8d0;border:1px solid #999;vertical-align:middle"></span> | `#57b8d0` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#389bc6;border:1px solid #999;vertical-align:middle"></span> | `#389bc6` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#1d7eb7;border:1px solid #999;vertical-align:middle"></span> | `#1d7eb7` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#085fa3;border:1px solid #999;vertical-align:middle"></span> | `#085fa3` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#084081;border:1px solid #999;vertical-align:middle"></span> | `#084081` |

### `Greens`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7fcf5;border:1px solid #999;vertical-align:middle"></span> | `#f7fcf5` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#e9f7e5;border:1px solid #999;vertical-align:middle"></span> | `#e9f7e5` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#d3eecd;border:1px solid #999;vertical-align:middle"></span> | `#d3eecd` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#b8e3b2;border:1px solid #999;vertical-align:middle"></span> | `#b8e3b2` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#98d594;border:1px solid #999;vertical-align:middle"></span> | `#98d594` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#73c476;border:1px solid #999;vertical-align:middle"></span> | `#73c476` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#4bb062;border:1px solid #999;vertical-align:middle"></span> | `#4bb062` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#2f974e;border:1px solid #999;vertical-align:middle"></span> | `#2f974e` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#157f3b;border:1px solid #999;vertical-align:middle"></span> | `#157f3b` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#006428;border:1px solid #999;vertical-align:middle"></span> | `#006428` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#00441b;border:1px solid #999;vertical-align:middle"></span> | `#00441b` |

### `Greys`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#f3f3f3;border:1px solid #999;vertical-align:middle"></span> | `#f3f3f3` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#e2e2e2;border:1px solid #999;vertical-align:middle"></span> | `#e2e2e2` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#cecece;border:1px solid #999;vertical-align:middle"></span> | `#cecece` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#b5b5b5;border:1px solid #999;vertical-align:middle"></span> | `#b5b5b5` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#959595;border:1px solid #999;vertical-align:middle"></span> | `#959595` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#7a7a7a;border:1px solid #999;vertical-align:middle"></span> | `#7a7a7a` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#5f5f5f;border:1px solid #999;vertical-align:middle"></span> | `#5f5f5f` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#404040;border:1px solid #999;vertical-align:middle"></span> | `#404040` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#1d1d1d;border:1px solid #999;vertical-align:middle"></span> | `#1d1d1d` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |

### `Oranges`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff5eb;border:1px solid #999;vertical-align:middle"></span> | `#fff5eb` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#fee9d4;border:1px solid #999;vertical-align:middle"></span> | `#fee9d4` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdd9b4;border:1px solid #999;vertical-align:middle"></span> | `#fdd9b4` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdc38d;border:1px solid #999;vertical-align:middle"></span> | `#fdc38d` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#fda762;border:1px solid #999;vertical-align:middle"></span> | `#fda762` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fd8c3b;border:1px solid #999;vertical-align:middle"></span> | `#fd8c3b` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#f3701b;border:1px solid #999;vertical-align:middle"></span> | `#f3701b` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#e25508;border:1px solid #999;vertical-align:middle"></span> | `#e25508` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#c54102;border:1px solid #999;vertical-align:middle"></span> | `#c54102` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#9e3303;border:1px solid #999;vertical-align:middle"></span> | `#9e3303` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#7f2704;border:1px solid #999;vertical-align:middle"></span> | `#7f2704` |

### `OrRd`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff7ec;border:1px solid #999;vertical-align:middle"></span> | `#fff7ec` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#feebd0;border:1px solid #999;vertical-align:middle"></span> | `#feebd0` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#fddcaf;border:1px solid #999;vertical-align:middle"></span> | `#fddcaf` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdca94;border:1px solid #999;vertical-align:middle"></span> | `#fdca94` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdb27b;border:1px solid #999;vertical-align:middle"></span> | `#fdb27b` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fc8c59;border:1px solid #999;vertical-align:middle"></span> | `#fc8c59` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#f26d4b;border:1px solid #999;vertical-align:middle"></span> | `#f26d4b` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#e0442f;border:1px solid #999;vertical-align:middle"></span> | `#e0442f` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#c91d13;border:1px solid #999;vertical-align:middle"></span> | `#c91d13` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#a80000;border:1px solid #999;vertical-align:middle"></span> | `#a80000` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#7f0000;border:1px solid #999;vertical-align:middle"></span> | `#7f0000` |

### `PuBu`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff7fb;border:1px solid #999;vertical-align:middle"></span> | `#fff7fb` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#f0eaf4;border:1px solid #999;vertical-align:middle"></span> | `#f0eaf4` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#dbdaeb;border:1px solid #999;vertical-align:middle"></span> | `#dbdaeb` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#c0c9e2;border:1px solid #999;vertical-align:middle"></span> | `#c0c9e2` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#9cb9d9;border:1px solid #999;vertical-align:middle"></span> | `#9cb9d9` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#73a9cf;border:1px solid #999;vertical-align:middle"></span> | `#73a9cf` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#4295c3;border:1px solid #999;vertical-align:middle"></span> | `#4295c3` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#187cb6;border:1px solid #999;vertical-align:middle"></span> | `#187cb6` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#0567a2;border:1px solid #999;vertical-align:middle"></span> | `#0567a2` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#045382;border:1px solid #999;vertical-align:middle"></span> | `#045382` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#023858;border:1px solid #999;vertical-align:middle"></span> | `#023858` |

### `PuBuGn`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff7fb;border:1px solid #999;vertical-align:middle"></span> | `#fff7fb` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#f0e7f2;border:1px solid #999;vertical-align:middle"></span> | `#f0e7f2` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#dbd8ea;border:1px solid #999;vertical-align:middle"></span> | `#dbd8ea` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#c0c9e2;border:1px solid #999;vertical-align:middle"></span> | `#c0c9e2` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#99b9d9;border:1px solid #999;vertical-align:middle"></span> | `#99b9d9` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#66a9cf;border:1px solid #999;vertical-align:middle"></span> | `#66a9cf` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#4095c3;border:1px solid #999;vertical-align:middle"></span> | `#4095c3` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#16879f;border:1px solid #999;vertical-align:middle"></span> | `#16879f` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#027976;border:1px solid #999;vertical-align:middle"></span> | `#027976` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#016451;border:1px solid #999;vertical-align:middle"></span> | `#016451` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#014636;border:1px solid #999;vertical-align:middle"></span> | `#014636` |

### `PuRd`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7f4f9;border:1px solid #999;vertical-align:middle"></span> | `#f7f4f9` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#eae5f1;border:1px solid #999;vertical-align:middle"></span> | `#eae5f1` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#dcc9e2;border:1px solid #999;vertical-align:middle"></span> | `#dcc9e2` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#d0abd3;border:1px solid #999;vertical-align:middle"></span> | `#d0abd3` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#cd8bc2;border:1px solid #999;vertical-align:middle"></span> | `#cd8bc2` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#df64af;border:1px solid #999;vertical-align:middle"></span> | `#df64af` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#e53592;border:1px solid #999;vertical-align:middle"></span> | `#e53592` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#d81b6a;border:1px solid #999;vertical-align:middle"></span> | `#d81b6a` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#b80b4e;border:1px solid #999;vertical-align:middle"></span> | `#b80b4e` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#8d003b;border:1px solid #999;vertical-align:middle"></span> | `#8d003b` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#67001f;border:1px solid #999;vertical-align:middle"></span> | `#67001f` |

### `Purples`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fcfbfd;border:1px solid #999;vertical-align:middle"></span> | `#fcfbfd` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#f2f0f7;border:1px solid #999;vertical-align:middle"></span> | `#f2f0f7` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#e2e2ef;border:1px solid #999;vertical-align:middle"></span> | `#e2e2ef` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#cecfe5;border:1px solid #999;vertical-align:middle"></span> | `#cecfe5` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#b6b6d8;border:1px solid #999;vertical-align:middle"></span> | `#b6b6d8` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#9e9ac8;border:1px solid #999;vertical-align:middle"></span> | `#9e9ac8` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#8683bd;border:1px solid #999;vertical-align:middle"></span> | `#8683bd` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#7262ac;border:1px solid #999;vertical-align:middle"></span> | `#7262ac` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#61409b;border:1px solid #999;vertical-align:middle"></span> | `#61409b` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#4f1f8b;border:1px solid #999;vertical-align:middle"></span> | `#4f1f8b` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#3f007d;border:1px solid #999;vertical-align:middle"></span> | `#3f007d` |

### `RdPu`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff7f3;border:1px solid #999;vertical-align:middle"></span> | `#fff7f3` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#fde5e2;border:1px solid #999;vertical-align:middle"></span> | `#fde5e2` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#fcd0cc;border:1px solid #999;vertical-align:middle"></span> | `#fcd0cc` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#fbb6bc;border:1px solid #999;vertical-align:middle"></span> | `#fbb6bc` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#f994b1;border:1px solid #999;vertical-align:middle"></span> | `#f994b1` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#f767a1;border:1px solid #999;vertical-align:middle"></span> | `#f767a1` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#e23e99;border:1px solid #999;vertical-align:middle"></span> | `#e23e99` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#c01588;border:1px solid #999;vertical-align:middle"></span> | `#c01588` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#99017b;border:1px solid #999;vertical-align:middle"></span> | `#99017b` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#6f0174;border:1px solid #999;vertical-align:middle"></span> | `#6f0174` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#49006a;border:1px solid #999;vertical-align:middle"></span> | `#49006a` |

### `RdYlGn`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** usable with caution (diverging; not strictly sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#a50026;border:1px solid #999;vertical-align:middle"></span> | `#a50026` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#d62f27;border:1px solid #999;vertical-align:middle"></span> | `#d62f27` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#f46d43;border:1px solid #999;vertical-align:middle"></span> | `#f46d43` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdad60;border:1px solid #999;vertical-align:middle"></span> | `#fdad60` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#fee08b;border:1px solid #999;vertical-align:middle"></span> | `#fee08b` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#feffbe;border:1px solid #999;vertical-align:middle"></span> | `#feffbe` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#d9ef8b;border:1px solid #999;vertical-align:middle"></span> | `#d9ef8b` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#a5d86a;border:1px solid #999;vertical-align:middle"></span> | `#a5d86a` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#66bd63;border:1px solid #999;vertical-align:middle"></span> | `#66bd63` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#199750;border:1px solid #999;vertical-align:middle"></span> | `#199750` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#006837;border:1px solid #999;vertical-align:middle"></span> | `#006837` |

### `Reds`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff5f0;border:1px solid #999;vertical-align:middle"></span> | `#fff5f0` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#fee5d8;border:1px solid #999;vertical-align:middle"></span> | `#fee5d8` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdcab5;border:1px solid #999;vertical-align:middle"></span> | `#fdcab5` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#fcab8f;border:1px solid #999;vertical-align:middle"></span> | `#fcab8f` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#fc8a6a;border:1px solid #999;vertical-align:middle"></span> | `#fc8a6a` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fb694a;border:1px solid #999;vertical-align:middle"></span> | `#fb694a` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#f14432;border:1px solid #999;vertical-align:middle"></span> | `#f14432` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#d92523;border:1px solid #999;vertical-align:middle"></span> | `#d92523` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#bc141a;border:1px solid #999;vertical-align:middle"></span> | `#bc141a` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#980c13;border:1px solid #999;vertical-align:middle"></span> | `#980c13` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#67000d;border:1px solid #999;vertical-align:middle"></span> | `#67000d` |

### `YlGn`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffe5;border:1px solid #999;vertical-align:middle"></span> | `#ffffe5` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#f9fdc2;border:1px solid #999;vertical-align:middle"></span> | `#f9fdc2` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#e5f5ac;border:1px solid #999;vertical-align:middle"></span> | `#e5f5ac` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#c8e99b;border:1px solid #999;vertical-align:middle"></span> | `#c8e99b` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#a2d88a;border:1px solid #999;vertical-align:middle"></span> | `#a2d88a` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#77c679;border:1px solid #999;vertical-align:middle"></span> | `#77c679` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#4cb063;border:1px solid #999;vertical-align:middle"></span> | `#4cb063` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#2f934d;border:1px solid #999;vertical-align:middle"></span> | `#2f934d` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#15793e;border:1px solid #999;vertical-align:middle"></span> | `#15793e` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#006034;border:1px solid #999;vertical-align:middle"></span> | `#006034` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#004529;border:1px solid #999;vertical-align:middle"></span> | `#004529` |

### `YlGnBu`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffd9;border:1px solid #999;vertical-align:middle"></span> | `#ffffd9` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#f1faba;border:1px solid #999;vertical-align:middle"></span> | `#f1faba` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#d6efb3;border:1px solid #999;vertical-align:middle"></span> | `#d6efb3` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#abdeb7;border:1px solid #999;vertical-align:middle"></span> | `#abdeb7` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#73c8bd;border:1px solid #999;vertical-align:middle"></span> | `#73c8bd` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#40b5c4;border:1px solid #999;vertical-align:middle"></span> | `#40b5c4` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#2498c1;border:1px solid #999;vertical-align:middle"></span> | `#2498c1` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#2072b1;border:1px solid #999;vertical-align:middle"></span> | `#2072b1` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#234da0;border:1px solid #999;vertical-align:middle"></span> | `#234da0` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#1f2f87;border:1px solid #999;vertical-align:middle"></span> | `#1f2f87` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#081d58;border:1px solid #999;vertical-align:middle"></span> | `#081d58` |

### `YlOrBr`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffe5;border:1px solid #999;vertical-align:middle"></span> | `#ffffe5` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff9c5;border:1px solid #999;vertical-align:middle"></span> | `#fff9c5` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#feeba2;border:1px solid #999;vertical-align:middle"></span> | `#feeba2` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#fed778;border:1px solid #999;vertical-align:middle"></span> | `#fed778` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#febb47;border:1px solid #999;vertical-align:middle"></span> | `#febb47` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fe9829;border:1px solid #999;vertical-align:middle"></span> | `#fe9829` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#f07818;border:1px solid #999;vertical-align:middle"></span> | `#f07818` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#d85a09;border:1px solid #999;vertical-align:middle"></span> | `#d85a09` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#b84203;border:1px solid #999;vertical-align:middle"></span> | `#b84203` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#8e3104;border:1px solid #999;vertical-align:middle"></span> | `#8e3104` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#662506;border:1px solid #999;vertical-align:middle"></span> | `#662506` |

### `YlOrRd`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffcc;border:1px solid #999;vertical-align:middle"></span> | `#ffffcc` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff1a9;border:1px solid #999;vertical-align:middle"></span> | `#fff1a9` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#fee187;border:1px solid #999;vertical-align:middle"></span> | `#fee187` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#feca66;border:1px solid #999;vertical-align:middle"></span> | `#feca66` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#feab49;border:1px solid #999;vertical-align:middle"></span> | `#feab49` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fd8c3c;border:1px solid #999;vertical-align:middle"></span> | `#fd8c3c` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#fc5b2e;border:1px solid #999;vertical-align:middle"></span> | `#fc5b2e` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ed2e21;border:1px solid #999;vertical-align:middle"></span> | `#ed2e21` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#d41020;border:1px solid #999;vertical-align:middle"></span> | `#d41020` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#b00026;border:1px solid #999;vertical-align:middle"></span> | `#b00026` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#800026;border:1px solid #999;vertical-align:middle"></span> | `#800026` |

### `afmhot`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#320000;border:1px solid #999;vertical-align:middle"></span> | `#320000` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#660000;border:1px solid #999;vertical-align:middle"></span> | `#660000` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#981800;border:1px solid #999;vertical-align:middle"></span> | `#981800` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#cc4d00;border:1px solid #999;vertical-align:middle"></span> | `#cc4d00` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff8101;border:1px solid #999;vertical-align:middle"></span> | `#ff8101` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffb333;border:1px solid #999;vertical-align:middle"></span> | `#ffb333` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffe667;border:1px solid #999;vertical-align:middle"></span> | `#ffe667` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff99;border:1px solid #999;vertical-align:middle"></span> | `#ffff99` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffcd;border:1px solid #999;vertical-align:middle"></span> | `#ffffcd` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `autumn`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff0000;border:1px solid #999;vertical-align:middle"></span> | `#ff0000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff1900;border:1px solid #999;vertical-align:middle"></span> | `#ff1900` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff3300;border:1px solid #999;vertical-align:middle"></span> | `#ff3300` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff4c00;border:1px solid #999;vertical-align:middle"></span> | `#ff4c00` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff6600;border:1px solid #999;vertical-align:middle"></span> | `#ff6600` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff8000;border:1px solid #999;vertical-align:middle"></span> | `#ff8000` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9900;border:1px solid #999;vertical-align:middle"></span> | `#ff9900` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffb300;border:1px solid #999;vertical-align:middle"></span> | `#ffb300` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffcc00;border:1px solid #999;vertical-align:middle"></span> | `#ffcc00` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffe600;border:1px solid #999;vertical-align:middle"></span> | `#ffe600` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff00;border:1px solid #999;vertical-align:middle"></span> | `#ffff00` |

### `binary`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6e6e6;border:1px solid #999;vertical-align:middle"></span> | `#e6e6e6` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#cccccc;border:1px solid #999;vertical-align:middle"></span> | `#cccccc` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3b3b3;border:1px solid #999;vertical-align:middle"></span> | `#b3b3b3` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#999999;border:1px solid #999;vertical-align:middle"></span> | `#999999` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#7f7f7f;border:1px solid #999;vertical-align:middle"></span> | `#7f7f7f` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#666666;border:1px solid #999;vertical-align:middle"></span> | `#666666` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#4c4c4c;border:1px solid #999;vertical-align:middle"></span> | `#4c4c4c` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#333333;border:1px solid #999;vertical-align:middle"></span> | `#333333` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#191919;border:1px solid #999;vertical-align:middle"></span> | `#191919` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |

### `bone`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#16161e;border:1px solid #999;vertical-align:middle"></span> | `#16161e` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#2d2d3e;border:1px solid #999;vertical-align:middle"></span> | `#2d2d3e` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#43425d;border:1px solid #999;vertical-align:middle"></span> | `#43425d` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#595c79;border:1px solid #999;vertical-align:middle"></span> | `#595c79` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#707b90;border:1px solid #999;vertical-align:middle"></span> | `#707b90` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#869aa6;border:1px solid #999;vertical-align:middle"></span> | `#869aa6` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#9db9bc;border:1px solid #999;vertical-align:middle"></span> | `#9db9bc` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#b9d2d2;border:1px solid #999;vertical-align:middle"></span> | `#b9d2d2` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#dde9e9;border:1px solid #999;vertical-align:middle"></span> | `#dde9e9` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `cividis`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#00224e;border:1px solid #999;vertical-align:middle"></span> | `#00224e` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#083370;border:1px solid #999;vertical-align:middle"></span> | `#083370` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#35456c;border:1px solid #999;vertical-align:middle"></span> | `#35456c` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#4f576c;border:1px solid #999;vertical-align:middle"></span> | `#4f576c` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#666970;border:1px solid #999;vertical-align:middle"></span> | `#666970` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#7d7c78;border:1px solid #999;vertical-align:middle"></span> | `#7d7c78` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#948e77;border:1px solid #999;vertical-align:middle"></span> | `#948e77` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#aea371;border:1px solid #999;vertical-align:middle"></span> | `#aea371` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#c8b866;border:1px solid #999;vertical-align:middle"></span> | `#c8b866` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#e5cf52;border:1px solid #999;vertical-align:middle"></span> | `#e5cf52` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fee838;border:1px solid #999;vertical-align:middle"></span> | `#fee838` |

### `cool`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#00ffff;border:1px solid #999;vertical-align:middle"></span> | `#00ffff` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#19e6ff;border:1px solid #999;vertical-align:middle"></span> | `#19e6ff` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#33ccff;border:1px solid #999;vertical-align:middle"></span> | `#33ccff` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#4cb3ff;border:1px solid #999;vertical-align:middle"></span> | `#4cb3ff` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#6699ff;border:1px solid #999;vertical-align:middle"></span> | `#6699ff` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#807fff;border:1px solid #999;vertical-align:middle"></span> | `#807fff` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#9966ff;border:1px solid #999;vertical-align:middle"></span> | `#9966ff` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#b34cff;border:1px solid #999;vertical-align:middle"></span> | `#b34cff` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#cc33ff;border:1px solid #999;vertical-align:middle"></span> | `#cc33ff` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#e619ff;border:1px solid #999;vertical-align:middle"></span> | `#e619ff` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff00ff;border:1px solid #999;vertical-align:middle"></span> | `#ff00ff` |

### `coolwarm`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** usable with caution (diverging; not strictly sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#3b4cc0;border:1px solid #999;vertical-align:middle"></span> | `#3b4cc0` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#5977e3;border:1px solid #999;vertical-align:middle"></span> | `#5977e3` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#7b9ff9;border:1px solid #999;vertical-align:middle"></span> | `#7b9ff9` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#9ebeff;border:1px solid #999;vertical-align:middle"></span> | `#9ebeff` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#c0d4f5;border:1px solid #999;vertical-align:middle"></span> | `#c0d4f5` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#dddcdc;border:1px solid #999;vertical-align:middle"></span> | `#dddcdc` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#f2cbb7;border:1px solid #999;vertical-align:middle"></span> | `#f2cbb7` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7ac8e;border:1px solid #999;vertical-align:middle"></span> | `#f7ac8e` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ee8468;border:1px solid #999;vertical-align:middle"></span> | `#ee8468` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#d65244;border:1px solid #999;vertical-align:middle"></span> | `#d65244` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#b40426;border:1px solid #999;vertical-align:middle"></span> | `#b40426` |

### `copper`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#1f140c;border:1px solid #999;vertical-align:middle"></span> | `#1f140c` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#3f2819;border:1px solid #999;vertical-align:middle"></span> | `#3f2819` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#5e3b26;border:1px solid #999;vertical-align:middle"></span> | `#5e3b26` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#7e5033;border:1px solid #999;vertical-align:middle"></span> | `#7e5033` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#9e6440;border:1px solid #999;vertical-align:middle"></span> | `#9e6440` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#bd784c;border:1px solid #999;vertical-align:middle"></span> | `#bd784c` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#dd8c59;border:1px solid #999;vertical-align:middle"></span> | `#dd8c59` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#fc9f65;border:1px solid #999;vertical-align:middle"></span> | `#fc9f65` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffb472;border:1px solid #999;vertical-align:middle"></span> | `#ffb472` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffc77f;border:1px solid #999;vertical-align:middle"></span> | `#ffc77f` |

### `gist_earth`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#112977;border:1px solid #999;vertical-align:middle"></span> | `#112977` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#225e7c;border:1px solid #999;vertical-align:middle"></span> | `#225e7c` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#32827b;border:1px solid #999;vertical-align:middle"></span> | `#32827b` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#3e915b;border:1px solid #999;vertical-align:middle"></span> | `#3e915b` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#5ea04b;border:1px solid #999;vertical-align:middle"></span> | `#5ea04b` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#8eac56;border:1px solid #999;vertical-align:middle"></span> | `#8eac56` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#b7b65e;border:1px solid #999;vertical-align:middle"></span> | `#b7b65e` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#c4a46f;border:1px solid #999;vertical-align:middle"></span> | `#c4a46f` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#e1bfb0;border:1px solid #999;vertical-align:middle"></span> | `#e1bfb0` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdfbfb;border:1px solid #999;vertical-align:middle"></span> | `#fdfbfb` |

### `gist_gray`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#191919;border:1px solid #999;vertical-align:middle"></span> | `#191919` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#333333;border:1px solid #999;vertical-align:middle"></span> | `#333333` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#4c4c4c;border:1px solid #999;vertical-align:middle"></span> | `#4c4c4c` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#666666;border:1px solid #999;vertical-align:middle"></span> | `#666666` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#808080;border:1px solid #999;vertical-align:middle"></span> | `#808080` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#999999;border:1px solid #999;vertical-align:middle"></span> | `#999999` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3b3b3;border:1px solid #999;vertical-align:middle"></span> | `#b3b3b3` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#cccccc;border:1px solid #999;vertical-align:middle"></span> | `#cccccc` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6e6e6;border:1px solid #999;vertical-align:middle"></span> | `#e6e6e6` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `gist_heat`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#260000;border:1px solid #999;vertical-align:middle"></span> | `#260000` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#4d0000;border:1px solid #999;vertical-align:middle"></span> | `#4d0000` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#720000;border:1px solid #999;vertical-align:middle"></span> | `#720000` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#990000;border:1px solid #999;vertical-align:middle"></span> | `#990000` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#c00100;border:1px solid #999;vertical-align:middle"></span> | `#c00100` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#e53300;border:1px solid #999;vertical-align:middle"></span> | `#e53300` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff6700;border:1px solid #999;vertical-align:middle"></span> | `#ff6700` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9933;border:1px solid #999;vertical-align:middle"></span> | `#ff9933` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffcd9b;border:1px solid #999;vertical-align:middle"></span> | `#ffcd9b` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `gist_ncar`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000080;border:1px solid #999;vertical-align:middle"></span> | `#000080` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#000ecd;border:1px solid #999;vertical-align:middle"></span> | `#000ecd` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#00edff;border:1px solid #999;vertical-align:middle"></span> | `#00edff` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#00fd3f;border:1px solid #999;vertical-align:middle"></span> | `#00fd3f` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#74e800;border:1px solid #999;vertical-align:middle"></span> | `#74e800` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#dbff20;border:1px solid #999;vertical-align:middle"></span> | `#dbff20` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffce05;border:1px solid #999;vertical-align:middle"></span> | `#ffce05` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff3400;border:1px solid #999;vertical-align:middle"></span> | `#ff3400` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#f107ff;border:1px solid #999;vertical-align:middle"></span> | `#f107ff` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ec84ef;border:1px solid #999;vertical-align:middle"></span> | `#ec84ef` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fef8fe;border:1px solid #999;vertical-align:middle"></span> | `#fef8fe` |

### `gist_rainbow`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff0029;border:1px solid #999;vertical-align:middle"></span> | `#ff0029` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff5e00;border:1px solid #999;vertical-align:middle"></span> | `#ff5e00` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffea00;border:1px solid #999;vertical-align:middle"></span> | `#ffea00` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#8dff00;border:1px solid #999;vertical-align:middle"></span> | `#8dff00` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#00ff00;border:1px solid #999;vertical-align:middle"></span> | `#00ff00` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#00ff8c;border:1px solid #999;vertical-align:middle"></span> | `#00ff8c` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#00ecff;border:1px solid #999;vertical-align:middle"></span> | `#00ecff` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#005eff;border:1px solid #999;vertical-align:middle"></span> | `#005eff` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#2a00ff;border:1px solid #999;vertical-align:middle"></span> | `#2a00ff` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#b700ff;border:1px solid #999;vertical-align:middle"></span> | `#b700ff` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff00bf;border:1px solid #999;vertical-align:middle"></span> | `#ff00bf` |

### `gist_yarg`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6e6e6;border:1px solid #999;vertical-align:middle"></span> | `#e6e6e6` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#cccccc;border:1px solid #999;vertical-align:middle"></span> | `#cccccc` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3b3b3;border:1px solid #999;vertical-align:middle"></span> | `#b3b3b3` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#999999;border:1px solid #999;vertical-align:middle"></span> | `#999999` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#7f7f7f;border:1px solid #999;vertical-align:middle"></span> | `#7f7f7f` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#666666;border:1px solid #999;vertical-align:middle"></span> | `#666666` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#4c4c4c;border:1px solid #999;vertical-align:middle"></span> | `#4c4c4c` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#333333;border:1px solid #999;vertical-align:middle"></span> | `#333333` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#191919;border:1px solid #999;vertical-align:middle"></span> | `#191919` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |

### `gnuplot`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#500093;border:1px solid #999;vertical-align:middle"></span> | `#500093` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#7202f3;border:1px solid #999;vertical-align:middle"></span> | `#7202f3` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#8b07f3;border:1px solid #999;vertical-align:middle"></span> | `#8b07f3` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#a11096;border:1px solid #999;vertical-align:middle"></span> | `#a11096` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#b52000;border:1px solid #999;vertical-align:middle"></span> | `#b52000` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#c63700;border:1px solid #999;vertical-align:middle"></span> | `#c63700` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#d65800;border:1px solid #999;vertical-align:middle"></span> | `#d65800` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#e48300;border:1px solid #999;vertical-align:middle"></span> | `#e48300` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#f2bb00;border:1px solid #999;vertical-align:middle"></span> | `#f2bb00` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff00;border:1px solid #999;vertical-align:middle"></span> | `#ffff00` |

### `gnuplot2`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#000064;border:1px solid #999;vertical-align:middle"></span> | `#000064` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#0000cc;border:1px solid #999;vertical-align:middle"></span> | `#0000cc` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#2600ff;border:1px solid #999;vertical-align:middle"></span> | `#2600ff` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#7800ff;border:1px solid #999;vertical-align:middle"></span> | `#7800ff` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#c92ad5;border:1px solid #999;vertical-align:middle"></span> | `#c92ad5` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff5ca3;border:1px solid #999;vertical-align:middle"></span> | `#ff5ca3` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff906f;border:1px solid #999;vertical-align:middle"></span> | `#ff906f` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffc23d;border:1px solid #999;vertical-align:middle"></span> | `#ffc23d` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff609;border:1px solid #999;vertical-align:middle"></span> | `#fff609` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `gray`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#191919;border:1px solid #999;vertical-align:middle"></span> | `#191919` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#333333;border:1px solid #999;vertical-align:middle"></span> | `#333333` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#4c4c4c;border:1px solid #999;vertical-align:middle"></span> | `#4c4c4c` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#666666;border:1px solid #999;vertical-align:middle"></span> | `#666666` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#808080;border:1px solid #999;vertical-align:middle"></span> | `#808080` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#999999;border:1px solid #999;vertical-align:middle"></span> | `#999999` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3b3b3;border:1px solid #999;vertical-align:middle"></span> | `#b3b3b3` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#cccccc;border:1px solid #999;vertical-align:middle"></span> | `#cccccc` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6e6e6;border:1px solid #999;vertical-align:middle"></span> | `#e6e6e6` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `hot`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#0b0000;border:1px solid #999;vertical-align:middle"></span> | `#0b0000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#4c0000;border:1px solid #999;vertical-align:middle"></span> | `#4c0000` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#900000;border:1px solid #999;vertical-align:middle"></span> | `#900000` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#d20000;border:1px solid #999;vertical-align:middle"></span> | `#d20000` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff1700;border:1px solid #999;vertical-align:middle"></span> | `#ff1700` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff5c00;border:1px solid #999;vertical-align:middle"></span> | `#ff5c00` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9d00;border:1px solid #999;vertical-align:middle"></span> | `#ff9d00` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffe100;border:1px solid #999;vertical-align:middle"></span> | `#ffe100` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff36;border:1px solid #999;vertical-align:middle"></span> | `#ffff36` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff9d;border:1px solid #999;vertical-align:middle"></span> | `#ffff9d` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `hsv`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff0000;border:1px solid #999;vertical-align:middle"></span> | `#ff0000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9400;border:1px solid #999;vertical-align:middle"></span> | `#ff9400` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#d1ff00;border:1px solid #999;vertical-align:middle"></span> | `#d1ff00` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#3dff00;border:1px solid #999;vertical-align:middle"></span> | `#3dff00` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#00ff5c;border:1px solid #999;vertical-align:middle"></span> | `#00ff5c` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#00fff6;border:1px solid #999;vertical-align:middle"></span> | `#00fff6` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#0074ff;border:1px solid #999;vertical-align:middle"></span> | `#0074ff` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#2500ff;border:1px solid #999;vertical-align:middle"></span> | `#2500ff` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#b900ff;border:1px solid #999;vertical-align:middle"></span> | `#b900ff` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff00ac;border:1px solid #999;vertical-align:middle"></span> | `#ff00ac` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff0018;border:1px solid #999;vertical-align:middle"></span> | `#ff0018` |

### `inferno`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000004;border:1px solid #999;vertical-align:middle"></span> | `#000004` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#160b39;border:1px solid #999;vertical-align:middle"></span> | `#160b39` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#420a68;border:1px solid #999;vertical-align:middle"></span> | `#420a68` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#6a176e;border:1px solid #999;vertical-align:middle"></span> | `#6a176e` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#932667;border:1px solid #999;vertical-align:middle"></span> | `#932667` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#bc3754;border:1px solid #999;vertical-align:middle"></span> | `#bc3754` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#dd513a;border:1px solid #999;vertical-align:middle"></span> | `#dd513a` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#f37819;border:1px solid #999;vertical-align:middle"></span> | `#f37819` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#fca50a;border:1px solid #999;vertical-align:middle"></span> | `#fca50a` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#f6d746;border:1px solid #999;vertical-align:middle"></span> | `#f6d746` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fcffa4;border:1px solid #999;vertical-align:middle"></span> | `#fcffa4` |

### `jet`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000080;border:1px solid #999;vertical-align:middle"></span> | `#000080` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#0000f1;border:1px solid #999;vertical-align:middle"></span> | `#0000f1` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#004dff;border:1px solid #999;vertical-align:middle"></span> | `#004dff` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#00b1ff;border:1px solid #999;vertical-align:middle"></span> | `#00b1ff` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#29ffce;border:1px solid #999;vertical-align:middle"></span> | `#29ffce` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#7dff7a;border:1px solid #999;vertical-align:middle"></span> | `#7dff7a` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ceff29;border:1px solid #999;vertical-align:middle"></span> | `#ceff29` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffc400;border:1px solid #999;vertical-align:middle"></span> | `#ffc400` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff6800;border:1px solid #999;vertical-align:middle"></span> | `#ff6800` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#f10800;border:1px solid #999;vertical-align:middle"></span> | `#f10800` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#800000;border:1px solid #999;vertical-align:middle"></span> | `#800000` |

### `magma`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000004;border:1px solid #999;vertical-align:middle"></span> | `#000004` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#140e36;border:1px solid #999;vertical-align:middle"></span> | `#140e36` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#3b0f70;border:1px solid #999;vertical-align:middle"></span> | `#3b0f70` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#641a80;border:1px solid #999;vertical-align:middle"></span> | `#641a80` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#8c2981;border:1px solid #999;vertical-align:middle"></span> | `#8c2981` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#b73779;border:1px solid #999;vertical-align:middle"></span> | `#b73779` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#de4968;border:1px solid #999;vertical-align:middle"></span> | `#de4968` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7705c;border:1px solid #999;vertical-align:middle"></span> | `#f7705c` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#fe9f6d;border:1px solid #999;vertical-align:middle"></span> | `#fe9f6d` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#fecf92;border:1px solid #999;vertical-align:middle"></span> | `#fecf92` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fcfdbf;border:1px solid #999;vertical-align:middle"></span> | `#fcfdbf` |

### `nipy_spectral`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#000000;border:1px solid #999;vertical-align:middle"></span> | `#000000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#870098;border:1px solid #999;vertical-align:middle"></span> | `#870098` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#0000dd;border:1px solid #999;vertical-align:middle"></span> | `#0000dd` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#0098dd;border:1px solid #999;vertical-align:middle"></span> | `#0098dd` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#00aa88;border:1px solid #999;vertical-align:middle"></span> | `#00aa88` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#00bc00;border:1px solid #999;vertical-align:middle"></span> | `#00bc00` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#00ff00;border:1px solid #999;vertical-align:middle"></span> | `#00ff00` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#efed00;border:1px solid #999;vertical-align:middle"></span> | `#efed00` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9900;border:1px solid #999;vertical-align:middle"></span> | `#ff9900` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#dc0000;border:1px solid #999;vertical-align:middle"></span> | `#dc0000` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#cccccc;border:1px solid #999;vertical-align:middle"></span> | `#cccccc` |

### `ocean`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#008000;border:1px solid #999;vertical-align:middle"></span> | `#008000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#005a19;border:1px solid #999;vertical-align:middle"></span> | `#005a19` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#003333;border:1px solid #999;vertical-align:middle"></span> | `#003333` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#000e4c;border:1px solid #999;vertical-align:middle"></span> | `#000e4c` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#001a66;border:1px solid #999;vertical-align:middle"></span> | `#001a66` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#004180;border:1px solid #999;vertical-align:middle"></span> | `#004180` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#006699;border:1px solid #999;vertical-align:middle"></span> | `#006699` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#1b8db3;border:1px solid #999;vertical-align:middle"></span> | `#1b8db3` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#66b3cc;border:1px solid #999;vertical-align:middle"></span> | `#66b3cc` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#b4dae6;border:1px solid #999;vertical-align:middle"></span> | `#b4dae6` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `pink`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#1e0000;border:1px solid #999;vertical-align:middle"></span> | `#1e0000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#684141;border:1px solid #999;vertical-align:middle"></span> | `#684141` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#915d5d;border:1px solid #999;vertical-align:middle"></span> | `#915d5d` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#af7272;border:1px solid #999;vertical-align:middle"></span> | `#af7272` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#c68b84;border:1px solid #999;vertical-align:middle"></span> | `#c68b84` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#d0ac94;border:1px solid #999;vertical-align:middle"></span> | `#d0ac94` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#dac6a1;border:1px solid #999;vertical-align:middle"></span> | `#dac6a1` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#e4dfae;border:1px solid #999;vertical-align:middle"></span> | `#e4dfae` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ededc6;border:1px solid #999;vertical-align:middle"></span> | `#ededc6` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7f7e5;border:1px solid #999;vertical-align:middle"></span> | `#f7f7e5` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `plasma`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#0d0887;border:1px solid #999;vertical-align:middle"></span> | `#0d0887` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#41049d;border:1px solid #999;vertical-align:middle"></span> | `#41049d` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#6a00a8;border:1px solid #999;vertical-align:middle"></span> | `#6a00a8` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#8f0da4;border:1px solid #999;vertical-align:middle"></span> | `#8f0da4` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#b12a90;border:1px solid #999;vertical-align:middle"></span> | `#b12a90` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#cc4778;border:1px solid #999;vertical-align:middle"></span> | `#cc4778` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#e16462;border:1px solid #999;vertical-align:middle"></span> | `#e16462` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#f2844b;border:1px solid #999;vertical-align:middle"></span> | `#f2844b` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#fca636;border:1px solid #999;vertical-align:middle"></span> | `#fca636` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#fcce25;border:1px solid #999;vertical-align:middle"></span> | `#fcce25` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#f0f921;border:1px solid #999;vertical-align:middle"></span> | `#f0f921` |

### `prism`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff0000;border:1px solid #999;vertical-align:middle"></span> | `#ff0000` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff0000;border:1px solid #999;vertical-align:middle"></span> | `#ff0000` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff2a00;border:1px solid #999;vertical-align:middle"></span> | `#ff2a00` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff4800;border:1px solid #999;vertical-align:middle"></span> | `#ff4800` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9500;border:1px solid #999;vertical-align:middle"></span> | `#ff9500` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffd700;border:1px solid #999;vertical-align:middle"></span> | `#ffd700` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffeb00;border:1px solid #999;vertical-align:middle"></span> | `#ffeb00` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#daff00;border:1px solid #999;vertical-align:middle"></span> | `#daff00` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#bdff00;border:1px solid #999;vertical-align:middle"></span> | `#bdff00` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#6fff00;border:1px solid #999;vertical-align:middle"></span> | `#6fff00` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#54ff00;border:1px solid #999;vertical-align:middle"></span> | `#54ff00` |

### `rainbow`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#8000ff;border:1px solid #999;vertical-align:middle"></span> | `#8000ff` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#4e4dfc;border:1px solid #999;vertical-align:middle"></span> | `#4e4dfc` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#1996f3;border:1px solid #999;vertical-align:middle"></span> | `#1996f3` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#18cde4;border:1px solid #999;vertical-align:middle"></span> | `#18cde4` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#4df3ce;border:1px solid #999;vertical-align:middle"></span> | `#4df3ce` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#81ffb4;border:1px solid #999;vertical-align:middle"></span> | `#81ffb4` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3f396;border:1px solid #999;vertical-align:middle"></span> | `#b3f396` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6cd73;border:1px solid #999;vertical-align:middle"></span> | `#e6cd73` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff964f;border:1px solid #999;vertical-align:middle"></span> | `#ff964f` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff4d27;border:1px solid #999;vertical-align:middle"></span> | `#ff4d27` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff0000;border:1px solid #999;vertical-align:middle"></span> | `#ff0000` |

### `seismic`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** usable with caution (diverging; not strictly sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#00004d;border:1px solid #999;vertical-align:middle"></span> | `#00004d` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#000093;border:1px solid #999;vertical-align:middle"></span> | `#000093` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#0000db;border:1px solid #999;vertical-align:middle"></span> | `#0000db` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#3131ff;border:1px solid #999;vertical-align:middle"></span> | `#3131ff` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#9999ff;border:1px solid #999;vertical-align:middle"></span> | `#9999ff` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fffdfd;border:1px solid #999;vertical-align:middle"></span> | `#fffdfd` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9999;border:1px solid #999;vertical-align:middle"></span> | `#ff9999` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff3131;border:1px solid #999;vertical-align:middle"></span> | `#ff3131` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#e60000;border:1px solid #999;vertical-align:middle"></span> | `#e60000` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#b20000;border:1px solid #999;vertical-align:middle"></span> | `#b20000` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#800000;border:1px solid #999;vertical-align:middle"></span> | `#800000` |

### `spring`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff00ff;border:1px solid #999;vertical-align:middle"></span> | `#ff00ff` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff19e6;border:1px solid #999;vertical-align:middle"></span> | `#ff19e6` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff33cc;border:1px solid #999;vertical-align:middle"></span> | `#ff33cc` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff4cb3;border:1px solid #999;vertical-align:middle"></span> | `#ff4cb3` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff6699;border:1px solid #999;vertical-align:middle"></span> | `#ff6699` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff807f;border:1px solid #999;vertical-align:middle"></span> | `#ff807f` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9966;border:1px solid #999;vertical-align:middle"></span> | `#ff9966` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffb34c;border:1px solid #999;vertical-align:middle"></span> | `#ffb34c` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffcc33;border:1px solid #999;vertical-align:middle"></span> | `#ffcc33` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffe619;border:1px solid #999;vertical-align:middle"></span> | `#ffe619` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff00;border:1px solid #999;vertical-align:middle"></span> | `#ffff00` |

### `summer`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#008066;border:1px solid #999;vertical-align:middle"></span> | `#008066` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#198c66;border:1px solid #999;vertical-align:middle"></span> | `#198c66` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#339966;border:1px solid #999;vertical-align:middle"></span> | `#339966` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#4ca666;border:1px solid #999;vertical-align:middle"></span> | `#4ca666` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#66b366;border:1px solid #999;vertical-align:middle"></span> | `#66b366` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#80c066;border:1px solid #999;vertical-align:middle"></span> | `#80c066` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#99cc66;border:1px solid #999;vertical-align:middle"></span> | `#99cc66` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3d966;border:1px solid #999;vertical-align:middle"></span> | `#b3d966` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#cce666;border:1px solid #999;vertical-align:middle"></span> | `#cce666` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6f366;border:1px solid #999;vertical-align:middle"></span> | `#e6f366` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff66;border:1px solid #999;vertical-align:middle"></span> | `#ffff66` |

### `tab20b`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#393b79;border:1px solid #999;vertical-align:middle"></span> | `#393b79` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#6b6ecf;border:1px solid #999;vertical-align:middle"></span> | `#6b6ecf` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#637939;border:1px solid #999;vertical-align:middle"></span> | `#637939` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#b5cf6b;border:1px solid #999;vertical-align:middle"></span> | `#b5cf6b` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#8c6d31;border:1px solid #999;vertical-align:middle"></span> | `#8c6d31` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#e7ba52;border:1px solid #999;vertical-align:middle"></span> | `#e7ba52` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#843c39;border:1px solid #999;vertical-align:middle"></span> | `#843c39` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#d6616b;border:1px solid #999;vertical-align:middle"></span> | `#d6616b` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#7b4173;border:1px solid #999;vertical-align:middle"></span> | `#7b4173` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#ce6dbd;border:1px solid #999;vertical-align:middle"></span> | `#ce6dbd` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#de9ed6;border:1px solid #999;vertical-align:middle"></span> | `#de9ed6` |

### `tab20c`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#3182bd;border:1px solid #999;vertical-align:middle"></span> | `#3182bd` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#9ecae1;border:1px solid #999;vertical-align:middle"></span> | `#9ecae1` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6550d;border:1px solid #999;vertical-align:middle"></span> | `#e6550d` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdae6b;border:1px solid #999;vertical-align:middle"></span> | `#fdae6b` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#31a354;border:1px solid #999;vertical-align:middle"></span> | `#31a354` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#a1d99b;border:1px solid #999;vertical-align:middle"></span> | `#a1d99b` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#756bb1;border:1px solid #999;vertical-align:middle"></span> | `#756bb1` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#bcbddc;border:1px solid #999;vertical-align:middle"></span> | `#bcbddc` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#636363;border:1px solid #999;vertical-align:middle"></span> | `#636363` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#bdbdbd;border:1px solid #999;vertical-align:middle"></span> | `#bdbdbd` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#d9d9d9;border:1px solid #999;vertical-align:middle"></span> | `#d9d9d9` |

### `terrain`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#333399;border:1px solid #999;vertical-align:middle"></span> | `#333399` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#1276dc;border:1px solid #999;vertical-align:middle"></span> | `#1276dc` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#00b3b3;border:1px solid #999;vertical-align:middle"></span> | `#00b3b3` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#31d670;border:1px solid #999;vertical-align:middle"></span> | `#31d670` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#99eb85;border:1px solid #999;vertical-align:middle"></span> | `#99eb85` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fefe98;border:1px solid #999;vertical-align:middle"></span> | `#fefe98` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#ccbe7d;border:1px solid #999;vertical-align:middle"></span> | `#ccbe7d` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#987b61;border:1px solid #999;vertical-align:middle"></span> | `#987b61` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#997c76;border:1px solid #999;vertical-align:middle"></span> | `#997c76` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#cdbfbc;border:1px solid #999;vertical-align:middle"></span> | `#cdbfbc` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffff;border:1px solid #999;vertical-align:middle"></span> | `#ffffff` |

### `turbo`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#30123b;border:1px solid #999;vertical-align:middle"></span> | `#30123b` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#4559cb;border:1px solid #999;vertical-align:middle"></span> | `#4559cb` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#3e9bfe;border:1px solid #999;vertical-align:middle"></span> | `#3e9bfe` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#19d5cd;border:1px solid #999;vertical-align:middle"></span> | `#19d5cd` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#46f884;border:1px solid #999;vertical-align:middle"></span> | `#46f884` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#a4fc3c;border:1px solid #999;vertical-align:middle"></span> | `#a4fc3c` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#e1dd37;border:1px solid #999;vertical-align:middle"></span> | `#e1dd37` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#fea431;border:1px solid #999;vertical-align:middle"></span> | `#fea431` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#f05b12;border:1px solid #999;vertical-align:middle"></span> | `#f05b12` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#c32503;border:1px solid #999;vertical-align:middle"></span> | `#c32503` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#7a0403;border:1px solid #999;vertical-align:middle"></span> | `#7a0403` |

### `twilight`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#e2d9e2;border:1px solid #999;vertical-align:middle"></span> | `#e2d9e2` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#a6bfca;border:1px solid #999;vertical-align:middle"></span> | `#a6bfca` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#6d90c0;border:1px solid #999;vertical-align:middle"></span> | `#6d90c0` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#5f58b0;border:1px solid #999;vertical-align:middle"></span> | `#5f58b0` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#531e7c;border:1px solid #999;vertical-align:middle"></span> | `#531e7c` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#2f1436;border:1px solid #999;vertical-align:middle"></span> | `#2f1436` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#64194b;border:1px solid #999;vertical-align:middle"></span> | `#64194b` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#9f3c50;border:1px solid #999;vertical-align:middle"></span> | `#9f3c50` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#c0755e;border:1px solid #999;vertical-align:middle"></span> | `#c0755e` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#d0b39e;border:1px solid #999;vertical-align:middle"></span> | `#d0b39e` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#e2d9e2;border:1px solid #999;vertical-align:middle"></span> | `#e2d9e2` |

### `twilight_shifted`

**Continuous:** yes  
**Sequential / ordered:** no  
**Confidence use:** not recommended (non-sequential)

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#301437;border:1px solid #999;vertical-align:middle"></span> | `#301437` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#541e7e;border:1px solid #999;vertical-align:middle"></span> | `#541e7e` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#5f59b1;border:1px solid #999;vertical-align:middle"></span> | `#5f59b1` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#6e91c0;border:1px solid #999;vertical-align:middle"></span> | `#6e91c0` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#a7c0cb;border:1px solid #999;vertical-align:middle"></span> | `#a7c0cb` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#e2d9e2;border:1px solid #999;vertical-align:middle"></span> | `#e2d9e2` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#d0b29c;border:1px solid #999;vertical-align:middle"></span> | `#d0b29c` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#c0745d;border:1px solid #999;vertical-align:middle"></span> | `#c0745d` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#9e3b50;border:1px solid #999;vertical-align:middle"></span> | `#9e3b50` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#63184b;border:1px solid #999;vertical-align:middle"></span> | `#63184b` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#2f1436;border:1px solid #999;vertical-align:middle"></span> | `#2f1436` |

### `viridis`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#440154;border:1px solid #999;vertical-align:middle"></span> | `#440154` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#482475;border:1px solid #999;vertical-align:middle"></span> | `#482475` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#414487;border:1px solid #999;vertical-align:middle"></span> | `#414487` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#355f8d;border:1px solid #999;vertical-align:middle"></span> | `#355f8d` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#2a788e;border:1px solid #999;vertical-align:middle"></span> | `#2a788e` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#21918c;border:1px solid #999;vertical-align:middle"></span> | `#21918c` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#22a884;border:1px solid #999;vertical-align:middle"></span> | `#22a884` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#44bf70;border:1px solid #999;vertical-align:middle"></span> | `#44bf70` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#7ad151;border:1px solid #999;vertical-align:middle"></span> | `#7ad151` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#bddf26;border:1px solid #999;vertical-align:middle"></span> | `#bddf26` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fde725;border:1px solid #999;vertical-align:middle"></span> | `#fde725` |

### `winter`

**Continuous:** yes  
**Sequential / ordered:** yes  
**Confidence use:** recommended

| value | color | hex |
|---:|:---:|:---|
| 0.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#0000ff;border:1px solid #999;vertical-align:middle"></span> | `#0000ff` |
| 0.1 | <span style="display:inline-block;width:3em;height:1.2em;background:#0019f3;border:1px solid #999;vertical-align:middle"></span> | `#0019f3` |
| 0.2 | <span style="display:inline-block;width:3em;height:1.2em;background:#0033e6;border:1px solid #999;vertical-align:middle"></span> | `#0033e6` |
| 0.3 | <span style="display:inline-block;width:3em;height:1.2em;background:#004cd9;border:1px solid #999;vertical-align:middle"></span> | `#004cd9` |
| 0.4 | <span style="display:inline-block;width:3em;height:1.2em;background:#0066cc;border:1px solid #999;vertical-align:middle"></span> | `#0066cc` |
| 0.5 | <span style="display:inline-block;width:3em;height:1.2em;background:#0080bf;border:1px solid #999;vertical-align:middle"></span> | `#0080bf` |
| 0.6 | <span style="display:inline-block;width:3em;height:1.2em;background:#0099b3;border:1px solid #999;vertical-align:middle"></span> | `#0099b3` |
| 0.7 | <span style="display:inline-block;width:3em;height:1.2em;background:#00b3a6;border:1px solid #999;vertical-align:middle"></span> | `#00b3a6` |
| 0.8 | <span style="display:inline-block;width:3em;height:1.2em;background:#00cc99;border:1px solid #999;vertical-align:middle"></span> | `#00cc99` |
| 0.9 | <span style="display:inline-block;width:3em;height:1.2em;background:#00e68c;border:1px solid #999;vertical-align:middle"></span> | `#00e68c` |
| 1.0 | <span style="display:inline-block;width:3em;height:1.2em;background:#00ff80;border:1px solid #999;vertical-align:middle"></span> | `#00ff80` |

## Qualitative maps

These maps assign discrete colors to classes and are not continuous or ordered for confidence.

### `Accent`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#7fc97f;border:1px solid #999;vertical-align:middle"></span> | `#7fc97f` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#beaed4;border:1px solid #999;vertical-align:middle"></span> | `#beaed4` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdc086;border:1px solid #999;vertical-align:middle"></span> | `#fdc086` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff99;border:1px solid #999;vertical-align:middle"></span> | `#ffff99` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#386cb0;border:1px solid #999;vertical-align:middle"></span> | `#386cb0` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#f0027f;border:1px solid #999;vertical-align:middle"></span> | `#f0027f` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#bf5b17;border:1px solid #999;vertical-align:middle"></span> | `#bf5b17` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#666666;border:1px solid #999;vertical-align:middle"></span> | `#666666` |

### `Dark2`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#1b9e77;border:1px solid #999;vertical-align:middle"></span> | `#1b9e77` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#d95f02;border:1px solid #999;vertical-align:middle"></span> | `#d95f02` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#7570b3;border:1px solid #999;vertical-align:middle"></span> | `#7570b3` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#e7298a;border:1px solid #999;vertical-align:middle"></span> | `#e7298a` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#66a61e;border:1px solid #999;vertical-align:middle"></span> | `#66a61e` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6ab02;border:1px solid #999;vertical-align:middle"></span> | `#e6ab02` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#a6761d;border:1px solid #999;vertical-align:middle"></span> | `#a6761d` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#666666;border:1px solid #999;vertical-align:middle"></span> | `#666666` |

### `Paired`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#a6cee3;border:1px solid #999;vertical-align:middle"></span> | `#a6cee3` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#1f78b4;border:1px solid #999;vertical-align:middle"></span> | `#1f78b4` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#b2df8a;border:1px solid #999;vertical-align:middle"></span> | `#b2df8a` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#33a02c;border:1px solid #999;vertical-align:middle"></span> | `#33a02c` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#fb9a99;border:1px solid #999;vertical-align:middle"></span> | `#fb9a99` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#e31a1c;border:1px solid #999;vertical-align:middle"></span> | `#e31a1c` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdbf6f;border:1px solid #999;vertical-align:middle"></span> | `#fdbf6f` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff7f00;border:1px solid #999;vertical-align:middle"></span> | `#ff7f00` |
| 8 | <span style="display:inline-block;width:3em;height:1.2em;background:#cab2d6;border:1px solid #999;vertical-align:middle"></span> | `#cab2d6` |
| 9 | <span style="display:inline-block;width:3em;height:1.2em;background:#6a3d9a;border:1px solid #999;vertical-align:middle"></span> | `#6a3d9a` |
| 10 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff99;border:1px solid #999;vertical-align:middle"></span> | `#ffff99` |
| 11 | <span style="display:inline-block;width:3em;height:1.2em;background:#b15928;border:1px solid #999;vertical-align:middle"></span> | `#b15928` |

### `Pastel1`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#fbb4ae;border:1px solid #999;vertical-align:middle"></span> | `#fbb4ae` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3cde3;border:1px solid #999;vertical-align:middle"></span> | `#b3cde3` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#ccebc5;border:1px solid #999;vertical-align:middle"></span> | `#ccebc5` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#decbe4;border:1px solid #999;vertical-align:middle"></span> | `#decbe4` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#fed9a6;border:1px solid #999;vertical-align:middle"></span> | `#fed9a6` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffcc;border:1px solid #999;vertical-align:middle"></span> | `#ffffcc` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#e5d8bd;border:1px solid #999;vertical-align:middle"></span> | `#e5d8bd` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#fddaec;border:1px solid #999;vertical-align:middle"></span> | `#fddaec` |
| 8 | <span style="display:inline-block;width:3em;height:1.2em;background:#f2f2f2;border:1px solid #999;vertical-align:middle"></span> | `#f2f2f2` |

### `Pastel2`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3e2cd;border:1px solid #999;vertical-align:middle"></span> | `#b3e2cd` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdcdac;border:1px solid #999;vertical-align:middle"></span> | `#fdcdac` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#cbd5e8;border:1px solid #999;vertical-align:middle"></span> | `#cbd5e8` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#f4cae4;border:1px solid #999;vertical-align:middle"></span> | `#f4cae4` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#e6f5c9;border:1px solid #999;vertical-align:middle"></span> | `#e6f5c9` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fff2ae;border:1px solid #999;vertical-align:middle"></span> | `#fff2ae` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#f1e2cc;border:1px solid #999;vertical-align:middle"></span> | `#f1e2cc` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#cccccc;border:1px solid #999;vertical-align:middle"></span> | `#cccccc` |

### `Set1`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#e41a1c;border:1px solid #999;vertical-align:middle"></span> | `#e41a1c` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#377eb8;border:1px solid #999;vertical-align:middle"></span> | `#377eb8` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#4daf4a;border:1px solid #999;vertical-align:middle"></span> | `#4daf4a` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#984ea3;border:1px solid #999;vertical-align:middle"></span> | `#984ea3` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff7f00;border:1px solid #999;vertical-align:middle"></span> | `#ff7f00` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffff33;border:1px solid #999;vertical-align:middle"></span> | `#ffff33` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#a65628;border:1px solid #999;vertical-align:middle"></span> | `#a65628` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#f781bf;border:1px solid #999;vertical-align:middle"></span> | `#f781bf` |
| 8 | <span style="display:inline-block;width:3em;height:1.2em;background:#999999;border:1px solid #999;vertical-align:middle"></span> | `#999999` |

### `Set2`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#66c2a5;border:1px solid #999;vertical-align:middle"></span> | `#66c2a5` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#fc8d62;border:1px solid #999;vertical-align:middle"></span> | `#fc8d62` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#8da0cb;border:1px solid #999;vertical-align:middle"></span> | `#8da0cb` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#e78ac3;border:1px solid #999;vertical-align:middle"></span> | `#e78ac3` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#a6d854;border:1px solid #999;vertical-align:middle"></span> | `#a6d854` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffd92f;border:1px solid #999;vertical-align:middle"></span> | `#ffd92f` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#e5c494;border:1px solid #999;vertical-align:middle"></span> | `#e5c494` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3b3b3;border:1px solid #999;vertical-align:middle"></span> | `#b3b3b3` |

### `Set3`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#8dd3c7;border:1px solid #999;vertical-align:middle"></span> | `#8dd3c7` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffffb3;border:1px solid #999;vertical-align:middle"></span> | `#ffffb3` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#bebada;border:1px solid #999;vertical-align:middle"></span> | `#bebada` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#fb8072;border:1px solid #999;vertical-align:middle"></span> | `#fb8072` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#80b1d3;border:1px solid #999;vertical-align:middle"></span> | `#80b1d3` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#fdb462;border:1px solid #999;vertical-align:middle"></span> | `#fdb462` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#b3de69;border:1px solid #999;vertical-align:middle"></span> | `#b3de69` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#fccde5;border:1px solid #999;vertical-align:middle"></span> | `#fccde5` |
| 8 | <span style="display:inline-block;width:3em;height:1.2em;background:#d9d9d9;border:1px solid #999;vertical-align:middle"></span> | `#d9d9d9` |
| 9 | <span style="display:inline-block;width:3em;height:1.2em;background:#bc80bd;border:1px solid #999;vertical-align:middle"></span> | `#bc80bd` |
| 10 | <span style="display:inline-block;width:3em;height:1.2em;background:#ccebc5;border:1px solid #999;vertical-align:middle"></span> | `#ccebc5` |
| 11 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffed6f;border:1px solid #999;vertical-align:middle"></span> | `#ffed6f` |

### `tab10`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#1f77b4;border:1px solid #999;vertical-align:middle"></span> | `#1f77b4` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff7f0e;border:1px solid #999;vertical-align:middle"></span> | `#ff7f0e` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#2ca02c;border:1px solid #999;vertical-align:middle"></span> | `#2ca02c` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#d62728;border:1px solid #999;vertical-align:middle"></span> | `#d62728` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#9467bd;border:1px solid #999;vertical-align:middle"></span> | `#9467bd` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#8c564b;border:1px solid #999;vertical-align:middle"></span> | `#8c564b` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#e377c2;border:1px solid #999;vertical-align:middle"></span> | `#e377c2` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#7f7f7f;border:1px solid #999;vertical-align:middle"></span> | `#7f7f7f` |
| 8 | <span style="display:inline-block;width:3em;height:1.2em;background:#bcbd22;border:1px solid #999;vertical-align:middle"></span> | `#bcbd22` |
| 9 | <span style="display:inline-block;width:3em;height:1.2em;background:#17becf;border:1px solid #999;vertical-align:middle"></span> | `#17becf` |

### `tab20`

**Continuous:** no  
**Sequential / ordered:** no  
**Confidence use:** not recommended

| index | color | hex |
|---:|:---:|:---|
| 0 | <span style="display:inline-block;width:3em;height:1.2em;background:#1f77b4;border:1px solid #999;vertical-align:middle"></span> | `#1f77b4` |
| 1 | <span style="display:inline-block;width:3em;height:1.2em;background:#aec7e8;border:1px solid #999;vertical-align:middle"></span> | `#aec7e8` |
| 2 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff7f0e;border:1px solid #999;vertical-align:middle"></span> | `#ff7f0e` |
| 3 | <span style="display:inline-block;width:3em;height:1.2em;background:#ffbb78;border:1px solid #999;vertical-align:middle"></span> | `#ffbb78` |
| 4 | <span style="display:inline-block;width:3em;height:1.2em;background:#2ca02c;border:1px solid #999;vertical-align:middle"></span> | `#2ca02c` |
| 5 | <span style="display:inline-block;width:3em;height:1.2em;background:#98df8a;border:1px solid #999;vertical-align:middle"></span> | `#98df8a` |
| 6 | <span style="display:inline-block;width:3em;height:1.2em;background:#d62728;border:1px solid #999;vertical-align:middle"></span> | `#d62728` |
| 7 | <span style="display:inline-block;width:3em;height:1.2em;background:#ff9896;border:1px solid #999;vertical-align:middle"></span> | `#ff9896` |
| 8 | <span style="display:inline-block;width:3em;height:1.2em;background:#9467bd;border:1px solid #999;vertical-align:middle"></span> | `#9467bd` |
| 9 | <span style="display:inline-block;width:3em;height:1.2em;background:#c5b0d5;border:1px solid #999;vertical-align:middle"></span> | `#c5b0d5` |
| 10 | <span style="display:inline-block;width:3em;height:1.2em;background:#8c564b;border:1px solid #999;vertical-align:middle"></span> | `#8c564b` |
| 11 | <span style="display:inline-block;width:3em;height:1.2em;background:#c49c94;border:1px solid #999;vertical-align:middle"></span> | `#c49c94` |
| 12 | <span style="display:inline-block;width:3em;height:1.2em;background:#e377c2;border:1px solid #999;vertical-align:middle"></span> | `#e377c2` |
| 13 | <span style="display:inline-block;width:3em;height:1.2em;background:#f7b6d2;border:1px solid #999;vertical-align:middle"></span> | `#f7b6d2` |
| 14 | <span style="display:inline-block;width:3em;height:1.2em;background:#7f7f7f;border:1px solid #999;vertical-align:middle"></span> | `#7f7f7f` |
| 15 | <span style="display:inline-block;width:3em;height:1.2em;background:#c7c7c7;border:1px solid #999;vertical-align:middle"></span> | `#c7c7c7` |
| 16 | <span style="display:inline-block;width:3em;height:1.2em;background:#bcbd22;border:1px solid #999;vertical-align:middle"></span> | `#bcbd22` |
| 17 | <span style="display:inline-block;width:3em;height:1.2em;background:#dbdb8d;border:1px solid #999;vertical-align:middle"></span> | `#dbdb8d` |
| 18 | <span style="display:inline-block;width:3em;height:1.2em;background:#17becf;border:1px solid #999;vertical-align:middle"></span> | `#17becf` |
| 19 | <span style="display:inline-block;width:3em;height:1.2em;background:#9edae5;border:1px solid #999;vertical-align:middle"></span> | `#9edae5` |

