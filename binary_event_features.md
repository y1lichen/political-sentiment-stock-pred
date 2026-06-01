# Trump Posts Binary Event Features

本文標註 `data/text/trump_posts_features_2017_2026.csv` 這份原始資料檔中，哪些欄位會被目前專案使用，以及程式會從它們衍生出哪些二元事件 feature。

重點區分：

- 原始 CSV 直接存在的二元欄位：`kw_*` 類欄位。
- Post-level 衍生二元欄位：由 `Content`、`Timestamp`、`kw_*` 產生，每篇貼文一列。
- Daily-level 二元事件 feature：依市場時間對齊後聚合到每日，實際進入 `event_combo.py` 的 brute-force event selection。

---

## 1. 原始 CSV 中會用到的二元欄位

原始檔：

```text
data/text/trump_posts_features_2017_2026.csv
```

目前事件規則會直接參考以下原始二元欄位：

| 原始欄位 | 是否使用 | 用途 |
| :-- | :--: | :-- |
| `kw_china` | yes | 產生 `ev_china`，再聚合為 `has_china`、`china_*` 類事件 |
| `kw_taiwan` | yes | 產生 `ev_taiwan`，再聚合為 `has_taiwan`、`*_taiwan_*` 類事件 |
| `kw_tariffs` | yes | 產生 `ev_tariff`，再聚合為 `has_tariff`、`tariff_*` 類事件 |
| `kw_chips` | yes | 產生 `ev_chips`，再聚合為 `has_chips`、`chips_*` 類事件 |
| `kw_ai` | yes | 產生 `ev_ai`，再聚合為 `has_ai` |

以下原始二元欄位存在於 CSV，但目前正式事件規則沒有直接轉成模型事件 feature：

| 原始欄位 | 目前狀態 |
| :-- | :-- |
| `kw_sanctions` | 保留，尚未納入正式事件 feature |
| `kw_tech` | 保留，尚未納入正式事件 feature |

另外，`vader_compound`、`emotion_label`、`emotion_score`、`weighted_vader`、`Likes`、`Retweets`、`Platform` 目前也沒有進入正式事件 feature。

---

## 2. Post-level 衍生二元欄位

這些欄位由 `data/data_preprocess.py` 的 `add_post_level_features()` 產生，或由 `event_combo.py` 在聚合時預期讀取。它們是每篇貼文層級的二元事件，不是原始 CSV 直接提供的欄位。

| Feature | 來源 | 說明 |
| :-- | :-- | :-- |
| `is_pre_market_post` | `Timestamp` | 是否在該市場開盤前發文 |
| `is_market_open_post` | `Timestamp` | 是否在該市場盤中發文 |
| `is_night_post` | `Timestamp` | 是否在深夜時段發文 |
| `is_all_caps_post` | `Content` | 類全大寫貼文；alphabetic 字元至少 10 個且大寫比例 >= 0.8 |
| `has_exclamation_mark` | `Content` | 該貼文是否至少包含一個驚嘆號 |
| `has_uppercase_phrase` | `Content` | 是否出現至少 1 段連續 5 個以上全大寫單字；空白與標點符號只作分隔 |
| `ev_tariff` | `Content` + `kw_tariffs` | 是否提到 tariff / duty / duties |
| `ev_deal` | `Content` | 是否提到 deal / agreement / signed / negotiation |
| `ev_relief` | `Content` | 是否提到 pause / exempt / suspend / delay / relief |
| `ev_action` | `Content` | 是否有 executive order、immediately、ordered 等行動語氣 |
| `ev_attack` | `Content` | 是否有 fake news、corrupt、fraud、hoax 等攻擊語氣 |
| `ev_positive` | `Content` | 是否有 great、tremendous、historic、strong、win 等正向語氣 |
| `ev_market_brag` | `Content` | 是否提到 stock market、record high、Dow、Nasdaq、S&P 等市場表現 |
| `ev_china` | `Content` + `kw_china` | 是否提到 China / Chinese / Beijing |
| `ev_taiwan` | `Content` + `kw_taiwan` | 是否提到 Taiwan / TSMC / Taipei |
| `ev_chips` | `Content` + `kw_chips` | 是否提到 chip / semiconductor / Nvidia / TSMC |
| `ev_ai` | `Content` + `kw_ai` | 是否提到 AI / artificial intelligence / OpenAI / ChatGPT |
| `ev_iran` | `Content` | 是否提到 Iran |
| `ev_russia` | `Content` | 是否提到 Russia / Putin / Ukraine |
| `sig_djt` | `Content` | 是否出現 `President DJT` 簽名 |
| `sig_potus` | `Content` | 是否出現 `PRESIDENT OF THE UNITED STATES` |
| `sig_tyfa` | `Content` | 是否出現 `Thank you for your attention` |

---

## 3. Daily-level 實際使用的二元事件 feature

以下 56 個欄位是目前 `event_combo.py` 中 `binary_features_from_interval_counts()` 會產生，並由 `make_event_combos()` 拿去做 brute-force event selection 的二元事件 feature。

### 3.1 發文量與沉默事件

| Feature | 定義 |
| :-- | :-- |
| `posts_high` | 當日貼文數 >= 20 |
| `posts_low` | 當日貼文數 <= 5 |
| `posts_very_high` | 當日貼文數 >= 35 |
| `silence_day` | 當日貼文數 = 0 |

### 3.2 關稅、交易協議與政策緩和

| Feature | 定義 |
| :-- | :-- |
| `has_tariff` | tariff 相關貼文數 >= 1 |
| `tariff_heavy` | tariff 相關貼文數 >= 3 |
| `has_deal` | deal 相關貼文數 >= 1 |
| `deal_heavy` | deal 相關貼文數 >= 2 |
| `has_relief` | relief / pause / delay 類貼文數 >= 1 |
| `has_action` | action / executive order 類貼文數 >= 1 |

### 3.3 語氣與市場炫耀

| Feature | 定義 |
| :-- | :-- |
| `has_attack` | attack 類貼文數 >= 1 |
| `attack_heavy` | attack 類貼文數 >= 3 |
| `has_positive` | positive 類貼文數 >= 1 |
| `positive_heavy` | positive 類貼文數 >= 3 |
| `has_market_brag` | stock market / record high 類貼文數 >= 1 |
| `brag_heavy` | market brag 類貼文數 >= 2 |

### 3.4 國家、地緣政治與科技主題

| Feature | 定義 |
| :-- | :-- |
| `has_china` | China 類貼文數 >= 1 |
| `has_iran` | Iran 類貼文數 >= 1 |
| `has_russia` | Russia / Putin / Ukraine 類貼文數 >= 1 |
| `has_taiwan` | Taiwan / TSMC / Taipei 類貼文數 >= 1 |
| `has_chips` | chip / semiconductor / Nvidia / TSMC 類貼文數 >= 1 |
| `chips_heavy` | chips 類貼文數 >= 2 |
| `has_ai` | AI 類貼文數 >= 1 |

### 3.5 發文時間條件事件

| Feature | 定義 |
| :-- | :-- |
| `pre_tariff` | 盤前 tariff 貼文數 >= 1 |
| `pre_deal` | 盤前 deal 貼文數 >= 1 |
| `pre_relief` | 盤前 relief 貼文數 >= 1 |
| `pre_action` | 盤前 action 貼文數 >= 1 |
| `open_tariff` | 盤中 tariff 貼文數 >= 1 |
| `open_tariff_heavy` | 盤中 tariff 貼文數 >= 2 |
| `open_deal` | 盤中 deal 貼文數 >= 1 |
| `has_night_post` | 深夜貼文數 >= 1 |

### 3.6 簽名與文字型態

| Feature | 定義 |
| :-- | :-- |
| `sig_djt` | `President DJT` 簽名數 >= 1 |
| `sig_potus` | `PRESIDENT OF THE UNITED STATES` 簽名數 >= 1 |
| `sig_tyfa` | `Thank you for your attention` 簽名數 >= 1 |
| `high_emotion` | 大寫字母比例 > 0.2 |
| `has_all_caps` | 類全大寫貼文數 >= 1；定義為 alphabetic 字元至少 10 個且大寫比例 >= 0.8 |
| `all_caps_heavy` | 類全大寫貼文數 >= 3 |
| `has_uppercase_phrase` | 至少 1 篇貼文含有連續 5 個以上全大寫單字 |
| `uppercase_phrase_heavy` | 含連續 5 個以上全大寫單字的貼文數 >= 3 |
| `has_exclamation` | 含驚嘆號的貼文數 >= 1 |
| `exclamation_heavy` | 含驚嘆號的貼文數 >= 3 |
| `lots_of_excl` | 驚嘆號總數 >= 5 |
| `long_posts` | 平均貼文長度 > 400 |
| `short_posts` | 平均貼文長度 < 150 且當日有貼文 |

### 3.7 組合型與時序型事件

| Feature | 定義 |
| :-- | :-- |
| `deal_over_tariff` | deal_count > tariff_count 且 deal_count >= 1 |
| `tariff_only` | tariff_count >= 1 且 deal_count = 0 |
| `deal_only` | deal_count >= 1 且 tariff_count = 0 |
| `china_taiwan_combo` | 同日 China 與 Taiwan 事件同時出現 |
| `china_chips_combo` | 同日 China 與 chips 事件同時出現 |
| `taiwan_chips_combo` | 同日 Taiwan 與 chips 事件同時出現 |
| `tariff_streak_3d` | 前 3 日皆有 tariff active |
| `tariff_rising` | 前 3 日至少 2 日 tariff active，且當日仍有 tariff |
| `china_streak_3d` | 前 3 日皆有 China active |
| `chips_streak_3d` | 前 3 日皆有 chips active |
| `volume_spike` | 當日貼文數 > 前 7 日平均的 2 倍 |
| `volume_drop` | 當日貼文數 < 前 7 日平均的 0.4 倍 |

---

## 4. 不是二元事件，但仍會進入 event branch 的 count features

以下欄位不是 binary event feature，因此不會被 `make_event_combos()` 當成二元事件組合來源；但在 full model 中，它們會被放進 event branch 作為連續或計數特徵：

```text
post_count
tariff_count
deal_count
relief_count
china_count
taiwan_count
chips_count
ai_count
night_post_count
pre_post_count
open_post_count
total_excl
avg_post_len
all_caps_post_count
exclaim_post_count
uppercase_phrase_count
```

---

## 5. 後續進入模型的方式

流程如下：

```text
raw CSV kw_* / Content / Timestamp
  -> post-level binary events
  -> daily binary events
  -> single events + two-way event combinations
  -> brute-force selection
  -> selected events + event x regime interactions
  -> deep learning event branch
```

因此報告中應避免說「原始 CSV 直接包含所有二元事件 feature」。更精確的說法是：

> 原始 CSV 只提供少數 `kw_*` 二元關鍵字欄位；本專案會再由 `Content`、`Timestamp` 與 `kw_*` 衍生 post-level 事件，聚合成 daily-level 二元事件 feature，最後再經 brute-force selection 選出實際進入模型的事件組合。
