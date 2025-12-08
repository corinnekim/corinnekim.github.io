---
layout: single
title: "Predicting Love: Feature Engineering with Speed Dating Data"  # 글 제목
date: 2025-12-08                 # 작성 날짜
category: [datascience]        # 카테고리 지정
# tags: [datascience, machinelearning, python, eda, feature-engineering] # 검색용 태그
author_profile: true             # (선택) 프로필 표시 여부
sidebar:                         # (선택) 목차 생성
  nav: "main"
  permalink: /datascience/feature-engineering/
math: true                       # (중요) 수학 공식을 쓴다면 true로 설정
---

## What Makes Data Scientists Different?    
![](/assets/images/coding/coding3.jpeg)

<br>
**Feature engineering** is where human knowledge and intuition matter the most.     
While we all use similar tools like **Pandas** or **Scikit-Learn**, the results of feature extraction vary significantly among data scientists.      
     

##  What is Feature Engineering?
Feature engineering is the process of transforming raw data into **meaningful** features that better represent the underlying problem to predictive models.     
Raw data is often messy, and it is hard to see meaningful patterns at first glance.    
      
    
The main goals are:
- To improve model performance (accuracy).
- To reduce computational costs by lessening the number of features (Dimensionality Reduction).
- To extract hidden insights that correlate with the target variable.    
      

## How To Do It?
There are various techniques to engineer features:     
* **Transformation:** Scaling (Log, MinMax) or Binning.   
* **Aggregation:** Using statistical measures (Count, Mean, Sum, Z-scores).    
* **Interaction:** Creating a new feature by combining two or more existing features.        
      

## Feature Engineering Example - Speed Dating Data
   
     
### 1. Dataset Overview
<br>
![](/assets/images/zootopia.jpeg){: width="60%"}
<br> 

We will use the **Speed Dating Dataset**.    
<br>
![](/assets/images/speed_dating.png)  
* **Data Structure:** Each row represents a single **dating session**. Since participants engage in multiple dates, one person appears in multiple rows.
* **Goal:** Predict whether the couple will have a **Match (1)** or not **(0)**.
* **Naming Convention:**
    * `i_` (Prefix): Refers to the **Subject** (Participant).
    * `o_` (Prefix): Refers to the **Partner**.
    * `[trait]`: Refers to specific attributes like `attractive`, `sincere`, etc.
       
**Variable Conventions**

| Variable Pattern | Description | Example |
|:---|:---|:---|
| `age`, `race` | Demographics of the **Subject** (Participant) | 28, 'Asian' |
| `age_o`, `race_o` | Demographics of the **Partner** | 27, 'Hispanic' |
| `o_importance_[trait]` | **Partner's Preference** <br>(How important `[trait]` is to the partner) | `o_importance_attractive` = 35 <br>(Partner values attractiveness at 35%) |
| `o_score_[trait]` | **Partner's Rating** <br>(Score the partner gave to the Subject) | `o_score_funny` = 9.0 <br>(Partner rated Subject's humor as 9/10) |
| `i_importance_[trait]` | **Subject's Preference** <br>(How important `[trait]` is to the subject) | `i_importance_sincere` = 25 <br>(I value sincerity at 25%) |
| `i_score_[trait]` | **Subject's Rating** <br>(Score the subject gave to the Partner) | `i_score_ambition` = 7.0 <br>(I rated partner's ambition as 7/10) |
| `match` | **Match Result** (1=Yes, 0=No) | 1 |

<br>
**Example Traits** (Importance vs Score)    

| Trait | Description | `i_importance` | `i_score` |
|:---|:---|:---:|:---:|
| `attractive` | Physical Appearance | 30 | 9 |
| `sincere` | Sincerity & Honesty | 25 | 7 |
| `funny` | Humor | 25 | 8 |
| `intelligence` | Smartness | 10 | 5 |
| `ambition` | Drive & Goals | 15 | 6 |
| `shared_interests` | Hobbies in common | 5 | 5 |
| **Summary** | | **Sum = 100** | **Avg = 6.7** |

- `Importance` scores must sum up to 100.
- `Score` is between 0 and 10.
- This person values `attractivenes`₩ the most and  `shared interests` the least.
- This person rated the partner's `intelligence` as 5/10.

## 1. Age Features
To evaluate the influence of `age` on the match, we will extract a new feature: `age_gap`. <br>
```python
import pandas as pd
dating_df = pd.read_csv('../data/dating.csv')

dating_df['age_gap'] = (dating_df['age'] - dating_df['age_o']).abs()
```
<br>
**Result**     ![](/assets/images/age_gap.png){: width="11%"}
<br>
 Now, we can drop the original columns `age` and `age_o`   
```python
dating_df.drop(['age', 'age_o'], axis = 1, inplace = True)
```
<br>
With this simple calculation, We extracted a potentially meaningful feature `age_gap`, <br>

## 2. Race Features
![](/assets/images/importance_same_race.png)      
Each participant rated the importance of having a partner of the same race (0-10).    
However, the actual impact depends on **how much the individual values it.**      

Therefore, we will create a new feature that combines both features:
> **New Feature = (Is Same Race?) * (Importance Rating)**  
  
```python
# Create a binary feature: 1 if same race, 0 if not
# Convert boolean (True/False) to integer (1/0)
dating_df['same_race'] = (dating_df['race'] == dating_df['race_o']).astype(int)
```
<br>

To capture the negative impact of a mismatch, let's map 'different race' to -1 instead of 0.

```python
dating_df['same_race'].replace({0: -1}, inplace = True)
```
<br>
**Result**     ![](/assets/images/same_race.png){: width="13%"}
<br>
Now, let's calculate the final weighted score by multiplying the same race status (-1/1) by the importance rating.

```python
dating_df['same_race_weighted'] = (dating_df['same_race'] * dating_df['importance_same_race'])
```
<br>
**Result** ![](/assets/images/same_race_weighted.png){: width="20%"}   
<br>
Now, the math works perfectly!     
`1 (Same)` boosts the score, while `-1 (Different)` penalizes it based on the importance rating.    

Now, let's drop the columns that we don't need anymore.

```python
dating_df.drop(['race', 'race_o', 'importance_same_race', 'same_race'], axis = 1, inplace = True)
```



## 3. Handling 24 Rating Features
So far, we've got two predictive features: `age_gap` and `same_race_weighted`. Yay!            
However, the **real problem** is...
<br>

![](/assets/images/columns24.png)

<br>
 
We're staring at **24** `importance` and `score` columns. Using them raw would just overwhelm the model.        
How do we boil these down to get only the key information?     
Let's tackle this in the next post. Until next time!  