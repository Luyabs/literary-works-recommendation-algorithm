## 基于混合推荐算法的文学作品推荐系统
1. 前端 https://github.com/Luyabs/literary-works-recommendation-frontend
2. 业务后端 https://github.com/Luyabs/literary-works-recommendation-backend
3. `你在这` 算法端 https://github.com/Luyabs/literary-works-recommendation-algorithm

## 项目环境
> Python: 3.9  (至少是Python 3)    
> MySQL：8.0  
> 依赖：numpy, torch, pandas, pymysql, yaml, tqdm，如果没有安装直接装最新版

## 项目运行
1. **启动前配置**: 在本项目中你需要在项目的根目录中创建一个数据库配置文件，文件名：config.yaml
```yaml
# 这是文件内容
mysql:
  host: 'localhost'  # 你的MySQL地址 [需更改 本地运行的数据库不用改]
  port: 3306   # 你的MySQL端口 [需更改 但大概率是3306]
  user: 'root'  # 你的MySQL用户名 [需更改 但大概率是root]
  password: 'PASSSSSSSSWORD'  # 你的MySQL密码 [需更改]
  database: 'literary_works_recommendation_platform'
```
2. 数据库导入：通过SQL文件在数据库中创建数据表 literary_works_recommendation_platform.sql
3. 运行**算法后端**：运行文件 controller/recommed_controller.py
4. 将数据导入数据库：运行文件 raw_data/save_into_db.py
5. 运行爬虫扩展数据：运行文件 raw_data/crawler.py  

## 项目结构
1. controller：由Flask实现的简易后端  
2. mapper：封装与数据库进行交互的函数工具类  
3. model_and_train：含模型(deep_fm, fm, lfm)文件(xxx_model.py)，训练、测试(数据集与top-k测试)、预测、召回文件(xxx_trainer_and_predictor.py)与对数据库索引映射到从0开始的自然数文件(id_mapping.py)
4. model_params: 模型参数保存位置，由于模型参数过大，不便上传，可以自行训练或联系我获取
5. mysql_connector: 对pymysql进行封装，通过config.yaml进行配置
6. raw_data: 存放预处理后的数据集(原数据集：豆瓣图书数据集 https://aistudio.baidu.com/datasetdetail/7492 )，将数据集导入导数据库的文件(save_into_db.py)与通过爬虫扩充数据的文件(crawler.py)
7. 

## 项目整体架构
![image](https://github.com/Luyabs/literary-works-recommendation-backend/assets/74538732/a3344555-7c3e-498e-a794-06a25da84354)

## 算法端架构
![image](https://github.com/Luyabs/literary-works-recommendation-algorithm/assets/74538732/262ddcde-ca79-4042-9d7d-de8bdd7fc8f1)

## 算法模型
![image](https://github.com/Luyabs/literary-works-recommendation-algorithm/assets/74538732/fc0ead24-dec2-4b6b-84a9-d162a572da39)
![image](https://github.com/Luyabs/literary-works-recommendation-algorithm/assets/74538732/eeaa8d8f-0ba1-4514-b1d0-9030d9404777)
![image](https://github.com/Luyabs/literary-works-recommendation-algorithm/assets/74538732/67a85b5f-bb25-44f4-b43e-697ad59ff24d)

## 混合推荐算法流程
1.	设置用户Id：user_id，推荐数量k，评分最低阈值threshold，混合权重mix_weight
2.	使用LFM模型进行召回（预测该用户与所有文学作品的评分），该次召回会得到长度为min(2000, k * 75)的文学作品列表（列表第一列为作品Id，第二列为作品在LFM模型下的评分），且列表中作品评分均≥threshold
3.	使用FM（或Deep FM）模型对步骤2得到的列表再次进行预测，得到长度为min(2000, k * 75)的文学作品列表 (列表第一列为作品Id，第二列为作品在FM（或Deep FM）模型下的评分(该模型输出的点击率会乘5以放大至[0,5]区间作为评分))
4.	合并2与3中列表的第二列，其中将最终评分设置为：LFM评分 * mix_weight + FM（或Deep FM）评分 * (1- mix_weight)
5.	依据最终评分对list排序，并筛选出列表的前k * 5个元素
6.	依概率（作品评分正比于被选择的概率）从这些元素中选取k个返回
7.	(于业务后端完成) 如果这k个推荐结果中存在目标用户已经评价过的文学作品，则将从最热门的k * 100个作品中随机等量选择以替换这些作品

## 基于Embedding的近似向量推荐流程
1.	设置文学作品Id：work_id，推荐数量k
2.	从LFM模型取出作品嵌入查找表E（即从模型取出Embedding层的参数）
3.	根据work_id，从查找表中取出对应的作品隐向量v
4.	计算该作品隐向量v与查找表E中每一行的余弦相似度
5.	依据该相似度进行排序，并筛选出列表的前k * 5个元素
6.	依概率从这些元素中选取k个返回
7.	(于业务后端完成) 如果这k个推荐结果中存在目标用户已经评价过的文学作品，则将从最热门的k * 100个作品中随机等量选择以替换这些作品

## 索引映射
```text
在3.4.1~3.4.3的模型构建中，均用到了Embedding层。Embedding层接收一个整型输入（范围为0~n-1，n为隐向量的个数），并返回对应位置的隐向量，该层非常类似一个隐向量数组。
而由于本项目中的系统中无论是作品Id、用户Id还是作品标签Id虽然是整数，但在数据库中的记录都并非从0开始，也不连续。因此需要做索引映射，以将离散的Id一一映射成连续的自然数序列。
这个索引映射通过创建三个字典（哈希表）实现。字典的键为离散Id，值为可以作为Embedding输入的连续Id，这种键值对设计的好处在于：由于值连续，因此值可以作为字典键列表的下标，根据值快速反查键，实现双向的查找表。
在推荐系统启动时，会初始化一个Id Mapping类的对象，并从数据库中读取“离散Id与连续Id”的映射关系，并加载到三个字典中。之后，每次将输入送入模型时，都会根据字典将数据库中直接读取的Id映射为可以被Embedding接收的Id，再通过Embedding层转换为隐向量以进行后续操作。
需要注意的是，在本系统中离散Id（也是表的主键）与连续Id的映射关系并不需要保存到本地，而是从数据库中查找{Id(主键), 该此查找中该记录所在行号 - 1}来确定映射关系。如通过以下SQL语句：“select user_id, ROW_NUMBER() over (ORDER BY user_id) - 1 as seq_id from user;”。这样处理的好处是不用额外存储该映射关系，但需要正确维护数据表结构，不能随便删除数据，只能采用逻辑删除方式，等到要重新训练模型前才允许物理删除数据，否则会引起Embedding的查找错误，如越界错误或索引不正确。
```

## 增量训练
```text
增量训练是推荐系统重要的问题之一，对于一个推荐系统，必然做好要接收新数据，时刻更新模型的准备。本系统没有非常高的实时性需求，因此选择了定时更新方案。在整个系统（前端、算法后端、业务后端）启动后，会通过业务后端的定时任务，请求算法后端每30分钟更新前30分钟的新数据（用户对物品的新评价）；每天00：00更新前一日的新数据。同时通过在数据中混入一定比例的旧数据（设置为新：旧=1：3）来更新模型，以起到让模型同时学习新知识和巩固旧知识的效果。
此外，在本系统的算法后端，训练与预测在同一个项目内，因此训练后的模型可以通过赋值的方式替换老模型。这样可以在训练模型的过程中，同时让老模型保持工作，维持系统的稳定性。
考虑到并发安全性与算力条件，系统一次仅允许进行一种训练，过程中采用互斥锁来限制。在业务后端请求算法后端进行训练前会检查该锁有没有被占用，如果被占用则直接放弃训练，否则获取锁开始训练。同时由于算法后端使用了多种推荐算法，因此每次更新需要额外设置锁，来确保所有模型的更新是原子性操作。
值得一提的是，系统可能随时都有新用户、新文学作品、文学作品的新标签加入。因此在模型增量训练前，这些新东西必然不在Embedding查找表的索引之内，无法被正确的映射成隐向量。本系统采用的解决方案是：所有关于这些新东西的推荐都将直接返回热门作品，不对新数据执行任何推荐算法或在推荐算法中使用这些新数据；而在增量训练时，会创建新的更大规模的Embedding层，并将之前已训练的参数赋值回新的更大规模的模型的前几层（如果维护数据表中的Id保持递增特性，则数据表新增的Id一定对应Embedding的最后几个索引，且之前的索引相对位置保持不变），这样能起到迭代训练的效果。
需注意3.4.6中的索引映射由于增量训练的存在，需要为每一个模型保持一份独立的索引映射对象，并在增量训练时分别独立更新，以免进行Embedding时产生越界错误。
```

## 爬虫流程
```text
1. 将<作品Id，作品名，作品标签>导入work表中，作品标签以字符串形式存储
2. 将<用户名>导入user表中，通过递增主键生成用户Id，约40万用户
3. 将评分记录<用户名，作品Id，评分>导入review_user_work表
4. 根据评分记录，修改work表的sum_rating与sum_rating_user_number字段
5. 删除出现在<用户名，作品Id，评分>三元组文件中，但未出现在<作品Id，作品名，作品标签>文件中的所有评分记录（约删除1万个未知作品名，但存在评分的评分记录）
6. 将<作品标签>同时导入tag与record_tag_work表中
7. 以上操作在导入（INSERT）过程中，同时手动插入当前时间到update_time与create_time两个字段，格式yyyy-MM-dd hh:MM:ss
```
