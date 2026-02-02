# F039 python五种算法美食推荐可视化大数据系统vue+flask前后端分离架构


> 完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从github来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775
> 

关注B站，有好处！
编号:  F039
## 视频
《待发布》
## 1 系统简介
系统简介：本系统是一个基于Vue+Flask构建的美食推荐可视化大数据系统，旨在为用户提供个性化的美食店铺推荐和丰富的数据分析功能。其核心功能围绕美食数据的展示、推荐和用户管理展开。主要包括：首页，用于展示热门美食推荐、搜索功能以及推荐算法的可视化概览；数据卡片，提供单个美食店铺的详细信息（如评分、价格、地理位置等），并支持用户点赞"收藏"功能，同时集成百度地图以展示店铺位置；可视化分析模块，通过多种图表形式（如城市分析、价格分布、词云分析、主页美食类型统计）为用户提供直观的数据洞察；推荐模块，基于五种推荐算法（UserCF、ItemCF、混合推荐、神经网络推荐、神经网络+混合推荐）为用户提供个性化的美食店铺推荐；美食库搜索功能，允许用户通过关键词搜索特定美食店铺；以及管理端功能，支持用户管理（注册、登录、权限分配）、个人信息管理（修改头像、密码等）以及美食店铺的增删改查操作，确保系统的安全性和管理效率。
## 2 功能设计
该系统采用典型的B/S（浏览器/服务器）架构模式，前后端分离，基于Vue+Flask+MySQL的技术栈构建。用户通过浏览器访问Vue前端界面，前端由HTML、CSS、JavaScript以及Vue.js生态系统中的Vuex（用于状态管理）、Vue Router（用于路由导航）和ECharts（用于数据可视化）等组件构建。前端通过RESTful API与Flask后端交互，Flask后端负责业务逻辑处理，并利用SQLAlchemy等ORM工具与MySQL数据库进行数据存储与管理。系统还集成了百度地图API，用于展示美食店铺的地理位置分布及价格、评分等信息的热力图。此外，系统还包含一个独立的数据抓取模块，负责从外部数据源抓取美食数据并导入MySQL数据库，为推荐算法和可视化分析提供数据支撑。推荐算法模块基于五种不同算法实现个性化推荐，包括基于用户的协同过滤（UserCF）、基于物品的协同过滤（ItemCF）、混合推荐算法、神经网络推荐以及神经网络与混合推荐的结合方案，确保推荐的多样性和准确性。管理端功能则提供了完善的后台管理界面，支持管理员对用户、美食店铺以及系统数据进行增删改查操作，保障系统的可维护性和可扩展性。
### 2.1系统架构图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3acb5412935e444d8529c7b442150832.jpeg)
### 2.2 功能模块图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b863a1d7e23d4d4cb61bd76477ccacaf.jpeg)
## 3 功能展示
### 3.1 登录 & 注册 【网站功能】
登录注册做的是一个可以切换的登录注册界面，点击去登录后者去注册可以切换，背景是一个视频，循环播放。
登录需要验证用户名和密码是否正确，如果**不正确会有错误提示**。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c48809f409274642be677735d88d3060.png)
注册需要**验证用户名是否存在**，如果错误会有提示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b695f1d1197641aebdc27b41aafbc3da.png)
### 3.2 主页 【网站功能】
主页的布局采用了上方是菜单，下方是操作面板的布局方法，右侧的上方还有用户的头像和退出按钮，主页显示了美食的分类统计：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8e1c9980866c4d8e827dd6aa16e9152f.png)
### 3.3 美食推荐 / 五种推荐算法 【网站功能】
1. UsercF
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd775c6e3fab467c885aa93e8083204e.png)
2. ItemCF
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4d678751317e45c1bd32bba14dd6971d.png)
3. 混合CF
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28f3fed207f94d9e960b9d094e38316b.png)
4. 神经网络
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/078f54976a6f4f70aa531fcdcc0b1ee9.png)
5. SVD
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/30ca370cef8244b091b3d9c1f5ed0d8d.png)
### 3.4 可视化数据分析 【网站功能】
1. 城市分析：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/03267477a96b4d8ea3158efa724ad703.png)
2. 价格分析：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3152832de155445fa48385e893fad028.png)
3. 词云分析：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1a244171584947c788ed9113f65cbe74.png)
### 3.5  基于地图的可视化 【网站功能】
1. 百度地图价格地图分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fe2c421c3c9a46d0946212db354f70c9.png)
2. 百度地图评分地图分析：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/275320f31e654139a3580265920f5ece.png)
### 3.6 美食店铺搜索 【网站功能】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5d44c3a2c564e199df49be3297cab47.png)
### 3.7 登录 【管理员】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aaecc95371d04f3b8dd355f1e34f2648.png)
### 3.8 后端主页 【管理员】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/037b35dec7814f9bb54b795d67b85b6c.png)
### 3.9 个人信息管理 【管理员】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f2e51d2b9693431ab89ee936acf8cd8e.png)
### 3.10 用户管理 【管理员】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0add59566f3847429358d0cb9dabffea.png)
### 3.11 美食店铺管理 【管理员】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/09990a91305648729fd524ae23ac420d.png)
## 4程序代码
### 4.1 代码说明
代码介绍：以下是一个基于MLP和感知机的美团美食推荐算法的代码实现。
### 4.2 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/13131ca3e87b4d8cb44a974ab7596fcf.png)
### 4.3 代码实例
```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class FoodRecommender:
    def __init__(self, user_features, item_features):
        self.user_features = user_features
        self.item_features = item_features
        self.perceptron = Perceptron(input_size=user_features.shape[1], output_size=64)
        self.mlp = MLP(input_size=64, hidden_size=32, output_size=1)
        self.scaler = StandardScaler()
        self.user_item_scores = defaultdict(dict)
        
    def fit(self, user_item_ratings, epochs=100, lr=0.01):
        optimizer = torch.optim.Adam(list(self.perceptron.parameters()) + list(self.mlp.parameters()), lr=lr)
        loss_fn = nn.MSELoss()
        
        for _ in range(epochs):
            total_loss = 0
            for user, items in user_item_ratings.items():
                user_feat = self.scaler.fit_transform([self.user_features[user]])
                for item, rating in items:
                    item_feat = self.item_features[item]
                    combined = torch.tensor([np.concatenate((user_feat[0], item_feat))], dtype=torch.float32)
                    score = self.perceptron(combined)
                    prediction = self.mlp(score)
                    loss = loss_fn(prediction, torch.tensor([rating], dtype=torch.float32))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            print(f"Epoch {_+1}, Loss: {total_loss / len(user_item_ratings)}")
            
    def recommend(self, user_id, top_k=10):
        user_feat = self.scaler.transform([self.user_features[user_id]])
        scores = []
        for item_id in self.item_features:
            if item_id not in self.user_item_scores[user_id]:
                combined = torch.tensor([np.concatenate((user_feat[0], self.item_features[item_id]))], dtype=torch.float32)
                score = self.perceptron(combined)
                prediction = self.mlp(score)
                self.user_item_scores[user_id][item_id] = prediction.item()
            scores.append((item_id, self.user_item_scores[user_id][item_id]))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# 使用示例
if __name__ == "__main__":
    # 假设user_features和item_features已经提取好
    user_features = np.random.rand(100, 10)  # 100 users, 10 features
    item_features = np.random.rand(500, 10)  # 500 items, 10 features
    user_item_ratings = defaultdict(dict)
    for user in range(100):
        for item in np.random.choice(500, 5):
            user_item_ratings[user][item] = np.random.randint(1,6)
    
    recommender = FoodRecommender(user_features, item_features)
    recommender.fit(user_item_ratings)
    recommended = recommender.recommend(0, top_k=10)
    print("Recommended items:", recommended)


```
