# RECOMMENDER SYSTEM
1. Team members

- Eyad Ahmed (273821)
- Carlo Fiammenghi (273691)
- Marta Shkreli  (268341)

2) Introduction
Our project consists of boosting the revenues of a prestigious fashion firm through the use of data science. As members of the company's data science team, our primary focus is on improving the recommender system for the firm's online platform. To achieve this goal, we will be testing and evaluating various recommendation systems to determine the most effective one.

3) Methods
We were provided with three datasets that contain information about customers, transactions, and articles. We performed a detailed explanatory data analysis on all of the datasets.

a) Data understanding 
The 'recsys_customers.csv' dataset includes customer IDs and ages, as well as optional information about participation in fashion news and membership in a club. The 'recsys_articles.csv' dataset contains article IDs, product names, color information ranging from very detailed to very general, departments, membership groups, sections, and types of garments. The 'recsys_transactions.csv' dataset includes information about purchases made by customers on specific dates.

b) Data Preparation
we cleaned and organized the datasets to make them more understandable. This process helped us to gain a better understanding of the relationships between customers and transactions. We also identified and addressed any missing values and highlighted important features within each dataset.

Deal with Null values in 'recsys_customer';
Create additional dataframes;
Deal with Null Values in 'recsys_article' by removing them;
c) EDA
Explanatory Data Analysis is a crucial step in the machine learning process for preparing data. It involves using various Python packages to understand the inherent characteristics of our datasets:

Pandas;
Numpy;
Sklearn;
Matplotlib.pyplot;
Seaborn.

Customers
We have data for over 40,000 customers, including their IDs, subscription status for fashion news, club membership status, and ages. Our explanatory data analysis revealed a small percentage (0.32%) of null values in the age column. We decided to replace these null values with the mode of the ages of all the customers.

We divided the customers into age categories, counted the number of people in each category, and calculated the proportion of people who are subscribed to the club membership or fashion news.

![club_membership](images/club_member.png)

![fashion_news](images/fashion_news.png)


Transactions
The transactions dataset includes information about all purchases made by each customer and the dates on which they were made. We calculated the number of transactions per customer and found that the maximum number of transactions made by a single customer was 104. To gain a better understanding of the dataset, we grouped the transactions into classes and observed that only a small proportion of customers made more than 30 transactions. We also calculated the number of customers who purchased each product.

![transactions_group](images/purchase.png)


Articles
Our dataset includes 6536 articles, each with various characteristics. Each product belongs to multiple categories, including a garment group, section, department, index group, and type.

During our explanatory data analysis, we searched for the best identifier for all the products. To avoid overfitting, we chose to discard more precise classes and also to exclude too general ones, as they might result in an inaccurate recommendation system. We ultimately decided to consider only the articles' type and section as identifiers. Another attribute of the articles is their color, and we selected the 'perceived_colour_master' attribute as the best balance between precision and avoiding overfitting.

We also identified 'Unknown' items in the data and removed them because they made up a small proportion of the data.

![articles](images/section_of_product_n_of_transaction.png)


4) Recommender System
Our recommendation system generates a list of articles that a customer may be interested in based on their current purchases and similarities with other users.

a) Content filtering
The first type of recommendation system we implemented was a content filtering system, which made suggestions based on features of the products such as color, group, and type. We could also have used all of the other product attributes for this purpose.

One Hot Encoding
We created a one-hot encoding for the color value of the articles and did the same for the section and type of the product. This resulted in three tables with the ID of the articles as rows and all the features as columns. Each cell in the table has a value of 1 if the corresponding article has the specific feature. We then concatenated all of the tables.

Cosine Similarity
To measure the similarity between products, we used cosine similarity, which is based on the cosine angle between two vectors. The smaller the angle, the greater the similarity between the products.

Final step
We are now able to return the top 10 recommended items for a given input product, identified by its ID. It is possible that the input product could be included in the list of recommended items, which is not necessarily a problem as a user may choose to repurchase the same product. However, we included a function to handle this scenario by removing the input product from the list and adding the 11th item in its place.

![finalstep](images/finalstep.png)


b) Collaborative filtering: CRS matrix
The second type of recommendation system we implemented was a user-based filtering system, which is based on the idea of using customer opinions about various products to make recommendations. This system suggests articles to a customer based on their own past purchases, as well as the purchases and opinions of similar customers. Essentially, this system uses information collected from different customers to recommend products to the current customer.

CSR Matrix
We started by creating a matrix with 'customer_id' and 'article_id' in order to map all of the transactions in the dataset. However, this matrix had many unobserved elements because each user only purchased a small number of products relative to the total number of products. We calculated the sparsity of the matrix and found that its value was too low to make reliable predictions.

Increase sparsity
To address this issue, we decided to only consider a portion of the dataset. We removed columns corresponding to customers who made fewer than a certain number of purchases and also eliminated some rows corresponding to products that had been purchased less frequently.

KNN
We imported the KNN algorithm from the 'sklearn' library. This algorithm clusters similar customers based on common transactions and makes predictions using the average rating of the top-k nearest neighbors. To evaluate similarity, we used cosine similarity, which measures the distance between instances to determine their "closeness".

![KNN](images/KNN.png)

c) Collaborative filtering: neural network
The last method we used for the recommender system is the artificial neural network. In this case, we generate recommendations based on the similarity between users’ transactions, rather than the similarity of customers and articles (done through the utility matrix).

Matrix factorization
is a method for uncovering the features or information underlying the interactions between customers and articles. To find similarities and make predictions, we examine the relationship between the customer and article matrices.
This method works by decomposing the user-item interaction matrix into the product of two lower-dimensional rectangular matrices, which helps to address the issue of sparsity caused by most customers only purchasing a small number of items.

Loss function
We used a loss function to evaluate the performance of our model and attempted to minimize the error. To optimize the performance of our function, we used the mean squared error (MSE) and two regularization coefficients, L2 and the gravity term.

![loss_function](images/loss_function.png)


To evaluate the performance of our recommendations, we split our dataset into a train and test set and plotted the loss function. While the loss function was not perfect, the small value of the number of transactions on the y-axis made it difficult to predict. We experimented with various hyperparameters to try to improve the accuracy.

After training the model, we used either the dot product or cosine similarity to rate it and retrieved the top 5 recommended results.

Our recommendation engine was designed to increase company revenue by predicting user preferences and recommending the right products to each user. We employed various types of recommendations using both content-based and collaborative filtering systems.

Regarding the first model, a content-based one, the model was quite accurate due to the extensive details available about all the products. For example, we could use the department instead of the group for increased precision, or the 'perceived_colour_value' instead of the 'perceived_colour_master' as hyperparameters to improve the recommendations.

For the second model, a collaborative filtering one, we used a CRS matrix to represent sparse matrices. However, we believe this model was not suitable for our dataset due to the high number of zero values. To achieve a decent level of sparsity, we would have had to significantly reduce the number of articles and customers, which would have resulted in a loss of information and an inadequate recommendation system. The initial dataset had a sparsity of 0.11%, which we were able to increase to about 0.6%.

The last method we tried to implement was an artificial neural network, which was challenging to set up as we had to experiment with various parameters to find the values that best fit our dataset.

One major issue we faced with all our models was the lack of ratings for products, requiring us to use the number of transactions to make recommendations between users. These values were relatively small due to the majority of customers only making a small number of purchases. Despite these challenges, we were able to develop three different recommendation systems that our company can use to increase revenue.
