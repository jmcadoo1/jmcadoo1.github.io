# jmcadoo1.github.io

  Motivation
The reason that there should be data science on this project is because I have wondered how YouTube recommends content, or how Facebook recommends new friends? This particular project is going to be recommending different movies. All of these recommendations are made possible by the implementation of recommender systems.
Recommender systems encompass a class of techniques and algorithms that can suggest “relevant” items to users. They predict future behavior based on past data through a multitude of techniques including matrix factorization.
In this article, I’ll look at why we need recommender systems and the different types of users online. Then, I’ll show you how to build your own movie recommendation system using an open-source dataset.
Why Do We Need Recommender Systems?
For any given product, there are sometimes thousands of options to choose from. Think of the examples above: streaming videos, social networking, online shopping; the list goes on. Recommender systems help to personalize a platform and help the user find something they like.
The easiest and simplest way to do this is to recommend the most popular items. However, to really enhance the user experience through personalized recommendations, we need dedicated recommender systems.
Now that we understand the importance of recommender systems, let’s have a look at types of recommendation systems.
  Understanding
Types of Recommender Systems
Machine learning algorithms in recommender systems typically fit into two categories: content-based and also collaborative filtering systems, but modern systems combine both approaches.
A) Content-Based Movie Recommendation Systems
Content-based methods are based on the similarity of movie attributes. Using this type of recommender system, if a user watches one movie, similar movies are recommended. For example, if a user watches a comedy movie starring Adam Sandler, the system will recommend them movies in the same genre or starring the same actor, or both. With this in mind, the input for building a content-based recommender system is movie attributes.



Figure 1: Overview of content-based recommendation system (Image created by author)
B) Collaborative Filtering Movie Recommendation Systems
With collaborative filtering, the system is based on past interactions between users and movies. With this in mind, the input for a collaborative filtering system is made up of past data of user interactions with the movies they watch.
For example, if user A watches M1, M2, and M3, and user B watches M1, M3, M4, we recommend M1 and M3 to a similar user C. You can see how this looks in the figure below for clearer reference.

This data is stored in a matrix called the user-movie interactions matrix, where the rows are the users and the columns are the movies.
Now, let’s implement our own movie recommendation system using the concepts discussed above.
Pipeline
Data Collection
The Dataset
For our own system, we’ll use the open-source MovieLens dataset from GroupLens. This dataset contains 100K data points of various movies and users.
We will use three columns from the data:
userId
movieId
rating
You can see a snapshot of the data in figure 3, below:

	Data Management/Representation
Designing our Movie Recommendation System
To obtain recommendations for our users, we will predict their ratings for movies they haven’t watched yet. Movies are then indexed and suggested to users based on these predicted ratings.
To do this, I will use past records of movies and user ratings to predict their future ratings. At this point, it’s worth mentioning that in the real world, we will likely encounter new users or movies without a history.
Implementation
For my recommender system, I will use both of the techniques mentioned above: content-based and collaborative filtering. To find the similarity between movies for my content based method, I’ll use a cosine similarity function. For my collaborative filtering method, I’ll use a matrix factorization technique.
The first step is creating a matrix factorization based model. I’ll use the output of this model and a few handcrafted features to provide inputs to the final model. The basic process will look like this:
Step 1: Build a matrix factorization-based model
Step 2: Create handcrafted features
Step 3: Implement the final model
We’ll look at these steps in greater detail below.
Step 1: Matrix Factorization-based Algorithm
Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. This family of methods became widely known during the Netflix prize challenge due to how effective it was.
Matrix factorization algorithms work by decomposing the user-movie interaction matrix into the product of two lower dimensionality rectangular matrices, say U and M. The decomposition is done in such a way that the product results in almost similar values to the user-movie interaction matrix. Here, U represents the user matrix, M represents the movie matrix, n is the number of users, and m is the number of movies.
Each row of the user matrix represents a user and each column of the movie matrix represents a movie.

 
Once we obtain the U and M matrices, based on the non-empty cells in the user-movie interaction matrix, we perform the product of U and M and predict the values of non-empty cells in the user-movie interaction matrix.
To implement matrix factorization, we use a simple Python library named Surprise, which is for building and testing recommender systems. The data frame is converted into a train set, a format of data set to be accepted by the Surprise library.

Now the model is ready. We’ll store these predictions to pass to the final model as an additional feature. This will help us incorporate collaborative filtering into our system.

Note that we have to perform the above steps for test data also.
Step 2: Creating Handcrafted Features
Let’s convert the data in the data frame format into a user-movie interaction matrix. Matrices used in this type of problem are generally sparse because there’s a high chance users may only rate a few movies.
The advantages of the sparse matrix format of data, also called CSR format, are as follows:
efficient arithmetic operations: CSR + CSR, CSR * CSR, etc.
efficient row slicing
fast matrix-vector products
scipy.sparse.csr_matrix is a utility function that efficiently converts the data frame into a sparse matrix.

‘train_sparse_matrix’ is the sparse matrix representation of the train_data data frame.
We’ll create 3 sets of features using this sparse matrix:
Features which represent global averages
Features which represent the top five similar users
Features which represent the top five similar movies
Let’s take a look at how to prepare each in more detail.
1. Features which represent the global averages
The three global averages we’ll employ are:
The average ratings of all movies given by all users
The average ratings of a particular movie given by all users
The average ratings of all movies given by a particular user

Next, let’s create a function which takes the sparse matrix as input and gives the average ratings of a movie given by all users, and the average rating of all movies given by a single user.

The average rating is given by a user:

Average ratings are given for a movie:

2. Features which represent the top 5 similar users
In this set of features, we will create the top 5 similar users who rated a particular movie. The similarity is calculated using the cosine similarity between the users.

3. Features which represent the top 5 similar movies
In this set of features, we obtain the top 5 similar movies rated by a particular user. This similarity is calculated using the cosine similarity between the movies.

We append all these features for each movie-user pair and create a data frame. Figure 5 is a snapshot of our data frame.

Figure 5: Overview of data with 13 features
 
	Exploratory Data analysis
Here’s a more detailed breakdown of its contents:
GAvg: Average rating of all ratings
Similar users rating of this movie: sur1, sur2, sur3, sur4, sur5 ( top 5 similar users who rated that movie )
Similar movies rated by this user: smr1, smr2, smr3, smr4, smr5 ( top 5 similar movies rated by user)
UAvg: User AVerage rating
MAvg: Average rating of this movie
rating: Rating of this movie by this user.
Once we have these 13 features ready, we’ll add the Matrix Factorization output as the 14th feature. In Figure 6 you can see a snapshot of our data after adding the output from Step 1.

Figure 6: Overview of data with 13 features and matrix factorization output(Image by author)
The last column, named, mf_svd, is the additional column that contains the output of the model performed in Step 1.
Step 3: Creating a final model for our movie recommendation system
To create our final model, let’s use XGBoost, an optimized distributed gradient boosting library.

	Hypothesis Testing
Performance Metrics
There are two main ways to evaluate a recommender system’s performance: Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE). RMSE measures the squared loss, while MAPE measures the absolute loss. Lower values mean lower error rates and thus better performance.
Both are good as they allow for easy interpretation. Let’s take a look at what each of them is:
Root Mean Squared Error (RMSE)
RMSE is the square root of the average of squared errors and is given by the below formula.

Where:
r is the actual rating,
r^ is the predicted ratings and
N is the total number of predictions
Mean Absolute Percentage Error (MAPE)
MAPE measures the error in percentage terms. It is given by the formula below:

Where:
r is the actual rating,
r^ is the predicted ratings and
N is the total number of predictions


	Communication of Insights Attained
Our model resulted in 0.68 RMSE, and 20.55 MAPE on the unseen test data, which is a good and usable model. An RMSE value of less than 2 is considered good, and a MAPE less than 25 is excellent.
Communication of Approach
 That said, this model can be further enhanced by adding features that would be recommended based on the top picks dependent on location or genre. We could also test the efficacy of our various models in real-time through A/B testing.
Summary
In this article, we learned the importance of recommender systems, the types of recommender systems being implemented, and how to use matrix factorization to enhance a system. We then built a movie recommendation system that considers user-user similarity, movie-movie similarity, global averages, and matrix factorization. These concepts can be applied to any other user-item interactions systems.

