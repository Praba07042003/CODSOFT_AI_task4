import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample dataset of user ratings for products
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4],
    'product_id': ['P1', 'P2', 'P3', 'P2', 'P3', 'P4', 'P1', 'P3', 'P1', 'P2', 'P3', 'P4'],
    'rating': [5, 3, 4, 4, 5, 2, 2, 3, 4, 5, 2, 1]
}

# Create a DataFrame
ratings_df = pd.DataFrame(data)

# Create a user-item matrix
user_product_matrix = ratings_df.pivot_table(index='user_id', columns='product_id', values='rating')

# Fill NaN values with 0 for similarity calculation
user_product_matrix_filled = user_product_matrix.fillna(0)

# Compute the cosine similarity matrix
user_similarity = cosine_similarity(user_product_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_product_matrix.index, columns=user_product_matrix.index)

def get_product_recommendations(user_ids, user_product_matrix, user_similarity_df, n_recommendations=3):
    recommendations = {}
    for user_id in user_ids:
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]

        # Weighted ratings by similarity scores
        weighted_ratings = {}
        for similar_user in similar_users:
            similarity_score = user_similarity_df.loc[user_id, similar_user]
            for product in user_product_matrix.columns:
                if not np.isnan(user_product_matrix.loc[similar_user, product]):
                    if product not in weighted_ratings:
                        weighted_ratings[product] = 0
                    weighted_ratings[product] += user_product_matrix.loc[similar_user, product] * similarity_score

        # Sort products by weighted ratings and exclude already rated products
        sorted_ratings = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)
        recommended_products = [product for product, score in sorted_ratings]

        # Get the top n product recommendations
        recommendations[user_id] = recommended_products[:n_recommendations]

    return recommendations

# Generate product recommendations for multiple user_ids
user_ids = [1, 2, 3, 4]
recommendations = get_product_recommendations(user_ids, user_product_matrix, user_similarity_df, n_recommendations=2)

# Print recommendations for each user
for user_id, products in recommendations.items():
    product_list = ', '.join([product for product in products])
    print(f"Recommended products for user {user_id}: {product_list}")