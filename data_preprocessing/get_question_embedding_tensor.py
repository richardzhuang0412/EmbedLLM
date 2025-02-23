from sentence_transformers import SentenceTransformer

import pandas as pd
import torch
import os

def get_question_embeddings(csv_path, model_name= 'all-mpnet-base-v2'):
    """
    Generate embeddings for questions in the order specified in the CSV file.
    
    Args:
        csv_path (str): Path to the question_order.csv file
        model_name (str): Name of the sentence-transformer model to use
        
    Returns:
        torch.Tensor: Tensor of shape (num_questions, embedding_dim) containing question embeddings
    """
    df = pd.read_csv(csv_path)
    model = SentenceTransformer(model_name)
    questions = df['prompt'].tolist()
    embeddings = model.encode(questions, show_progress_bar=True)
    embedding_tensor = torch.tensor(embeddings)
    
    return embedding_tensor

if __name__ == "__main__":
    # Set paths
    csv_path = "../data/question_order.csv"
    
    # Generate embeddings
    question_embeddings = get_question_embeddings(csv_path)
    
    # Save the tensor
    output_path = "../data/question_embeddings.pth"
    torch.save(question_embeddings, output_path)
    
    print(f"Generated embeddings tensor of shape: {question_embeddings.shape}")
    print(f"Saved embeddings to: {output_path}")
