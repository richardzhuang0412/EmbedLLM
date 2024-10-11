from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
from tqdm import tqdm
import argparse

def load_tensor_data(train_x_path, train_y_path, test_x_path, test_y_path):
    train_x = torch.load(train_x_path)
    train_y = torch.load(train_y_path)
    test_x = torch.load(test_x_path)
    test_y = torch.load(test_y_path)

    print(f"Data shapes - train_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")
    return train_x, train_y, test_x, test_y

def evaluate(train_x, train_y, test_x, test_y, num_neighbors):
    accs = []
    num_model, q_embed_dim, num_train_questions = train_x.shape
    for i in tqdm(range(num_model)):
        X_train, y_train = train_x[i, :, :].tolist(), train_y[i, :].tolist()
        X_test, y_test = test_x[i, :, :].tolist(), test_y[i, :].tolist()
        
        neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
        neigh.fit(X_train, y_train)
        
        y_pred = neigh.predict(X_test).tolist()
        bool_ls = list(map(lambda x, y: int(x == y), y_pred, y_test))
        accs.append(sum(bool_ls) / len(y_test))

    mean_accuracy = sum(accs) / len(accs)
    print(f"Mean Test Accuracy for {num_neighbors} neighbors: {mean_accuracy}")
    return {"mean_accuracy": mean_accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_x_path", type=str, default='data/train_x.pth')
    parser.add_argument("--train_y_path", type=str, default='data/train_y.pth')
    parser.add_argument("--test_x_path", type=str, default='data/test_x.pth')
    parser.add_argument("--test_y_path", type=str, default='data/test_y.pth')
    parser.add_argument("--num_neighbors", type=int, default=5)
    args = parser.parse_args()

    print("Start Initializing Dataset...")
    train_x, train_y, test_x, test_y = load_tensor_data(args.train_x_path, args.train_y_path, args.test_x_path, args.test_y_path)
    print("Finish Initializing Dataset")
    
    evaluate(train_x, train_y, test_x, test_y, args.num_neighbors)