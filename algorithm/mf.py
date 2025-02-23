import argparse
import random
import torch
import pandas as pd
import numpy as np
import time

from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam
from tqdm import tqdm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TextMF(nn.Module):
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=768, num_classes=2):
        super(TextMF, self).__init__()
        # Model embedding network
        self.P = nn.Embedding(num_models, model_embedding_dim)

        # Question embedding network
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)
        self.text_proj = nn.Linear(text_dim, model_embedding_dim)

        # Noise/Regularization level
        self.alpha = alpha
        self.classifier = nn.Linear(model_embedding_dim, num_classes)

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            # Adding a small amount of noise in question embedding to reduce overfitting
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)
        return self.classifier(p * q)
    
    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True) # During inference no noise is applied
        return torch.argmax(logits, dim=1)
    
# Helper functions to load and process the data into desired format needed for MF
# For MF we need a "model ID" either in the form of name or index and so we use the tabular data instead of tensors
def load_and_process_data(train_data, test_data, batch_size=64):
    # NOTE: Due to the nature of the embedding layer we need to take max prompt ID from both train and test data
    # But during training we won't be using test question
    num_prompts = int(max(max(train_data["prompt_id"]), max(test_data["prompt_id"]))) + 1
    class CustomDataset(Dataset):
        def __init__(self, data):
            model_ids = torch.tensor(data["model_id"], dtype=torch.int64)
            unique_ids, inverse_indices = torch.unique(model_ids, sorted=True, return_inverse=True)
            id_to_rank = {id.item(): rank for rank, id in enumerate(unique_ids)}
            ranked_model_ids = torch.tensor([id_to_rank[id.item()] for id in model_ids])
            self.models = ranked_model_ids
            self.prompts = torch.tensor(data["prompt_id"], dtype=torch.int64)
            self.labels = torch.tensor(data["label"], dtype=torch.int64)
            self.num_models = len(data["model_id"].unique())
            self.num_prompts = num_prompts
            self.num_classes = len(data["label"].unique())

        def get_num_models(self):
            return self.num_models

        def get_num_prompts(self):
            return self.num_prompts

        def get_num_classes(self):
            return self.num_classes

        def __len__(self):
            return len(self.models)

        def __getitem__(self, index):
            return self.models[index], self.prompts[index], self.labels[index]

        def get_dataloaders(self, batch_size):
            return DataLoader(self, batch_size, shuffle=False)

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    train_loader = train_dataset.get_dataloaders(batch_size)
    test_loader = test_dataset.get_dataloaders(batch_size)

    return train_loader, test_loader
    
def correctness_prediction_evaluator(net, test_loader, device):
    """Standard evaluator for correctness prediction"""
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0
    correct = 0
    num_samples = 0

    with torch.no_grad():
        for models, prompts, labels in test_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)
            logits = net(models, prompts)
            loss = loss_fn(logits, labels)
            pred_labels = net.predict(models, prompts)
            correct += (pred_labels == labels).sum().item()
            total_loss += loss.item()
            num_samples += labels.shape[0]

    mean_loss = total_loss / num_samples
    accuracy = correct / num_samples
    net.train()
    return mean_loss, accuracy

def evaluate(net, test_loader, device, eval_mode="correctness", acc_dict=None, model_num=112):
    """Unified evaluation function that routes to specific evaluator based on mode"""
    if eval_mode == "correctness":
        return correctness_prediction_evaluator(net, test_loader, device)
    elif eval_mode == "router":
        return evaluator_router(net, test_loader, [device], acc_dict, model_num)
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

def evaluator_router(net, test_iter, devices, acc_dict, model_num=112):
    start_time = time.time()
    net.eval()
    successful_num_routes = 0
    num_prompts = 0
    
    model_counts = [0] * model_num
    correctness_result = {}
    with torch.no_grad():
        for prompts, models, labels, categories in test_iter:
            prompts = prompts.to(devices[0])
            models = models.to(devices[0])
            labels = labels.to(devices[0])
            categories = categories.to(devices[0])

            logits = net(models, prompts)
            logit_diff = (logits[:, 1] - logits[:, 0]).unsqueeze(1)
            max_index = torch.argmax(logit_diff)
            model_counts[max_index.item()] += 1
            successful_num_routes += int(labels[max_index] == 1)
            num_prompts += 1
            correctness_result[int(prompts[0])] = int(labels[max_index] == 1)

    # Calculate route accuracy
    route_acc = float(successful_num_routes / num_prompts)
    print(f"Route Accuracy: {route_acc:.4f}")

    # Calculate the highest accuracy baseline
    highest_accuracy = max(acc_dict.values())
    print(f"Highest Model Accuracy: {highest_accuracy:.4f}")

    # Calculate the weighted accuracy based on route_to
    weighted_acc_sum = 0
    for model_id, count in enumerate(model_counts):
        if count > 0:
            weighted_acc_sum += acc_dict[model_id] * count
    
    if sum(model_counts) > 0:
        weighted_accuracy = weighted_acc_sum / sum(model_counts)
    else:
        weighted_accuracy = 0

    print(f"Weighted Baseline Accuracy: {weighted_accuracy:.4f}")

    net.train()
    end_time = time.time()
    print(f"Time used to route {num_prompts} questions: {end_time - start_time}")
    return "N/A", route_acc, correctness_result, model_counts

def create_router_dataloader(original_dataloader):
    """
    Transform a standard dataloader into a router-compatible dataloader.
    Returns:
    - router_dataloader: DataLoader with (prompt, models, labels, categories) batches
    - label_dict: Dictionary mapping {prompt_id: {model_id: label}}
    - acc_dict: Dictionary mapping {model_id: accuracy}
    """
    # Concatenate all batches
    all_models, all_prompts, all_labels = [], [], []
    for models, prompts, labels in original_dataloader:
        all_models.append(models)
        all_prompts.append(prompts)
        all_labels.append(labels)
    
    all_models = torch.cat(all_models)
    all_prompts = torch.cat(all_prompts)
    all_labels = torch.cat(all_labels)

    # Create label dictionary
    label_dict = {}
    for i in range(len(all_prompts)):
        prompt_id = int(all_prompts[i])
        model_id = int(all_models[i])
        label = int(all_labels[i])
        
        if prompt_id not in label_dict:
            label_dict[prompt_id] = {}
        label_dict[prompt_id][model_id] = label

    # Get unique prompts and models
    unique_prompts = sorted(set(all_prompts.tolist()))
    unique_models = sorted(set(all_models.tolist()))
    model_num = len(unique_models)

    # Build router dataloader content
    new_models, new_prompts, new_labels = [], [], []
    for prompt_id in unique_prompts:
        prompt_tensor = torch.tensor([prompt_id] * model_num)
        model_tensor = torch.tensor(unique_models)
        label_tensor = torch.tensor([label_dict[prompt_id].get(model_id, 0) for model_id in unique_models])
        
        new_prompts.append(prompt_tensor)
        new_models.append(model_tensor)
        new_labels.append(label_tensor)

    # Concatenate tensors
    new_prompts = torch.cat(new_prompts)
    new_models = torch.cat(new_models)
    new_labels = torch.cat(new_labels)
    
    # Add dummy categories (all zeros) since they're not used in current implementation
    new_categories = torch.zeros_like(new_labels)

    # Create router dataloader
    router_dataset = TensorDataset(new_prompts, new_models, new_labels, new_categories)
    router_dataloader = DataLoader(router_dataset, batch_size=model_num, shuffle=False)

    # Compute accuracy for each model
    acc_dict = {}
    for model_id in range(model_num):
        correct = sum(label_dict[p].get(model_id, 0) for p in label_dict)
        total = sum(1 for p in label_dict if model_id in label_dict[p])
        acc_dict[model_id] = correct / total if total > 0 else 0.0

    return router_dataloader, label_dict, acc_dict

# Main training loop
def train(net, train_loader, test_loader, num_epochs, lr, device, eval_mode="correctness", 
          acc_dict=None, model_num=112, weight_decay=1e-5, save_path=None):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=num_epochs)

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for models, prompts, labels in train_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = net(models, prompts)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")

        eval_results = evaluate(net, test_loader, device, eval_mode, acc_dict, model_num)
        
        if eval_mode == "correctness":
            test_loss, test_accuracy = eval_results
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
        else:  # router mode
            _, route_acc, correctness_result, model_counts = eval_results
            print(f"Route Accuracy: {route_acc:.4f}")
            progress_bar.set_postfix(train_loss=train_loss, route_acc=route_acc)
            
        progress_bar.update(1)
    
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def load_model(net, path, device):
    net.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", type=int, default=232)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-data-path", type=str, default="../data/train.csv")
    parser.add_argument("--test-data-path", type=str, default="../data/test.csv")
    parser.add_argument("--question-embedding-path", type=str, default="../data/question_embeddings.pth")
    parser.add_argument("--embedding-save-path", type=str, default="../data/model_embeddings.pth")
    parser.add_argument("--model-save-path", type=str, default="../data/saved_model.pth")
    parser.add_argument("--model-load-path", type=str, default=None)
    parser.add_argument("--eval-mode", type=str, default="correctness", 
                       choices=["correctness", "router"],
                       help="Evaluation mode: correctness or router")
    parser.add_argument("--model-num", type=int, default=112,
                       help="Number of models for router evaluation")
    args = parser.parse_args()

    print("Loading dataset...")
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)
    question_embeddings = torch.load(args.question_embedding_path)
    num_prompts = question_embeddings.shape[0]
    num_models = len(test_data["model_id"].unique())
    model_names = list(np.unique(list(test_data["model_name"])))

    train_loader, test_loader = load_and_process_data(train_data, test_data, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing model...")
    model = TextMF(question_embeddings=question_embeddings, 
                   model_embedding_dim=args.embedding_dim, alpha=args.alpha,
                   num_models=num_models, num_prompts=num_prompts)
    model.to(device)

    if args.model_load_path:
        model.load_state_dict(torch.load(args.model_load_path, map_location=device))
        print(f"Model loaded from {args.model_load_path}")

    print("Training model...")
    # Transform test_loader for router mode if needed
    if args.eval_mode == "router":
        test_loader, label_dict, acc_dict = create_router_dataloader(test_loader)
    else:
        acc_dict = None

    train(model, train_loader, test_loader, 
          num_epochs=args.num_epochs, 
          lr=args.learning_rate,
          device=device, 
          eval_mode=args.eval_mode,
          acc_dict=acc_dict,
          model_num=args.model_num,
          save_path=args.model_save_path)
    if args.embedding_save_path:
        torch.save(model.P.weight.detach().to("cpu"), args.embedding_save_path) # Save model embeddings if needed
    if args.model_save_path:
        torch.save(model.state_dict(), args.model_save_path)
        print(f"Model saved to {args.model_save_path}")