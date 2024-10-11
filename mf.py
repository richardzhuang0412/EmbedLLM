import argparse
import random
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam
from tqdm import tqdm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
    
def evaluate(net, test_loader, device):
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

# Main training loop
def train(net, train_loader, test_loader, num_epochs, lr, device, weight_decay=1e-5, save_path=None):
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

        test_loss, test_accuracy = evaluate(net, test_loader, device)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
        progress_bar.update(1)
    
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def load_model(net, path, device):
    net.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=232)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_data_path", type=str, default="data/train.csv")
    parser.add_argument("--test_data_path", type=str, default="data/test.csv")
    parser.add_argument("--question_embedding_path", type=str, default="data/question_embeddings.pth")
    parser.add_argument("--embedding_save_path", type=str, default="data/model_embeddings.pth")
    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--model_load_path", type=str, default=None)
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
    train(model, train_loader, test_loader, num_epochs=args.num_epochs, lr=args.learning_rate,
          device=device, save_path=args.model_save_path)
    # torch.save(model.P.weight.detach().to("cpu"), args.embedding_save_path) # Save model embeddings if needed