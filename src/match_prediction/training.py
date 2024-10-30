import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
# Training loop without stock targets
def train_model(model, train_loader,optimizer ,frontier_loss_fn,epochs=10):
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_frontier_loss = 0

        for batch in train_loader:
            # Unpack the data; only bid_ask_prices and frontier_targets are provided
            bid_ask_prices, frontier_targets = batch

            # Forward pass
            frontier_pred = model(bid_ask_prices).float()
            # Frontier classification loss
            frontier_loss = frontier_loss_fn(frontier_pred.view(-1,frontier_pred.size(-1)), frontier_targets.view(-1).long())

            # Total loss (can be weighted if necessary)
            total_loss = frontier_loss  # We focus only on frontier prediction

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_frontier_loss += frontier_loss.item()

        print(f"Epoch {epoch+1}/{epochs},Frontier Loss: {total_frontier_loss}")

# Assuming you have a DataLoader that provides batches of option orders
# train_loader = your_data_loader
# train_model(model, train_loader, epochs=20)



def validate_model(model, val_loader, frontier_loss_fn):
    model.eval()
    total_frontier_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            bid_ask_prices, frontier_targets = batch
            frontier_pred = model(bid_ask_prices).float()
            frontier_loss = frontier_loss_fn(frontier_pred.view(-1, frontier_pred.size(-1)), frontier_targets.view(-1).long())
            total_frontier_loss += frontier_loss.item()
    print(f"Validation Loss: {total_frontier_loss}")

def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in test_loader:
            bid_ask_prices, frontier_targets = batch  # Unpack batch data

            # Forward pass: Get logits from the model
            frontier_pred = model(bid_ask_prices)  # Output shape: [batch_size, seq_len, num_classes]

            # Convert logits to predicted class (0 or 1) using argmax
            predicted = frontier_pred.argmax(dim=-1).cpu().numpy()  # Get the class with the highest logit
            targets = frontier_targets.cpu().numpy()  # Convert targets to numpy for metrics

            # Accumulate predictions and true labels
            all_predictions.append(predicted.flatten())  # Flatten in case of sequence
            all_targets.append(targets.flatten())

    # Convert accumulated lists to numpy arrays
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Compute precision, recall, f1 score


# Compute precision, recall, f1 score
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')


    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    # Print metrics
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# import torch
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# import numpy as np
#
# def train_model(model, train_loader, optimizer, frontier_loss_fn, epochs=10):
#     model.train()  # Set model to training mode
#
#     for epoch in range(epochs):
#         total_frontier_loss = 0
#
#         for batch in train_loader:
#             # Unpack the data; only bid_ask_prices and frontier_targets are provided
#             bid_ask_prices, frontier_targets = batch
#
#             # Forward pass
#             frontier_pred = model(bid_ask_prices)  # Output: [batch_size, seq_len, 1] or [batch_size, 1]
#
#             # Reshape frontier_pred and frontier_targets for BCEWithLogitsLoss
#             frontier_loss = frontier_loss_fn(frontier_pred.view(-1), frontier_targets.view(-1).float())  # Target must be float
#
#             # Total loss (can be weighted if necessary)
#             total_loss = frontier_loss  # We focus only on frontier prediction
#
#             # Backpropagation and optimization
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#             total_frontier_loss += frontier_loss.item()
#
#         print(f"Epoch {epoch+1}/{epochs}, Frontier Loss: {total_frontier_loss}")
#
# def validate_model(model, val_loader, frontier_loss_fn):
#     model.eval()
#     total_frontier_loss = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             bid_ask_prices, frontier_targets = batch
#             frontier_pred = model(bid_ask_prices)  # Output: [batch_size, seq_len, 1] or [batch_size, 1]
#
#             # Reshape frontier_pred and frontier_targets for BCEWithLogitsLoss
#             frontier_loss = frontier_loss_fn(frontier_pred.view(-1), frontier_targets.view(-1).float())  # Target must be float
#             total_frontier_loss += frontier_loss.item()
#
#     print(f"Validation Loss: {total_frontier_loss}")
#
#
# def test_model(model, test_loader):
#     model.eval()  # Set the model to evaluation mode
#     all_targets = []
#     all_predictions = []
#
#     with torch.no_grad():  # Disable gradient calculation for testing
#         for batch in test_loader:
#             bid_ask_prices, frontier_targets = batch  # Unpack batch data
#
#             # Forward pass: Get logits from the model
#             frontier_pred = model(bid_ask_prices)  # Output shape: [batch_size, seq_len, 1] or [batch_size, 1]
#
#             # Apply sigmoid to convert logits to probabilities
#             probs = torch.sigmoid(frontier_pred).cpu().numpy()  # Convert logits to probabilities
#
#             # Convert probabilities to predicted class (0 or 1) using a threshold of 0.5
#             predicted = (probs > 0.5).astype(int)
#
#             # Collect true targets
#             targets = frontier_targets.cpu().numpy()
#
#             # Accumulate predictions and true labels
#             all_predictions.append(predicted.flatten())  # Flatten in case of sequence
#             all_targets.append(targets.flatten())
#
#     # Convert accumulated lists to numpy arrays
#     all_predictions = np.concatenate(all_predictions)
#     all_targets = np.concatenate(all_targets)
#
#     # Compute precision, recall, f1 score
#     precision = precision_score(all_targets, all_predictions, average='binary')
#     recall = recall_score(all_targets, all_predictions, average='binary')
#     f1 = f1_score(all_targets, all_predictions, average='binary')
#
#     # Confusion matrix
#     conf_matrix = confusion_matrix(all_targets, all_predictions)
#
#     # Print metrics
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print("Confusion Matrix:")
#     print(conf_matrix)
#
#
#
# # Modified training loop
# def train_model_with_auxiliary_task(model, train_loader, epochs=10):
#     model.train()  # Set model to training mode
#
#     for epoch in range(epochs):
#         total_stock_loss = 0
#         total_frontier_loss = 0
#
#         for batch in train_loader:
#             bid_ask_prices, frontier_targets = batch
#
#             # Forward pass
#             stock_pred, frontier_pred = model(bid_ask_prices)
#
#             # Compute losses
#             # If no ground truth for stock, ignore stock loss or use proxy task
#             # stock_loss = stock_loss_fn(stock_pred, proxy_stock_targets)  # Uncomment if using stock ground truth
#             stock_loss = stock_loss_fn(stock_pred, torch.zeros_like(stock_pred))  # Assuming no ground truth
#
#             frontier_loss = frontier_loss_fn(frontier_pred.view(-1, frontier_pred.size(-1)), frontier_targets.view(-1).long())
#
#             # Weighted total loss
#             total_loss = stock_loss_weight * stock_loss + frontier_loss_weight * frontier_loss
#
#             # Backpropagation and optimization
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#             total_stock_loss += stock_loss.item()
#             total_frontier_loss += frontier_loss.item()
#
#         print(f"Epoch {epoch+1}/{epochs}, Stock Loss: {total_stock_loss}, Frontier Loss: {total_frontier_loss}")
