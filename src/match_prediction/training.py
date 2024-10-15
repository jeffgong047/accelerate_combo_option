import torch
# Training loop without stock targets
def train_model(model, train_loader,optimizer ,frontier_loss_fn,epochs=10):
    model.train()  # Set model to training mode
    
    for epoch in range(epochs):
        total_stock_loss = 0
        total_frontier_loss = 0
        
        for batch in train_loader:
            # Unpack the data; only bid_ask_prices and frontier_targets are provided
            bid_ask_prices, frontier_targets = batch
            
            # Forward pass
            stock_pred, frontier_pred = model(bid_ask_prices)
            
            # Compute losses
            # No stock_targets, so stock_loss can be set to zero or ignored
            stock_loss = torch.tensor(0.0)  # Stock loss is zero, as no ground truth is provided
            
            # Frontier classification loss
            frontier_loss = frontier_loss_fn(frontier_pred.squeeze(-1), frontier_targets)
            
            # Total loss (can be weighted if necessary)
            total_loss = frontier_loss  # We focus only on frontier prediction
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_stock_loss += stock_loss.item()
            total_frontier_loss += frontier_loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Stock Loss: {total_stock_loss}, Frontier Loss: {total_frontier_loss}")

# Assuming you have a DataLoader that provides batches of option orders
# train_loader = your_data_loader
# train_model(model, train_loader, epochs=20)



def validate_model(model, val_loader, frontier_loss_fn):
    model.eval()
    total_frontier_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            bid_ask_prices, frontier_targets = batch
            stock_pred, frontier_pred = model(bid_ask_prices)
            frontier_loss = frontier_loss_fn(frontier_pred.squeeze(-1), frontier_targets)
            total_frontier_loss += frontier_loss.item()
    print(f"Validation Loss: {total_frontier_loss}")

# Define the testing function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            bid_ask_prices, frontier_targets = batch
            stock_pred, frontier_pred = model(bid_ask_prices)
            predicted = (frontier_pred > 0.5).float()  # Convert predictions to binary
            correct += (predicted.squeeze(-1) == frontier_targets).sum().item()
            total += frontier_targets.size(0)
            print(correct/total)
            breakpoint()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


        
# Modified training loop
def train_model_with_auxiliary_task(model, train_loader, epochs=10):
    model.train()  # Set model to training mode
    
    for epoch in range(epochs):
        total_stock_loss = 0
        total_frontier_loss = 0
        
        for batch in train_loader:
            bid_ask_prices, frontier_targets = batch
            
            # Forward pass
            stock_pred, frontier_pred = model(bid_ask_prices)
            
            # Compute losses
            # If no ground truth for stock, ignore stock loss or use proxy task
            # stock_loss = stock_loss_fn(stock_pred, proxy_stock_targets)  # Uncomment if using stock ground truth
            stock_loss = stock_loss_fn(stock_pred, torch.zeros_like(stock_pred))  # Assuming no ground truth
            
            frontier_loss = frontier_loss_fn(frontier_pred, frontier_targets)
            
            # Weighted total loss
            total_loss = stock_loss_weight * stock_loss + frontier_loss_weight * frontier_loss
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_stock_loss += stock_loss.item()
            total_frontier_loss += frontier_loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Stock Loss: {total_stock_loss}, Frontier Loss: {total_frontier_loss}")
