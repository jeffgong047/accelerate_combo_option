import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque
import time
from .DNN import DQNWithEmbeddings, BiAttentionClassifier
from utils import profit_with_penalty_reward, profit_minus_liability_reward, get_reward_function

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) if len(self.memory) >= batch_size else self.memory
        
    def __len__(self):
        return len(self.memory)

class MarketDQNAgent:
    def __init__(self, model, hidden_size=64, lr=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 replay_buffer_size=10000, features=None, reward_type='profit_with_penalty',
                 **reward_kwargs):
        """
        DQN Agent for market order matching
        
        Args:
            model: Base model for embeddings extraction 
            hidden_size: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            epsilon_*: Exploration parameters
            replay_buffer_size: Size of replay buffer
            features: Feature names for DataFrame conversion
            reward_type: Type of reward function to use
            **reward_kwargs: Additional parameters for the reward function
        """
        self.device = next(model.parameters()).device
        self.model = model
        self.features = features or ['option1', 'option2', 'C=Call, P=Put', 
                                    'Strike Price of the Option Times 1000',
                                    'B/A_price', 'transaction_type']
        
        # Get reward function
        self.reward_function = get_reward_function(reward_type, **reward_kwargs)
        
        # Initialize action space (flip one bit at a time)
        # For a market with N orders, we have N possible actions
        # Action i means "flip the selection status of order i"
        self.num_actions = 1  # Will be set dynamically based on market size
        
        # Initialize Q-network 
        self.q_network = DQNWithEmbeddings(
            input_size=len(self.features), 
            hidden_size=hidden_size,
            num_actions=self.num_actions,
            base_model=model
        ).to(self.device)
        
        self.target_network = DQNWithEmbeddings(
            input_size=len(self.features), 
            hidden_size=hidden_size,
            num_actions=self.num_actions,
            base_model=model
        ).to(self.device)
        
        # Initialize target network with q-network weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Other parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, market_data, state, reward_fn, eval_mode=False):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            market_data: Market data tensor
            state: Current state (binary vector of selected orders)
            reward_fn: Function to compute reward 
            eval_mode: Whether to evaluate (no exploration)
            
        Returns:
            action: Selected action
            next_state: Next state after taking the action
            reward: Reward for taking the action
        """
        # Dynamically update action space based on market size
        self.num_actions = len(state)
        
        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                # Reshape state to match q_network input
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                # Get q-values for all actions
                q_values = self.q_network(market_data.unsqueeze(0).to(self.device), state_tensor)
                # Select action with highest q-value
                action = q_values.argmax(1).item()
        
        # Create next state by flipping the bit at action index
        next_state = state.copy()
        next_state[action] = 1 - next_state[action]  # Flip 0 to 1 or 1 to 0
        
        # Calculate reward for next state
        reward = self._calculate_reward(market_data, next_state, reward_fn)
        
        return action, next_state, reward
    
    def _calculate_reward(self, market_data, state, reward_fn):
        """
        Calculate reward for a given state using the configured reward function
        
        Args:
            market_data: Market data tensor
            state: Binary vector of selected orders
            reward_fn: Function to compute profit
                
        Returns:
            reward: Calculated reward
        """
        # Convert market data to DataFrame
        df = pd.DataFrame(market_data.detach().cpu().numpy(), columns=self.features)
        
        # Add selection column
        df['selected'] = state
        
        # Filter selected orders
        selected_df = df[df['selected'] == 1]
        
        # If no orders selected, return negative reward
        if len(selected_df) == 0:
            return -1.0
        
        # Separate buy and sell books
        buy_book = selected_df[selected_df['transaction_type'] == 1]
        sell_book = selected_df[selected_df['transaction_type'] == 0]
        
        # If either book is empty, return negative reward
        if len(buy_book) == 0 or len(sell_book) == 0:
            return -1.0
        
        # Add liquidity column if not present (default to 1)
        if 'liquidity' not in buy_book.columns:
            buy_book['liquidity'] = 1.0
        if 'liquidity' not in sell_book.columns:
            sell_book['liquidity'] = 1.0
        
        # Calculate reward using the configured reward function
        try:
            # Try with run_matching_with_timeout if available
            from match_prediction.training import run_matching_with_timeout
            reward = self.reward_function(buy_book, sell_book, 
                                         lambda b, s, **kwargs: run_matching_with_timeout(reward_fn, b, s, **kwargs), 
                                         full_df=df)
        except ImportError:
            # Fall back to direct call
            reward = self.reward_function(buy_book, sell_book, reward_fn, full_df=df)
        
        return reward
    
    def update(self, batch_size):
        """
        Update Q-network using a batch from replay buffer
        
        Args:
            batch_size: Batch size for update
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn_for_market_matching(model, train_loader, reward_fn, 
                                  num_episodes=100, batch_size=32, 
                                  target_update=10, features=None,
                                  reward_type='profit_with_penalty', **kwargs):
    """
    Train DQN agent for market order matching
    
    Args:
        model: Base model for embeddings
        train_loader: DataLoader for training data
        reward_fn: Function to compute reward
        num_episodes: Number of episodes to train
        batch_size: Batch size for Q-network update
        target_update: Frequency of target network update
        features: Feature names for DataFrame conversion
        reward_type: Type of reward function to use
        **kwargs: Additional arguments for DQN agent
    
    Returns:
        agent: Trained DQN agent
    """
    # Get device from model
    device = next(model.parameters()).device
    
    # Initialize DQN agent
    agent = MarketDQNAgent(
        model=model,
        features=features,
        reward_type=reward_type,
        **kwargs
    )
    
    # Lists to store metrics
    episode_rewards = []
    avg_losses = []
    
    # Train for num_episodes
    for episode in range(num_episodes):
        episode_start_time = time.time()
        total_reward = 0
        losses = []
        
        # Process each batch in train_loader
        for batch_idx, (market_data, _) in enumerate(train_loader):
            if isinstance(market_data, list):
                market_data = market_data[0]
            
            market_data = market_data.to(device)
            
            # Process each market in batch
            for i in range(market_data.size(0)):
                # Get current market
                current_market = market_data[i]
                
                # Initialize state (all orders unselected)
                state = np.zeros(current_market.size(0))
                
                # Use the model to predict initial state 
                # (orders that are likely to be matched)
                with torch.no_grad():
                    logits = model(current_market.unsqueeze(0))
                    probs = torch.softmax(logits, dim=-1)[0, :, 1]
                    predicted_state = (probs > 0.5).float().cpu().numpy()
                    
                    # Use predicted state as initial state
                    state = predicted_state
                
                # Run episode for this market
                for step in range(50):  # Max 50 steps per market
                    # Select action
                    action, next_state, reward = agent.select_action(
                        current_market, state, reward_fn
                    )
                    
                    # Mark episode as done if no change in state
                    done = np.array_equal(state, next_state) or step == 49
                    
                    # Store transition in replay buffer
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    
                    # Update state
                    state = next_state
                    
                    # Accumulate reward
                    total_reward += reward
                    
                    # Update Q-network
                    if len(agent.replay_buffer) >= batch_size:
                        loss = agent.update(batch_size)
                        losses.append(loss)
                    
                    # Break if episode is done
                    if done:
                        break
            
            # Update target network periodically
            if episode % target_update == 0 and batch_idx % target_update == 0:
                agent.update_target_network()
        
        # Compute metrics
        avg_reward = total_reward / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Store metrics
        episode_rewards.append(avg_reward)
        avg_losses.append(avg_loss)
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        
        # Print metrics
        print(f"Episode {episode+1}/{num_episodes}, "
              f"Avg Reward: {avg_reward:.4f}, "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Epsilon: {agent.epsilon:.4f}, "
              f"Time: {episode_time:.2f}s")
    
    return agent

def dqn_evaluate_market(model, agent, test_loader, reward_fn, features=None):
    """
    Evaluate DQN agent on test data
    
    Args:
        model: Base model for embeddings
        agent: Trained DQN agent
        test_loader: DataLoader for test data
        reward_fn: Function to compute reward
        features: Feature names for DataFrame conversion
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    device = next(model.parameters()).device
    features = features or ['option1', 'option2', 'C=Call, P=Put', 
                           'Strike Price of the Option Times 1000',
                           'B/A_price', 'transaction_type']
    
    total_profit = 0
    total_matches = 0
    total_markets = 0
    total_selected_orders = 0
    total_orders = 0
    
    # Set to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # For list-based test_loader
        if not isinstance(test_loader, torch.utils.data.DataLoader):
            for market_data, _ in test_loader:
                # Process each market
                if isinstance(market_data, np.ndarray):
                    current_market = torch.tensor(market_data, dtype=torch.float32).to(device)
                else:
                    current_market = market_data.to(device)
                
                # Initialize state (all orders unselected)
                state = np.zeros(current_market.size(0))
                
                # Use model's prediction as initial state
                logits = model(current_market.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)[0, :, 1]
                predicted_state = (probs > 0.5).float().cpu().numpy()
                state = predicted_state
                
                # Run DQN to find optimal state
                for step in range(100):  # Max 100 steps per market
                    # Select action
                    action, next_state, reward = agent.select_action(
                        current_market, state, reward_fn, eval_mode=True
                    )
                    
                    # Mark episode as done if no change in state
                    done = np.array_equal(state, next_state) or step == 99
                    
                    # Update state
                    state = next_state
                    
                    # Break if episode is done
                    if done:
                        break
                
                # Convert market data to DataFrame
                df = pd.DataFrame(current_market.cpu().numpy(), columns=features)
                
                # Add selection column
                df['selected'] = state
                
                # Filter selected orders
                selected_df = df[df['selected'] == 1]
                
                # Separate buy and sell books
                buy_book = selected_df[selected_df['transaction_type'] == 1]
                sell_book = selected_df[selected_df['transaction_type'] == 0]
                
                # Run matching algorithm
                try:
                    _, _, profit, isMatch, _ = reward_fn(buy_book, sell_book)
                    
                    # Update metrics
                    total_profit += profit
                    total_matches += 1 if isMatch else 0
                    total_markets += 1
                    total_selected_orders += len(selected_df)
                    total_orders += len(df)
                    
                except Exception as e:
                    print(f"Error in evaluation: {e}")
        
        # For DataLoader-based test_loader
        else:
            for batch_idx, (market_data, _) in enumerate(test_loader):
                if isinstance(market_data, list):
                    market_data = market_data[0]
                
                market_data = market_data.to(device)
                
                # Process each market in batch
                for i in range(market_data.size(0)):
                    # Get current market
                    current_market = market_data[i]
                    
                    # Initialize state (all orders unselected)
                    state = np.zeros(current_market.size(0))
                    
                    # Use model's prediction as initial state
                    logits = model(current_market.unsqueeze(0))
                    probs = torch.softmax(logits, dim=-1)[0, :, 1]
                    predicted_state = (probs > 0.5).float().cpu().numpy()
                    state = predicted_state
                    
                    # Run DQN to find optimal state
                    for step in range(100):  # Max 100 steps per market
                        # Select action
                        action, next_state, reward = agent.select_action(
                            current_market, state, reward_fn, eval_mode=True
                        )
                        
                        # Mark episode as done if no change in state
                        done = np.array_equal(state, next_state) or step == 99
                        
                        # Update state
                        state = next_state
                        
                        # Break if episode is done
                        if done:
                            break
                    
                    # Convert market data to DataFrame
                    df = pd.DataFrame(current_market.cpu().numpy(), columns=features)
                    
                    # Add selection column
                    df['selected'] = state
                    
                    # Filter selected orders
                    selected_df = df[df['selected'] == 1]
                    
                    # Separate buy and sell books
                    buy_book = selected_df[selected_df['transaction_type'] == 1]
                    sell_book = selected_df[selected_df['transaction_type'] == 0]
                    
                    # Run matching algorithm
                    try:
                        _, _, profit, isMatch, _ = reward_fn(buy_book, sell_book)
                        
                        # Update metrics
                        total_profit += profit
                        total_matches += 1 if isMatch else 0
                        total_markets += 1
                        total_selected_orders += len(selected_df)
                        total_orders += len(df)
                        
                    except Exception as e:
                        print(f"Error in evaluation: {e}")
    
    # Compute final metrics
    avg_profit = total_profit / total_markets if total_markets > 0 else 0
    match_rate = total_matches / total_markets if total_markets > 0 else 0
    selection_ratio = total_selected_orders / total_orders if total_orders > 0 else 0
    
    metrics = {
        'avg_profit': avg_profit,
        'match_rate': match_rate,
        'selection_ratio': selection_ratio,
        'total_profit': total_profit,
        'total_matches': total_matches,
        'total_markets': total_markets,
    }
    
    # Print metrics
    print(f"Evaluation Results:")
    print(f"Avg Profit: {avg_profit:.4f}")
    print(f"Match Rate: {match_rate:.4f}")
    print(f"Selection Ratio: {selection_ratio:.4f}")
    
    return metrics 