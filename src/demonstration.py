import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from match_prediction import BiAttentionClassifier,train_model, validate_model, test_model
import numpy as np
import pickle
from torch.nn.utils.rnn import pad_sequence
from utils import collate_fn
from tqdm import tqdm
import seaborn as sns
from lime import lime_tabular
from captum.attr import IntegratedGradients
import shap
import os 
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.decomposition import PCA

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Option Order DNN with custom hyperparameters")
    parser.add_argument('--input_size', type=int, default=6, help='Number of input features (e.g. bid/ask prices)')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test set size as a percentage of the entire data')
    return parser.parse_args()




def pca_analysis(X,y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y)
    fig.update_layout(
        title="PCA visualization of Custom Classification dataset",
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component",
    )
    fig.show()

def attention_analysis(data_loader,model,sanity_check):
    # investigate interpretability of attention mechanism 
    attention_proportions = []
    attention_stock_price_ratios = []
    attention_mechanism_distribution = []
    for i, batch in enumerate(tqdm(data_loader)):
        if i >= 3:  # Only process first 3 batches
            break
        sample_checking_raw, sample_target_raw  = batch
        if sanity_check:
            sample_checking = torch.cat([sample_checking_raw, sample_checking_raw], dim=1)
            sample_target = torch.cat([sample_target_raw, sample_target_raw], dim=1)
        else:
            sample_checking = sample_checking_raw
            sample_target = sample_target_raw

        with torch.no_grad():
            attention_weights = model.check_attention_score(sample_checking).squeeze(0)

        sorted_attention_weights , sorted_index = torch.sort(attention_weights, descending=True)
        # given the sorted_index, we want to test on Hypothesis:
        # All instances whether, the highest attention, 95% of attention is located on two options which
        #stock price estimation falls on two extreme end of collective stock price estimation.

        average_attention_options = torch.sum(attention_weights, dim = 0)/attention_weights.shape[0]
        average_attention_option_sorted, average_attention_options_sorted_index = torch.sort(average_attention_options)
        attention0_option_idx, attention1_option_idx = average_attention_options_sorted_index[0], average_attention_options_sorted_index[1]
        attention_proportion =  (average_attention_option_sorted[0] + average_attention_option_sorted[1])/ torch.sum(average_attention_options)
        attention_proportions.append(attention_proportion)
        sample_target = sample_target.unsqueeze(0)
        sample_target = sample_target.expand(-1,sample_target.shape[-1], -1)
        sample_target_squeezed = sample_target.squeeze(0)  # Shape: [114, 114]
        sorted_index_squeezed = sorted_index.squeeze(0)    # Shape: [114, 114]
        
        # Use torch.gather to reorder sample_target_squeezed based on sorted_index along the last dimension
        sample_target_sorted = sample_target_squeezed.gather(1, sorted_index_squeezed)
        
        #sanity check.. top 50 percent attention weight  versus bottom 50 percent attention weight on the target
        top_50_percent_labels = torch.sum(sample_target_sorted[:,:sample_target_sorted.shape[-1]//2])
        bottom_50_percent_labels = torch.sum(sample_target_sorted[:, sample_target_sorted.shape[-1]//2:])
        attention_distribution_sample = [top_50_percent_labels, bottom_50_percent_labels]
        print(top_50_percent_labels/(top_50_percent_labels+bottom_50_percent_labels),top_50_percent_labels+bottom_50_percent_labels)
        
        attention_mechanism_distribution.append(attention_distribution_sample)
        # Convert tensors to numpy for plotting
        sorted_attention_weights_np = sorted_attention_weights.cpu().numpy().squeeze(0)
        sorted_sample_target_np = sample_target_sorted.cpu().numpy()
        # Prepare annotations with symbols for label=1
        # calculate the estimated stock price

        stock_price_estimation = ((sample_checking[0,:,2] + sample_checking[0,:,3])/2*sample_checking[0,:,0] + sample_checking[0,:,1])
        stock_price_estimation_range = torch.max(stock_price_estimation) - torch.min(stock_price_estimation)
        attention_stock_price_range = torch.abs(stock_price_estimation[attention0_option_idx] - stock_price_estimation[attention1_option_idx])
        attention_stock_price_ratio = attention_stock_price_range/stock_price_estimation_range
        attention_stock_price_ratios.append(attention_stock_price_ratio)
        average_stock_estimation = torch.sum(stock_price_estimation)/sample_checking.shape[1]
        abs_diff_stock_price_estimation = torch.abs(stock_price_estimation - average_stock_estimation)
        abs_diff_sorted , option_index_abs_diff_sorted = torch.sort(abs_diff_stock_price_estimation)
        print(option_index_abs_diff_sorted)
        print(attention_weights.shape)
        attention_weights_np = attention_weights.cpu().numpy()
        sample_target_np  = np.tile(sample_target.cpu().numpy(),(attention_weights_np.shape[0],1))

        print(attention_weights_np.shape)
        print(sample_target_np.shape)
        annotations = np.empty_like( sample_target_np, dtype=str)
        annotations[sample_target_np == 1] = '*'  # Mark label=1 with '*'
        annotations[sample_target_np == 0] = ''   # Leave label=0 cells blank

        # Plot the heatmap with conditional annotations
        if sanity_check:
            plt.figure(figsize=(16, 12))
        else:
            plt.figure(figsize=(8,6))
        ax = sns.heatmap(np.log1p(attention_weights_np), annot=annotations, fmt="",
                            cmap="viridis", cbar=True, square=True, annot_kws={"size": 10})
        
        # Set plot title and labels
        halfway = attention_weights.shape[1] // 2
        if sanity_check:
            plt.axvline(x=halfway, color='red', linestyle='-', linewidth=5)
        # ax.set_title(f"Attention Weights with Target Labels: high attention on frontier percentile (top 50 vs bottom 50) = {top_50_percent_labels/(top_50_percent_labels+bottom_50_percent_labels)}")
        ax.set_xlabel("option index")
        ax.set_ylabel("option index")
        if sanity_check:
            plt.savefig('sanity_check_attention_map')
        else:
            plt.savefig(f'attention_map')
        breakpoint()
    print(attention_proportions)
    print(attention_stock_price_ratio)
    breakpoint()
    attention_mechanism_distribution_np = np.log1p(np.array(attention_mechanism_distribution))
    np.savetxt('attention_mechanism_distribution.txt',attention_mechanism_distribution_np)
    top_50_sum = np.sum(attention_mechanism_distribution_np[:, 0])
    total_sum = np.sum(attention_mechanism_distribution_np)
    proportion_top_50 = top_50_sum / total_sum
    print("Proportion of top 50% labels to total labels:", proportion_top_50)


def t_sne_analysis(X, y):
    n_samples = X.shape[0]
    max_perplexity = min(50, n_samples - 1)  # Cap maximum perplexity
    
    # Generate perplexity range based on data size
    perplexity_range = np.arange(5, max_perplexity, 5)
    divergence = []
    
    # Calculate KL divergence for different perplexity values
    for p in perplexity_range:
        model = TSNE(n_components=3, init="pca", perplexity=p, random_state=42)
        _ = model.fit_transform(X)
        divergence.append(model.kl_divergence_)
    
    # Find optimal perplexity (elbow point)
    # Using simple method: find point where increasing perplexity gives diminishing returns
    divergence_diff = np.diff(divergence)
    optimal_idx = np.argmin(np.abs(divergence_diff - np.mean(divergence_diff))) + 1
    optimal_perplexity = perplexity_range[optimal_idx]
    
    # Plot perplexity vs divergence
    fig_perp = px.line(x=perplexity_range, y=divergence, markers=True)
    fig_perp.update_layout(
        title="Perplexity vs KL Divergence",
        xaxis_title="Perplexity Values", 
        yaxis_title="Divergence"
    )
    fig_perp.update_traces(line_color="red", line_width=1)
    fig_perp.show()
    
    # Perform t-SNE with optimal perplexity
    tsne = TSNE(n_components=3, 
                random_state=42, 
                perplexity=optimal_perplexity,
                init="pca")
    X_tsne = tsne.fit_transform(X)
    
    # Plot t-SNE results
    fig_tsne = px.scatter_3d(x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2], color=y)
    fig_tsne.update_layout(
        title=f"t-SNE visualization (perplexity={optimal_perplexity:.1f})",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
    )      
    fig_tsne.show()
    
    return X_tsne


def explain_predictions_integrated_gradients(model, input_data):
    model.eval()
    
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.FloatTensor(input_data)
    
    # Keep original attention analysis
    attention_scores = model.check_attention_score(input_data)
    
    # Add Integrated Gradients analysis
    ig = IntegratedGradients(model)
    
    # Get model prediction for each position
    with torch.no_grad():
        output = model(input_data)  # [batch_size, seq_len, num_classes]
        predicted_classes = output.argmax(dim=-1)  # [batch_size, seq_len]
    
    # Initialize storage for attributions at each position
    all_attributions = []
    all_errors = []
    
    # For each position in the sequence
    for pos in range(predicted_classes.size(1)):
        # Create a wrapper function that returns only the output for the positive class (class 1)
        def wrapper_fn(inputs):
            return model(inputs)[:, pos, 1]  # Return only the score for class 1 at this position
        
        # Create a new IG instance with the wrapper
        ig_pos = IntegratedGradients(wrapper_fn)
        
        # Get attributions for this position
        attr, error = ig_pos.attribute(
            input_data,
            n_steps=50,
            return_convergence_delta=True
        )
        
        all_attributions.append(attr)
        all_errors.append(error)
    
    # Stack all position-wise attributions
    combined_attributions = torch.stack(all_attributions, dim=1)  # [batch_size, seq_len, input_size]
    combined_errors = torch.stack(all_errors, dim=1)  # [batch_size, seq_len]
    
    return {
        'attention_scores': attention_scores,
        'ig_attributions': combined_attributions,
        'ig_error': combined_errors
    }

def explain_predictions_lime(data_loader, model):
    # Get all data from data_loader
    all_data = []
    for batch in data_loader:
        sample_checking, _ = batch
        all_data.append(sample_checking.squeeze(0))
    
    training_data = torch.cat(all_data, dim=0).numpy()
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=['Option Type', 'Strike Price', 'Bid Price', 'Ask Price'],
        class_names=['Not in Frontier', 'In Frontier'],
        mode='classification'
    )
    
    # Function to get model predictions
    def predict_fn(x):
        x_tensor = torch.FloatTensor(x).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(x_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
        return probs.numpy()
    
    # Get explanations for each instance
    explanations = []
    for data, _ in data_loader:
        data = data.squeeze(0).numpy()
        exp = explainer.explain_instance(data, predict_fn)
        explanations.append(exp)
    
    return explanations

def analyze_and_visualize_integrated_gradients(model, data_loader, feature_names,visualization_path = '.'):
    model.eval()
    all_attributions = []
    # Process first 3 batches to get their attributions
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Computing IG")):        
        # inputs, targets, dominated_by = batch
        # dominated_by = dominated_by[0]
        dominated_by = None 
        inputs, targets = batch 
        current_seq_len = inputs.size(1)
        # Get attributions for each position in the sequence
        batch_attributions = []
        for pos in range(current_seq_len):
            def wrapper_fn(x):
                output = model(x)
                return output[:, pos, 1]
            
            ig_pos = IntegratedGradients(wrapper_fn)
            attributions = ig_pos.attribute(
                inputs,
                n_steps=50,
            )
            batch_attributions.append(attributions)
        
        # Combine attributions across positions
        combined_attributions = torch.stack(batch_attributions).mean(dim=0)
        all_attributions.append(combined_attributions.detach().cpu().numpy())
        stacked_attributions = torch.stack(batch_attributions)
        print(stacked_attributions.shape)
        # Average across features to get option-level attribution
        option_attributions = stacked_attributions.mean(dim=-1)  # [seq_len_output, batch, seq_len_input]
        
        # Create target annotations with matching shape
        option_attributions_np = option_attributions.detach().cpu().numpy()
        targets_np = targets.cpu().numpy()
        # Reshape annotations to match option_attributions_np shape
        # Broadcast target indicators across all output positions using numpy tile
        target_broadcast = np.tile(targets_np.squeeze(), (option_attributions_np.squeeze(1).shape[0], 1))
        annotations = np.empty_like(option_attributions_np.squeeze(1), dtype=str)
        annotations[target_broadcast == 1] = '*'  # Mark frontier options with *
        annotations[target_broadcast == 0] = ''   # Leave non-frontier options blank
        # Visualize option-level attributions
        plt.figure(figsize=(15, 10))
        
        # First plot the heatmap without annotations
        data = option_attributions_np.squeeze(1)
        transformed_x = inputs.squeeze(0).detach().cpu().numpy()
        transformed_y = np.sum(data,axis=0)
        print(transformed_x.shape,transformed_y.shape)
        breakpoint()
        t_sne_analysis(transformed_x,transformed_y)
        pca_analysis(transformed_x,transformed_y)
        ax = sns.heatmap(data,
                        cmap='RdBu_r',
                        center=0)
        
        # Add annotations with circles and x's
        for i in range(annotations.shape[0]):
            for j in range(annotations.shape[1]):
                # For frontier options (hollow circle)
                if annotations[i, j] == '*':
                    # Draw hollow circle for frontier options
                    circle = plt.Circle((j + 0.5, i + 0.5), 0.3,
                                         fill=False,  # Makes the circle hollow
                                         color='black',
                                         clip_on=True,
                                         zorder=2)
                    ax.add_artist(circle)

                # For dominated options (x mark)
                if dominated_by is not None and j in dominated_by[i]:
                    # Add x mark inside the existing circle
                    ax.text(j + 0.5, i + 0.5, '×',  # Using × symbol for better visibility
                            ha='center', va='center',
                            color='red',
                            fontsize=8,  # Increased font size for better visibility
                            fontweight='bold',
                            zorder=3)  # Ensure x appears on top of circle

        plt.title(f'Batch {batch_idx} Option Attribution Heatmap\nRows: Output positions, Columns: Input options')
        plt.xlabel('Input Option Position')
        plt.ylabel('Output Position')
        plt.tight_layout()
        # plt.savefig(f'{visualization_path}/comb_2_batch_{batch_idx}_option_attributions.png',
        #             dpi=300, bbox_inches='tight')
        plt.savefig(f'combo_{len(feature_names)-4}_{batch_idx}.png')
        plt.close()

        # For each target position, find top 2 contributing options
        abs_attributions = np.abs(data)  # [num_targets, num_options]
        top_options = []
        feature_names = ['Option Type', 'Strike Price', 'Highest Bid', 'Lowest Ask']
        for target_idx in range(abs_attributions.shape[0]):
            target_attributions = abs_attributions[target_idx]
            top_2_indices = np.argsort(target_attributions)[-2:]
            top_2_indices = np.array(sorted(top_2_indices))  # Ensure positive strides
            top_options_features = inputs[0, top_2_indices].detach().cpu().numpy()  # [2, feature_dim]
            top_options.append({
                'target_idx': target_idx,
                'top_indices': top_2_indices,
                'top_features': top_options_features,
                'feature_names': feature_names,
                'attribution_values': target_attributions[top_2_indices]
            })
# ecking
     
        breakpoint()


    # Calculate raw (signed) feature importance
    feature_importance = []
    for batch_attribution in all_attributions:
        # Remove np.abs() to keep sign information
        batch_importance = np.mean(batch_attribution, axis=(0, 1))
        feature_importance.append(batch_importance)
    
    # Average feature importance across batches
    avg_feature_importance = np.mean(feature_importance, axis=0)
    
    # Plotting with positive/negative values
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(avg_feature_importance)), avg_feature_importance)
    
    # Color bars based on positive/negative values
    for bar, value in zip(bars, avg_feature_importance):
        if value < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
            
    plt.title('Global Feature Importance (Integrated Gradients)', fontsize=14)
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Average Attribution (with sign)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # Add zero line
    plt.tight_layout()
    plt.savefig('feature_importance_ig.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print top features (by absolute importance)
    top_n = 10
    abs_importance = np.abs(avg_feature_importance)
    top_indices = np.argsort(abs_importance)[-top_n:][::-1]
    print("\nTop {} most important features (with direction):".format(top_n))
    for idx in top_indices:
        print(f"Feature {idx}: {avg_feature_importance[idx]:.4f}")
    
    return {'feature_importance': avg_feature_importance}

def explain_with_shap(model, data_loader, feature_names):
    model.eval()
    
    # Get first batch and pad all sequences to the same length
    first_batch = next(iter(data_loader))[0]
    max_seq_len = first_batch.size(1)
    
    # Create a wrapper model that returns scalar output
    class ModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, x):
            # Get model output and return average probability for positive class
            output = self.base_model(x)  # [batch_size, seq_len, num_classes]
            probs = torch.softmax(output, dim=-1)  # Convert to probabilities
            return probs[:, :, 1].mean()  # Return mean probability for positive class
    
    wrapped_model = ModelWrapper(model)
    
    # Create background dataset with padded sequences
    background_data = first_batch[:100]
    
    # Create SHAP explainer with wrapped model
    explainer = shap.DeepExplainer(wrapped_model, background_data)
    
    all_shap_values = []
    all_data = []
    
    # Iterate through all batches
    for i, batch in enumerate(tqdm(data_loader)):
        if i >= 3:  # Only process first 3 batches
            break
        batch_data, _ = batch
        
        # Pad or truncate sequence to match background data size
        if batch_data.size(1) < max_seq_len:
            # Pad with zeros
            padded_data = torch.zeros(batch_data.size(0), max_seq_len, batch_data.size(2))
            padded_data[:, :batch_data.size(1), :] = batch_data
            batch_data = padded_data
        elif batch_data.size(1) > max_seq_len:
            # Truncate
            batch_data = batch_data[:, :max_seq_len, :]
        
        # Calculate SHAP values for this batch
        shap_values = explainer.shap_values(batch_data)
        all_shap_values.append(shap_values)
        all_data.append(batch_data)
    
    # Combine all SHAP values
    combined_shap_values = np.concatenate(all_shap_values, axis=1)
    combined_data = torch.cat(all_data, dim=0)
    
    # Calculate average feature importance
    avg_shap_values = np.mean(np.abs(combined_shap_values), axis=0)
    feature_importance = np.mean(avg_shap_values, axis=0)
    
    # # Plot feature importance
    # plt.figure(figsize=(10, 6))
    # plt.bar(feature_names, feature_importance)
    # plt.title('Average Feature Importance (SHAP)')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('shap_feature_importance.png')
    
    # # Create SHAP summary plot
    # shap.summary_plot(combined_shap_values, combined_data, feature_names=feature_names)
    # plt.tight_layout()
    # plt.savefig('shap_summary_plot.png')
    
    return {
        'shap_values': combined_shap_values,
        'feature_importance': feature_importance
    }

def main():
    args = parse_arguments()
    sanity_check = False
    # Assuming your data is loaded and preprocessed here
    # X = bid_ask_prices_features, y = frontier_targets
    # Split the data into training, validation, and test sets
    # data = pd.read_csv('training_data.csv')
    # with open('./../data/training_data_frontier_bid_ask.pkl','rb') as f:
    #     data = pickle.load(f)



    directory_path = '/common/home/hg343/Research/accelerate_combo_option/data/combo_2_frontier'
    model_path = f'frontier_option_classifier_combo2_no_matched.pt'
    # file_path = '/common/home/hg343/Research/accelerate_combo_option/data/single_frontier_labels_corrected_dominant_labels.pkl' 
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)

    data = [] 
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if the file is a pickle file
        if filename.endswith('.pkl'):  # Assuming your files are pickle files
            try:
                with open(file_path, 'rb') as f:
                    if file_path.endswith('no_matched.pkl'):
                        data_point = pickle.load(f)
                    # Check if data_point has the expected structure
                    if len(data_point) > 0:
                        df = pd.DataFrame(data_point, columns=['option1', 'option2','C=Call, P=Put', 
                                                                   'Strike Price of the Option Times 1000', 
                                                                   'transaction_type', 'B/A_price', 
                                                                   'belongs_to_frontier'])
                        data.append(df)
                    else:
                        print(f"Warning: {file_path} does not contain valid data.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    feature_names = [
        'option1', 
         'option2', 
        'C=Call, P=Put',
        'Strike Price of the Option Times 1000',
        'B/A_price',
        'transaction_type'
    ]
    X = [x[['option1', 'option2', 'C=Call, P=Put',
            'Strike Price of the Option Times 1000',
            'B/A_price',
            'transaction_type']].to_numpy(dtype = np.float32) for x in data]
    # dominated_by = [x['dominated_by'].tolist() for x in data]
    y = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in data]
    
    # # Verify dominating options are in frontier
    # for i, (dominated_list, frontier_labels) in enumerate(zip(dominated_by, y)):
    #     for option_idx, dominating_indices in enumerate(dominated_list):
    #         if len(dominating_indices) > 0:  # If this option is dominated by others
    #             # Check if all dominating options are in frontier
    #             all_dominators_in_frontier = all(frontier_labels[idx] == 1 for idx in dominating_indices)
    #             if not all_dominators_in_frontier:
    #                 print(f"Warning in sample {i}: Option {option_idx} is dominated by options {dominating_indices}")
    #                 print(f"Frontier labels for dominators: {[frontier_labels[idx] for idx in dominating_indices]}")

    attention_mechanism_distribution = []
# lets use t-sne to analyze one single example first
# we could implement get embedding before the attention; 
# we need attention scores; attribution scores for each embedding which we denotes annotation


    #lets check whether all the dominating options are in the frontier 

    #load model
    # data_loader = DataLoader(list(zip(X, y,dominated_by)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader = DataLoader(list(zip(X, y)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = BiAttentionClassifier(input_size=args.input_size, hidden_size=args.hidden_size,  num_classes=2, bidirectional=True)
    model_state_dict = torch.load(model_path)
    #lets first try to evaluate the loaded model performance
    # test_model(model,data_loader)
    # breakpoint()
    model.load_state_dict(model_state_dict)
    model.eval()
    # t_sne_analysis(X[1],y[1])
    # for i in range(len(X)):
    #     embedding = model.get_embedding(torch.tensor(X[i]).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    # #  t_sne_analysis(embedding,y[1])
    #     attention_scores = model.check_attention_score(torch.tensor(X[i]).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    #     print(embedding.shape, attention_scores.shape)
    #     t_sne_analysis(embedding,attention_scores.sum(axis=0))
    #     breakpoint()
    attention_proportions = []
    attention_stock_price_ratios =[]

    visualization_path = '/common/home/hg343/Research/accelerate_combo_option/visualizations'
    # Use the existing analysis function
    analyze_and_visualize_integrated_gradients(
        model=model,
        data_loader=data_loader,
        feature_names=feature_names, 
        visualization_path=visualization_path
    )
    
    # Comment out SHAP analysis and other visualizations
    # shap_results = explain_with_shap(model, data_loader, feature_names)










if __name__ == '__main__':
    main()












    # # Add LIME analysis
    # lime_explanations = explain_predictions_lime(data_loader, model)
    
    # # Analyze LIME results
    # feature_importance = {
    #     'Option Type': [],
    #     'Strike Price': [],
    #     'Bid Price': [],
    #     'Ask Price': []
    # }
    
    # for exp in lime_explanations:
    #     # Get feature weights for positive class (frontier)
    #     explanation = dict(exp.as_list(label=1))
    #     for feature in feature_importance.keys():
    #         if feature in explanation:
    #             feature_importance[feature].append(abs(explanation[feature]))
    
    # # Plot average feature importance
    # plt.figure(figsize=(10, 6))
    # avg_importance = {k: np.mean(v) for k, v in feature_importance.items()}
    # plt.bar(avg_importance.keys(), avg_importance.values())
    # plt.title('Average Feature Importance (LIME)')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('lime_feature_importance.png')