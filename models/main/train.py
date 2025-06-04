#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import os
import json
import argparse

from main_depression_model import MultimodalDepressionNet
from dataset import InterviewDataset, collate_interviews 

try:
    from text_augmenter_daic_synthetic import augment_texts_for_fold_data
    from audio_augmenter_daic_synthetic import augment_audio_features_for_fold_data
    SYNTHETIC_AUG_AVAILABLE = True
except ImportError:
    print("Warning: Synthetic data augmentation scripts not found. Augmentation will be skipped.")
    SYNTHETIC_AUG_AVAILABLE = False

def calculate_metrics(predictions, labels, task_type='binary_classification'):
    if task_type == 'binary_classification':
        preds_binary = (predictions > 0.5).int()
        accuracy = (preds_binary == labels.int()).float().mean().item()
        print(f"  Dummy Metrics: Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy, "f1": accuracy} 
    elif task_type == 'regression':
        mse = F.mse_loss(predictions, labels).item()
        rmse = np.sqrt(mse)
        mae = F.l1_loss(predictions, labels).item()
        print(f"  Dummy Metrics: RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return {"rmse": rmse, "mae": mae}
    return {}

# Main training and evaluation functions
def train_epoch(model, dataloader, optimizer, criterion_main, criterion_adv, lambda_adv, lambda_grl, device, task_type):
    model.train()
    total_loss_main = 0
    total_loss_adv = 0
    total_loss_combined = 0
    
    all_main_preds = []
    all_main_labels = []

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        q_text_feat = batch['q_text_features'].to(device)
        a_text_feat = batch['a_text_features'].to(device)
        a_audio_raw = batch['a_audio_raw_features'].to(device)
        turn_indices = batch['turn_indices'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        iqf_labels_true = batch['iqf_labels'].to(device)
        
        if task_type == 'binary_classification':
            main_labels_true = batch['binary_labels'].to(device)
        elif task_type == 'regression':
            main_labels_true = batch['regression_scores'].to(device)
        else:
            raise NotImplementedError("Multiclass labels not handled in this example train_epoch")

        optimizer.zero_grad()
        
        # Forward pass
        main_output, iqf_output_sequence = model(
            q_text_feat, a_text_feat, a_audio_raw, 
            turn_indices, attention_mask, 
            lambda_grl=lambda_grl # This lambda is for GRL scaling factor
        )
        
        # Calculate main loss
        loss_main = criterion_main(main_output, main_labels_true)
        
        # Calculate adversarial loss
        active_loss_mask = attention_mask.view(-1) == True
        
        active_iqf_logits = iqf_output_sequence.view(-1, model.config.num_iqf_classes)[active_loss_mask]
        active_iqf_labels = iqf_labels_true.view(-1)[active_loss_mask]
        
        loss_adv = torch.tensor(0.0).to(device) 
        if active_iqf_logits.shape[0] > 0: 
             loss_adv = criterion_adv(active_iqf_logits, active_iqf_labels)
        
        # Total loss
        combined_loss = loss_main - lambda_adv * loss_adv
        
        combined_loss.backward()
        optimizer.step()
        
        total_loss_main += loss_main.item()
        total_loss_adv += loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv
        total_loss_combined += combined_loss.item()

        all_main_preds.append(main_output.detach().cpu())
        all_main_labels.append(main_labels_true.detach().cpu())

        if batch_idx % 5 == 0: 
            print(f"  Batch {batch_idx}/{len(dataloader)}: Main Loss: {loss_main.item():.4f}, Adv Loss: {loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv:.4f}, Combined: {combined_loss.item():.4f}")

    avg_loss_main = total_loss_main / len(dataloader)
    avg_loss_adv = total_loss_adv / len(dataloader)
    avg_loss_combined = total_loss_combined / len(dataloader)
    
    all_main_preds_cat = torch.cat(all_main_preds)
    all_main_labels_cat = torch.cat(all_main_labels)
    epoch_metrics = calculate_metrics(all_main_preds_cat, all_main_labels_cat, task_type)
    
    return avg_loss_main, avg_loss_adv, avg_loss_combined, epoch_metrics


def evaluate_epoch(model, dataloader, criterion_main, criterion_adv, lambda_adv, lambda_grl, device, task_type):
    model.eval()
    total_loss_main = 0
    total_loss_adv = 0
    total_loss_combined = 0
    
    all_main_preds = []
    all_main_labels = []

    with torch.no_grad():
        for batch in dataloader:
            q_text_feat = batch['q_text_features'].to(device)
            a_text_feat = batch['a_text_features'].to(device)
            a_audio_raw = batch['a_audio_raw_features'].to(device)
            turn_indices = batch['turn_indices'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            iqf_labels_true = batch['iqf_labels'].to(device)
            if task_type == 'binary_classification':
                main_labels_true = batch['binary_labels'].to(device)
            elif task_type == 'regression':
                main_labels_true = batch['regression_scores'].to(device)
            else:
                raise NotImplementedError("Multiclass eval not fully implemented here")

            main_output, iqf_output_sequence = model(
                q_text_feat, a_text_feat, a_audio_raw, 
                turn_indices, attention_mask, 
                lambda_grl=lambda_grl
            )
            
            loss_main = criterion_main(main_output, main_labels_true)
            
            active_loss_mask = attention_mask.view(-1) == True
            active_iqf_logits = iqf_output_sequence.view(-1, model.config.num_iqf_classes)[active_loss_mask]
            active_iqf_labels = iqf_labels_true.view(-1)[active_loss_mask]
            
            loss_adv = torch.tensor(0.0).to(device)
            if active_iqf_logits.shape[0] > 0:
                loss_adv = criterion_adv(active_iqf_logits, active_iqf_labels)
            
            combined_loss = loss_main - lambda_adv * loss_adv
            
            total_loss_main += loss_main.item()
            total_loss_adv += loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv
            total_loss_combined += combined_loss.item()

            all_main_preds.append(main_output.cpu())
            all_main_labels.append(main_labels_true.cpu())

    avg_loss_main = total_loss_main / len(dataloader)
    avg_loss_adv = total_loss_adv / len(dataloader)
    avg_loss_combined = total_loss_combined / len(dataloader)
    
    all_main_preds_cat = torch.cat(all_main_preds)
    all_main_labels_cat = torch.cat(all_main_labels)
    epoch_metrics = calculate_metrics(all_main_preds_cat, all_main_labels_cat, task_type)
    
    return avg_loss_main, avg_loss_adv, avg_loss_combined, epoch_metrics



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")


    try:
        if args.dataset_name.lower() == 'daic-woz':
            from config_daic_woz import PROCESSED_DATA_DIR as DATASET_PROC_DIR
            from config_daic_woz import PROCESSED_DAIC_WOZ_JSON_WITH_ALL_FEATURES_FILENAME as MANIFEST_FILENAME
        elif args.dataset_name.lower() == 'eatd':
            from config_eatd import PROCESSED_DATA_OUTPUT_DIR as DATASET_PROC_DIR
            from config_eatd import PROCESSED_EATD_JSON_WITH_ALL_FEATURES_FILENAME as MANIFEST_FILENAME
        elif args.dataset_name.lower() == 'androids':
            from config_androids import PROCESSED_DATA_ANDROID_DIR as DATASET_PROC_DIR
            from config_androids import PROCESSED_ANDROID_JSON_WITH_ALL_FEATURES_FILENAME as MANIFEST_FILENAME
        else:
            raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")
        
        full_manifest_path = os.path.join(DATASET_PROC_DIR, MANIFEST_FILENAME)

    except ImportError:
        print(f"Error: Could not import config for {args.dataset_name}. Make sure config_{args.dataset_name.lower()}.py exists.")
        return
    except AttributeError:
        print(f"Error: Necessary paths not found in config for {args.dataset_name}. Ensure JSON filenames are correct.")
        return


    print(f"Attempting to load manifest: {full_manifest_path}")
    if not os.path.exists(full_manifest_path):
        print(f"FATAL: Final manifest {full_manifest_path} not found! Make sure all preprocessing steps, including IQF labeling, are complete.")
        return

    # Create the full dataset instance
    full_dataset = InterviewDataset(
        manifest_file_path=full_manifest_path,
        processed_data_base_dir=DATASET_PROC_DIR,
        max_seq_len=args.model_max_seq_len # Max QA pairs for transformer
    )
    
    # Model config

    model_config_dict = {
        "text_embed_dim": 768, # XLM-R base
        "audio_raw_embed_dim": 1024, # XLSR-53 large pooled
        "audio_proj_dim": 768,
        "fusion_output_dim": 768,
        "dcope_qc_mlp_hidden_dim": args.dcope_mlp_hidden,
        "dcope_max_seq_len": args.model_max_seq_len,
        "dialogue_transformer_layers": args.dt_layers,
        "dialogue_transformer_heads": args.dt_heads,
        "dialogue_transformer_ffn_dim": args.dt_ffn_dim,
        "dialogue_transformer_dropout": args.dropout_transformer,
        "attn_pool_intermediate_dim": args.attn_pool_dim,
        "pred_head_hidden_dim": args.pred_head_hidden,
        "pred_head_dropout": args.dropout_pred_head,
        "num_depression_classes": 1 if args.task_type != 'multiclass_classification' else args.num_classes,
        "prediction_task_type": args.task_type,
        "adv_classifier_hidden_dim": args.adv_hidden,
        "num_iqf_classes": full_dataset.num_iqf_classes,
        "adv_classifier_dropout": args.dropout_adv
    }

    class ModelConfig:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    model_config = ModelConfig(**model_config_dict)

    # K-cold cv
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    dataset_indices = list(range(len(full_dataset)))

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_indices)):
        print(f"\n--- Fold {fold+1}/{args.num_folds} ---")
        

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        current_train_data_for_fold = [train_subset.dataset.interviews[i] for i in train_subset.indices]
        
        if args.dataset_name.lower() == 'daic-woz' and args.use_synthetic_augmentation and SYNTHETIC_AUG_AVAILABLE:
            print(f"  Applying text augmentation for DAIC-WOZ training fold {fold+1}...")
            current_train_data_for_fold = augment_texts_for_fold_data(current_train_data_for_fold)
            
            print(f"  (Conceptual) Audio augmentation would be applied here for DAIC-WOZ training fold {fold+1}.")
            
            class TempDataset(Dataset): #
                def __init__(self, interviews_list, base_dir, max_len, iqf_map, num_iqf_cls):
                    self.interviews = interviews_list
                    self.processed_data_base_dir = base_dir
                    self.max_seq_len = max_len
                    self.iqf_to_idx = iqf_map
                    self.num_iqf_classes = num_iqf_cls
                def __len__(self): return len(self.interviews)

            train_dataset_for_loader = TempDataset(
                current_train_data_for_fold, 
                full_dataset.processed_data_base_dir, 
                full_dataset.max_seq_len,
                full_dataset.iqf_to_idx,
                full_dataset.num_iqf_classes
            )
            print(f"  Length of training dataset for fold {fold+1} after text augmentation: {len(train_dataset_for_loader)}")
        else:
            train_dataset_for_loader = train_subset


        train_loader = DataLoader(train_dataset_for_loader, batch_size=args.batch_size, shuffle=True, collate_fn=collate_interviews)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_interviews)


        model = MultimodalDepressionNet(model_config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Define loss functions
        if args.task_type == 'binary_classification':
            criterion_main = nn.BCEWithLogitsLoss().to(device)
        elif args.task_type == 'regression':
            criterion_main = nn.MSELoss().to(device)
        else: # multiclass
            criterion_main = nn.CrossEntropyLoss().to(device) 
            
        criterion_adv = nn.CrossEntropyLoss(ignore_index=-100 if args.pad_iqf_with_ignore_index else model_config.num_iqf_classes-1).to(device) # Example: ignore padding for IQF if padded with -100
                                                                            

        best_val_metric = float('-inf') if args.task_type != 'regression' else float('inf')
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")
            
            train_loss_main, train_loss_adv, train_loss_comb, train_metrics = train_epoch(
                model, train_loader, optimizer, criterion_main, criterion_adv, 
                args.lambda_adversarial, args.lambda_grl, device, args.task_type
            )
            print(f"  Train: MainLoss={train_loss_main:.4f}, AdvLoss={train_loss_adv:.4f}, Combined={train_loss_comb:.4f}")
            for metric_name, metric_val in train_metrics.items(): print(f"    Train {metric_name}: {metric_val:.4f}")

            val_loss_main, val_loss_adv, val_loss_comb, val_metrics = evaluate_epoch(
                model, val_loader, criterion_main, criterion_adv, 
                args.lambda_adversarial, args.lambda_grl, device, args.task_type
            )
            print(f"  Val:   MainLoss={val_loss_main:.4f}, AdvLoss={val_loss_adv:.4f}, Combined={val_loss_comb:.4f}")
            for metric_name, metric_val in val_metrics.items(): print(f"    Val {metric_name}: {metric_val:.4f}")
            
           
            current_val_metric = val_metrics.get(args.early_stopping_metric,
                                                val_loss_main if args.task_type == 'regression' else 0) 

            improved = False
            if args.task_type == 'regression':
                if current_val_metric < best_val_metric:
                    best_val_metric = current_val_metric
                    improved = True
            else: 
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    improved = True
            
            if improved:
                patience_counter = 0
                print(f"  Validation metric ({args.early_stopping_metric}) improved to {best_val_metric:.4f}. Saving model for fold {fold+1}...")
                os.makedirs(args.model_save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, f"best_model_fold_{fold+1}.pt"))
            else:
                patience_counter += 1
                print(f"  Validation metric did not improve. Patience: {patience_counter}/{args.early_stopping_patience}")

            if patience_counter >= args.early_stopping_patience:
                print(f"  Early stopping triggered for fold {fold+1} at epoch {epoch}.")
                break
        
        fold_results.append({'fold': fold+1, 'best_val_metric': best_val_metric, 'val_metrics_at_best': val_metrics}) 


    print("\n--- Cross-Validation Results ---")
    all_best_metrics = [res['best_val_metric'] for res in fold_results]
    print(f"Average best validation metric ({args.early_stopping_metric}) across {args.num_folds} folds: {np.mean(all_best_metrics):.4f} +/- {np.std(all_best_metrics):.4f}")

    for res in fold_results:
        print(f"  Fold {res['fold']}: Best {args.early_stopping_metric} = {res['best_val_metric']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Multimodal Depression Detection Model")

    parser.add_argument('--dataset_name', type=str, required=True, choices=['daic-woz', 'eatd', 'androids'], help='Name of the dataset to use')
   

    parser.add_argument('--model_max_seq_len', type=int, default=100, help='Max QA pairs for D-CoPE and Transformer input')
    parser.add_argument('--dcope_mlp_hidden', type=int, default=256)
    parser.add_argument('--dt_layers', type=int, default=2)
    parser.add_argument('--dt_heads', type=int, default=8)
    parser.add_argument('--dt_ffn_dim', type=int, default=2048)
    parser.add_argument('--dropout_transformer', type=float, default=0.1)
    parser.add_argument('--attn_pool_dim', type=int, default=256)
    parser.add_argument('--pred_head_hidden', type=int, default=256)
    parser.add_argument('--dropout_pred_head', type=float, default=0.3)
    parser.add_argument('--adv_hidden', type=int, default=128)
    parser.add_argument('--dropout_adv', type=float, default=0.2)
    parser.add_argument('--task_type', type=str, default='binary_classification', choices=['binary_classification', 'regression', 'multiclass_classification'])
    parser.add_argument('--num_classes', type=int, default=1, help="1 for binary/regression, N for multiclass")


    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lambda_adversarial', type=float, default=0.1, help="Weight for adversarial loss term")
    parser.add_argument('--lambda_grl', type=float, default=1.0, help="Scaling factor for GRL gradient reversal")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--use_cuda', type=bool, default=True, help="Use CUDA if available")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help="Directory to save best models")
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--early_stopping_metric', type=str, default='f1', help="Metric for early stopping (e.g., f1, rmse, mae)")
    parser.add_argument('--pad_iqf_with_ignore_index', action='store_true', help="If true, IQF padding uses -100 for CrossEntropyLoss ignore_index")


    # DAIC-Synthetic augmentation specific args
    parser.add_argument('--use_synthetic_augmentation', action='store_true', help="Enable DAIC-Synthetic style augmentation for DAIC-WOZ training folds")


    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(args)

