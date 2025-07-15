from mini_enformer import DeepHistoneEnformer
import copy
import numpy as np
from utils_enformer_1kb import (metrics, model_train, model_eval, model_predict)
import torch
import os
from datetime import datetime
import random
import argparse
import sys
import json

def parse_args():
    parser = argparse.ArgumentParser(description="DeepHistone-Enformer Training Script")
    
    # data arguments
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the .npz data file (e.g., data/E005_deephistone_chr22.npz)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # model arguments
    parser.add_argument('--channels', type=int, default=1536,
                        help='Number of channels in the model (default: 1536)')
    parser.add_argument('--num_transformer_layers', type=int, default=11,
                        help='Number of transformer layers (default: 11)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (default: 0.4)')
    parser.add_argument('--pooling_type', type=str, default='attention',
                        choices=['attention', 'max'],
                        help='Pooling type (default: attention)')
    
    # training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs (default: 50)')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    
    
    return parser.parse_args()

def setup_logging(output_dir: str):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, 'training.log')
    
    
    import logging
    logger = logging.getLogger('enformer_training')
    logger.setLevel(logging.INFO)
    
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
   
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
   
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_data(data_file: str):
    print(f"Loading data from {data_file}...")
    
    try:
        with np.load(data_file, mmap_mode='r') as f:
            print(f"Data shapes:")
            print(f"  Keys: {f['keys'].shape}")
            print(f"  DNA: {f['dna'].shape}")
            print(f"  DNase: {f['dnase'].shape}")
            print(f"  Labels: {f['label'].shape}")
            sys.stdout.flush()
            
            # load data into memory
            indexs = f['keys'][:]
            print(f"Creating dictionaries for {len(indexs)} samples...")
            print("WARNING: Loading entire dataset into memory. This may take a while for large datasets...")
            sys.stdout.flush()
            
            
            dna_dict = dict(zip(f['keys'], f['dna']))
            print("DNA dictionary created")
            sys.stdout.flush()
            
            dns_dict = dict(zip(f['keys'], f['dnase']))
            print("DNase dictionary created")
            sys.stdout.flush()
            
            lab_dict = dict(zip(f['keys'], f['label']))
            print("Label dictionary created")
            sys.stdout.flush()
            
            print(f"Successfully loaded {len(indexs)} samples")
            sys.stdout.flush()
            
            return indexs, dna_dict, dns_dict, lab_dict
            
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_folds(indexs: np.ndarray, n_folds: int = 5):
    np.random.shuffle(indexs)
    fold_size = len(indexs) // n_folds
    folds = [indexs[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    

    if len(indexs) % n_folds != 0:
        folds[-1] = np.concatenate([folds[-1], indexs[n_folds*fold_size:]])
    
    return folds

def train_fold(fold_idx: int, 
               folds: list, 
               model_config: dict,
               training_config: dict,
               data_dicts: dict,
               output_dir: str,
               logger):

    
    fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx+1}')
    os.makedirs(fold_output_dir, exist_ok=True)
    
    logger.info(f"Starting fold {fold_idx+1}/5")
    
    # create train/validation/test splits
    test_index = folds[fold_idx]
    train_valid_folds = [folds[i] for i in range(5) if i != fold_idx]
    train_valid_index = np.concatenate(train_valid_folds)
    np.random.shuffle(train_valid_index)
    
    # 80% train, 20% validation
    train_size = int(len(train_valid_index) * 0.8)
    train_index = train_valid_index[:train_size]
    valid_index = train_valid_index[train_size:]
    
    logger.info(f"  Train samples: {len(train_index)}")
    logger.info(f"  Validation samples: {len(valid_index)}")
    logger.info(f"  Test samples: {len(test_index)}")
    sys.stdout.flush()
    
    # initialize model
    model = DeepHistoneEnformer(
        use_gpu=torch.cuda.is_available(),
        learning_rate=training_config['learning_rate'],
        **model_config
    )
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")  # ADD THIS
    sys.stdout.flush() 
    
    
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_auPRC': [],
        'valid_auROC': [],
        'learning_rate': []
    }
    
    best_valid_auPRC = 0
    best_valid_loss = float('inf')
    early_stop_count = 0
    best_model_path = os.path.join(fold_output_dir, 'best_model.pth')
    
    # training loop
    for epoch in range(training_config['max_epochs']):
        epoch_start = datetime.now()
        np.random.shuffle(train_index)
        
        logger.info(f"  Epoch {epoch+1}/{training_config['max_epochs']}")
        sys.stdout.flush() 

        # training
        train_loss = model_train(
            train_index, model, training_config['batch_size'],
            data_dicts['dna'], data_dicts['dns'], data_dicts['lab']
        )
        logger.info(f"    Training completed. Loss: {train_loss:.4f}")
        logger.info(f"    Validating on {len(valid_index)} samples...")
        sys.stdout.flush() 
        
        # validation
        valid_loss, valid_lab, valid_pred = model_eval(
            valid_index, model, training_config['batch_size'],
            data_dicts['dna'], data_dicts['dns'], data_dicts['lab']
        )
        
        valid_auPRC, valid_auROC = metrics(valid_lab, valid_pred, 'Valid', valid_loss)
        
        mean_auPRC = np.mean(list(valid_auPRC.values()))
        mean_auROC = np.mean(list(valid_auROC.values()))
        
        # save history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_auPRC'].append(mean_auPRC)
        history['valid_auROC'].append(mean_auROC)
        history['learning_rate'].append(model.optimizer.param_groups[0]['lr'])
        
        # save best model
        if mean_auPRC > best_valid_auPRC:
            torch.save(model.forward_fn.state_dict(), best_model_path)
            best_valid_auPRC = mean_auPRC
            logger.info(f"    New best model saved! auPRC: {best_valid_auPRC:.4f}")
        
        # early stopping logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_count = 0
        else:
            model.updateLR(0.1)  # reduce learning rate
            early_stop_count += 1
            logger.info(f"    Early stopping counter: {early_stop_count}/{training_config['early_stopping_patience']}")
            
            if early_stop_count >= training_config['early_stopping_patience']:
                logger.info("    Early stopping triggered")
                sys.stdout.flush()
                break
        
        epoch_time = datetime.now() - epoch_start
        logger.info(f"    Epoch completed in {epoch_time}")
        logger.info(f"    train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")
        logger.info(f"    valid_auPRC={mean_auPRC:.4f}, valid_auROC={mean_auROC:.4f}")
        sys.stdout.flush()
    
    # load best model for testing
    if os.path.exists(best_model_path):
        model.forward_fn.load_state_dict(torch.load(best_model_path))
        logger.info("Best model loaded for testing")
    
    # test evaluation
    logger.info("Evaluating on test set...")
    sys.stdout.flush()
    test_lab, test_pred = model_predict(
        test_index, model, training_config['batch_size'],
        data_dicts['dna'], data_dicts['dns'], data_dicts['lab']
    )
    
    test_auPRC, test_auROC = metrics(test_lab, test_pred, 'Test')
    
    # save fold results
    fold_results = {
        'history': history,
        'test_auPRC': test_auPRC,
        'test_auROC': test_auROC,
        'test_predictions': test_pred,
        'test_labels': test_lab,
        'best_valid_auPRC': best_valid_auPRC,
        'model_config': model_config,
        'training_config': training_config
    }
    
    # save results
    results_file = os.path.join(fold_output_dir, 'training_history.npz')
    np.savez(results_file, **fold_results)
    
    # save predictions and labels as text files
    np.savetxt(os.path.join(fold_output_dir, 'test_labels.txt'), test_lab, fmt='%d', delimiter='\t')
    np.savetxt(os.path.join(fold_output_dir, 'test_predictions.txt'), test_pred, fmt='%.4f', delimiter='\t')
    
    logger.info(f"Fold {fold_idx+1} completed successfully")
    sys.stdout.flush()
    
    return fold_results

def aggregate_results(output_dir: str, logger):
    logger.info("Aggregating results from all folds...")
    
    fold_results = []
    all_auPRCs = []
    all_auROCs = []
    
    for fold_idx in range(5):
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx+1}')
        results_file = os.path.join(fold_dir, 'training_history.npz')
        
        if os.path.exists(results_file):
            try:
                results = np.load(results_file, allow_pickle=True)
                fold_results.append(results)
                
                # calculate mean metrics for this fold
                test_auPRC = results['test_auPRC'].item()
                test_auROC = results['test_auROC'].item()
                
                mean_auPRC = np.mean(list(test_auPRC.values()))
                mean_auROC = np.mean(list(test_auROC.values()))
                
                all_auPRCs.append(mean_auPRC)
                all_auROCs.append(mean_auROC)
                
                logger.info(f"Fold {fold_idx+1}: auPRC={mean_auPRC:.4f}, auROC={mean_auROC:.4f}")
                
            except Exception as e:
                logger.error(f"Error loading results for fold {fold_idx+1}: {e}")
    
    if all_auPRCs and all_auROCs:
        # calculate overall statistics
        final_results = {
            'mean_auPRC': np.mean(all_auPRCs),
            'std_auPRC': np.std(all_auPRCs),
            'mean_auROC': np.mean(all_auROCs),
            'std_auROC': np.std(all_auROCs),
            'fold_auPRCs': all_auPRCs,
            'fold_auROCs': all_auROCs
        }
        
        # calculate per-histone statistics
        from utils_enformer_1kb import histones
        per_histone_results = {}
        
        for histone in histones:
            histone_auPRCs = []
            histone_auROCs = []
            
            for results in fold_results:
                test_auPRC = results['test_auPRC'].item()
                test_auROC = results['test_auROC'].item()
                
                if histone in test_auPRC:
                    histone_auPRCs.append(test_auPRC[histone])
                if histone in test_auROC:
                    histone_auROCs.append(test_auROC[histone])
            
            per_histone_results[histone] = {
                'mean_auPRC': np.mean(histone_auPRCs) if histone_auPRCs else 0,
                'std_auPRC': np.std(histone_auPRCs) if histone_auPRCs else 0,
                'mean_auROC': np.mean(histone_auROCs) if histone_auROCs else 0,
                'std_auROC': np.std(histone_auROCs) if histone_auROCs else 0
            }
        
        final_results['per_histone'] = per_histone_results
        
        # save aggregated results
        final_results_file = os.path.join(output_dir, 'final_results.json')
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # print final results
        logger.info("=" * 60)
        logger.info("5-FOLD CROSS-VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Mean auPRC: {final_results['mean_auPRC']:.4f} ± {final_results['std_auPRC']:.4f}")
        logger.info(f"Mean auROC: {final_results['mean_auROC']:.4f} ± {final_results['std_auROC']:.4f}")
        
        logger.info("\nPer-histone results:")
        for histone in histones:
            if histone in per_histone_results:
                h_results = per_histone_results[histone]
                logger.info(f"  {histone}: auPRC={h_results['mean_auPRC']:.4f}±{h_results['std_auPRC']:.4f}, "
                           f"auROC={h_results['mean_auROC']:.4f}±{h_results['std_auROC']:.4f}")
        
        return final_results
    
    else:
        logger.error("No valid results found")
        return None


def main():
    args = parse_args()
    
    # setup
    print("=" * 60)
    print("DEEPHISTONE-ENFORMER TRAINING STARTED")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    sys.stdout.flush()
    
    
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # create output directory
    dataset_name = os.path.splitext(os.path.basename(args.data_file))[0]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{dataset_name}_enformer_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # setup logging
    logger = setup_logging(output_dir)
    
    # save configuration
    config = {
        'model_config': {
            'channels': args.channels,
            'num_transformer_layers': args.num_transformer_layers,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'pooling_type': args.pooling_type
        },
        'training_config': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_epochs': args.max_epochs,
            'early_stopping_patience': args.early_stopping_patience
        },
        'data_file': args.data_file,
        'seed': args.seed,
    }
    
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_file}")
    logger.info(f"Results will be saved to {output_dir}")
    sys.stdout.flush()
    
    # load data
    start_time = datetime.now()
    indexs, dna_dict, dns_dict, lab_dict = load_data(args.data_file)
    data_dicts = {'dna': dna_dict, 'dns': dns_dict, 'lab': lab_dict}
    
    load_time = datetime.now() - start_time
    logger.info(f"Data loading completed in {load_time}")
    sys.stdout.flush()
    
    # create folds
    folds = create_folds(indexs)
    logger.info(f"Created 5 folds with sizes: {[len(fold) for fold in folds]}")
    sys.stdout.flush()
    
    # 5-fold cross-validation
    all_fold_results = []
    
    for fold_idx in range(5):
        fold_results = train_fold(
            fold_idx, folds, 
            config['model_config'], config['training_config'],
            data_dicts, output_dir, logger
        )
        all_fold_results.append(fold_results)
        sys.stdout.flush()
    
    # aggregate results
    final_results = aggregate_results(output_dir, logger)
    
    
    total_time = datetime.now() - start_time
    logger.info(f"Training completed successfully in {total_time}")
    logger.info(f"All results saved to {output_dir}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()