from model import DeepHistone
import copy
import numpy as np
from utils import metrics,model_train,model_eval,model_predict
import torch
import os
from datetime import datetime
import random
from sklearn.model_selection import KFold

SEED = 42  
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#setting 
batchsize=20
data_file = 'data/E005_deephistone_chr22.npz'

dataset_name = os.path.splitext(os.path.basename(data_file))[0]

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/{dataset_name}_{run_id}"
os.makedirs(output_dir, exist_ok=True)

model_save_file = f'{output_dir}/model.txt'
lab_save_file = f'{output_dir}/label.txt'
pred_save_file = f'{output_dir}/pred.txt'

print('Begin loading data...')
with np.load(data_file) as f:
    indexs = f['keys']
    dna_dict = dict(zip(f['keys'],f['dna']))
    dns_dict = dict(zip(f['keys'],f['dnase']))
    lab_dict = dict(zip(f['keys'],f['label']))
np.random.shuffle(indexs)
idx_len = len(indexs)
train_index=indexs[:int(idx_len*3/5)]
valid_index=indexs[int(idx_len*3/5):int(idx_len*4/5)]
test_index=indexs[int(idx_len*4/5):]


use_gpu = torch.cuda.is_available()
model = DeepHistone(use_gpu)

train_loss_history = []
valid_loss_history = []
valid_auPRC_history = []
valid_auROC_history = []
lr_history = []

print('Begin training model...')
# torch.save(model.forward_fn.state_dict(), 'best_model.pth')
# print("made deep copy of model")
best_valid_auPRC=0
best_valid_loss = np.float64('inf')
for epoch in range(50):
    np.random.shuffle(train_index)
    device = "cuda" if use_gpu else "cpu"
    print("Using device:", device)
    train_loss= model_train(train_index,model,batchsize,dna_dict,dns_dict,lab_dict,)
    valid_loss,valid_lab,valid_pred= model_eval(valid_index, model,batchsize,dna_dict,dns_dict,lab_dict,)
    valid_auPRC,valid_auROC= metrics(valid_lab,valid_pred,'Valid',valid_loss)

    mean_auPRC = np.mean(list(valid_auPRC.values()))
    mean_auROC = np.mean(list(valid_auROC.values()))

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
      f"valid_auPRC={mean_auPRC:.4f}, valid_auROC={mean_auROC:.4f}")

    # save best model based on auPRC
    if mean_auPRC > best_valid_auPRC:
        torch.save(model.forward_fn.state_dict(), f"{output_dir}/best_model.pth")
        best_valid_auPRC = mean_auPRC  

    # Save metrics
    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)
    valid_auPRC_history.append(mean_auPRC)
    valid_auROC_history.append(mean_auROC)
    lr_history.append(model.optimizer.param_groups[0]['lr'])

    # early stopping based on validation loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss  
        early_stop_time = 0
    else:
        model.updateLR(0.1)
        early_stop_time += 1
        if early_stop_time >= 5:
            print("Early stopping triggered.")
            break


print('Begin predicting...')
# load best weights
model.forward_fn.load_state_dict(torch.load(f"{output_dir}/best_model.pth"))

# run prediction
test_lab, test_pred = model_predict(test_index, model, batchsize, dna_dict, dns_dict, lab_dict)

test_auPR,test_roc= metrics(test_lab,test_pred,'Test')


print('Begin saving...')
np.savetxt(lab_save_file, test_lab, fmt='%d', delimiter='\t')
np.savetxt(pred_save_file, test_pred, fmt='%.4f', delimiter='\t')
model.save_model(model_save_file)

history = {
    'train_loss': train_loss_history,
    'valid_loss': valid_loss_history,
    'valid_auPRC': valid_auPRC_history,
    'valid_auROC': valid_auROC_history,
    'learning_rate': lr_history,
    'test_auPRC': test_auPR,
    'test_auROC': test_roc,
    'test_predictions': test_pred,
    'test_labels': test_lab,
    'per_histone_metrics': {
        'auPRC': test_auPR,
        'auROC': test_roc
    }
}
np.savez(f"{output_dir}/training_history.npz", **history)


print('Finished.')
