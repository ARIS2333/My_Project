import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import shutil


def show_acc_loss_test(history, subject=0):
    '''
    This function is used to plot the <accuracy> and <loss> of the model during the test
    Args:
    history: dictionary of the accuracy and loss of the model during the training and testing
    subject: the subject number, start from 0, will be added 1 during the plotting
    '''
    plt.figure(figsize=(12, 5))
    # big title
    plt.suptitle(f'Subject {subject+1}', fontsize=20)

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='train_acc', color='blue')
    plt.plot(history['valid_acc'], label='test_acc', color='red')
    plt.legend(fontsize=12)
    plt.title('Accuracy', fontsize=15)
    plt.ylim(0.3, 1)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='train_loss', color='blue')
    plt.plot(history['valid_loss'], label='test_loss', color='red')
    plt.legend(fontsize=12)
    plt.title('Loss', fontsize=15)

    plt.show()
    

def evaluation_test(config_train, checkpoint, model, test_dataloader,
               display_labels = ['left','right','feet',"tongue"], subject=0):
    '''
    This function is used to evaluate the model on the test set, and plot the <confusion matrix> and print the <evaluation metrics>

    '''

    model.load_state_dict(checkpoint['model'])
    y_pred, y_true = [], []
    model.eval()
    for data, target in test_dataloader:
        data = data.to(config_train['device'])
        target = target.to(config_train['device'])
        output = model(data)
        y_pred.extend(output.argmax(dim=1).cpu().detach().numpy())
        y_true.extend(target.cpu().detach().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = display_labels)
    disp.plot(cmap='Blues')

    # acc
    acc = np.trace(cm) / np.sum(cm)
    # precision
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    # recall
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    # kappa
    total = np.sum(cm)
    p0 = np.trace(cm) / total
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / total**2
    kappa = (p0 - pe) / (1 - pe)

    # put the evaluation metrics on to the plot
    plt.text(4.5, 1.5, f'Accuracy: {acc:.3f}', fontsize=12, color='black')
    plt.text(4.5, 2, f'Kappa: {kappa:.3f}', fontsize=12, color='black')
    plt.text(4.5, 2.5, f'Precisions: {[round(precision[i],3) for i in range(len(precision))]}', fontsize=12, color='black')
    plt.text(4.5, 3, f'Recalls: {[round(recall[i],3) for i in range(len(recall))]}', fontsize=12, color='black')
    plt.suptitle(f'Subject {subject+1}', fontsize=20)
    plt.show()

    return acc, kappa, precision, recall


############################################################################################################
class save_results:
    def __init__(self, save_dir='results', trial='1', subject='1'):
        data_path = save_dir + '/trial_' + trial + '/subject_' + subject
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path

    def info_Preprocessing(self, config_processor):
        '''
        This function is used to save the information of the preprocessing steps

        Args:
            config_processor: dict, the configuration of the preprocessing steps
            info_path: str, the path to save the information
        '''
        with open(self.data_path / 'info_preprocessing.csv', 'w') as f:
            for key, value in config_processor.items():
                if not key=='save_path_test' and not key=='save_path_train':
                    # key在第一行， value在第二行
                    f.write("%s,%s\n"%(key,value))

    
    def info_Model(self, config_train):
        '''
        This function is used to save the information of the model configuration

        Args:
            config_train: class, the configuration of the model
            info_path: str, the path to save the information
        '''
        with open(self.data_path / 'info_model.csv', 'w') as f:
            for key, value in config_train.items():
                f.write("%s,%s\n"%(key,value))

    def info_Training(self, history):
        with open(self.data_path / 'info_training.csv', 'w') as f:
            f.write("%s,%s\n"%('Max valid ACC', max(history['valid_acc'])))
            f.write("%s,%s\n"%('Min valid Loss', min(history['valid_loss'])))
            f.write("%s,%s\n"%('Best Epoch for ACC', np.argmax(history['valid_acc'])+1))
            f.write("%s,%s\n"%('Best Epoch for Loss', np.argmin(history['valid_loss'])+1))

    
    def info_model_optimizer(self, checkpoint):
        # the default setting for checkpoint is None
        if checkpoint is not None:
            torch.save(checkpoint, self.data_path / 'checkpoint.pth')
        else:
            print('No checkpoint is saved')
            # save a empty file
            with open(self.data_path / 'checkpoint_NotFound.pth', 'w') as f:
                pass

    def show_acc_loss(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='train_acc', color='blue')
        plt.plot(history['valid_acc'], label='test_acc', color='red')
        plt.legend(fontsize=12)
        plt.title('Accuracy', fontsize=15)
        plt.ylim(0.3, 1)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='train_loss', color='blue')
        plt.plot(history['valid_loss'], label='test_loss', color='red')
        plt.legend(fontsize=12)
        plt.title('Loss', fontsize=15)

        plt.savefig(self.data_path / 'acc_loss.png', bbox_inches='tight')
        plt.close()
    
    def evaluation(self, config_train, checkpoint, model, test_dataloader,
               display_labels = ['left','right','feet',"tongue"]):
        '''
        This function is used to evaluate the model on the test set, and plot the <confusion matrix> and print the <evaluation metrics>
        
        '''
        if checkpoint is None:
            with open(self.data_path / 'Confusion_Matrix_checkpoint_NotFound.png', 'wb') as f:
                pass

        else:
            model.load_state_dict(checkpoint['model'])
            y_pred, y_true = [], []
            model.eval()
            for data, target in test_dataloader:
                data = data.to(config_train['device'])
                target = target.to(config_train['device'])
                output = model(data)
                y_pred.extend(output.argmax(dim=1).cpu().detach().numpy())
                y_true.extend(target.cpu().detach().numpy())
            
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = display_labels)
            disp.plot(cmap='Blues')

            # acc
            acc = np.trace(cm) / np.sum(cm)
            # precision
            precision = np.diag(cm) / np.sum(cm, axis = 0)
            # recall
            recall = np.diag(cm) / np.sum(cm, axis = 1)
            # kappa
            total = np.sum(cm)
            p0 = np.trace(cm) / total
            pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / total**2
            kappa = (p0 - pe) / (1 - pe)

            # put the evaluation metrics on to the plot
            plt.text(4.5, 1.5, f'Accuracy: {acc:.3f}', fontsize=12, color='black')
            plt.text(4.5, 2, f'Kappa: {kappa:.3f}', fontsize=12, color='black')
            plt.text(4.5, 2.5, f'Precisions: {[round(precision[i],3) for i in range(len(precision))]}', fontsize=12, color='black')
            plt.text(4.5, 3, f'Recalls: {[round(recall[i],3) for i in range(len(recall))]}', fontsize=12, color='black')
            plt.savefig(self.data_path / 'confusion_matrix.png', bbox_inches='tight')
            plt.close()

        return acc, kappa, precision, recall
        

    def write_TensorBoard(self, history):
        # remove the previous logs
        try:
            shutil.rmtree(self.data_path / 'TensorBoard_logs')
        except:
            pass

        # create log file
        writer = SummaryWriter(self.data_path / 'TensorBoard_logs')
        # Log the metrics to TensorBoard
        for epoch, (train_loss, train_acc, test_loss, test_acc) in enumerate(zip(history['train_loss'], history['train_acc'], history['valid_loss'], history['valid_acc'])):
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)

        writer.close()




def summary_results(acc_all, kappa_all, precision_all, recall_all, save_dir='results', trial='1'):
    data_path = save_dir + '/trial_' + trial
    data_path = Path(data_path)

    with open(data_path / 'summary_results.csv', 'w') as f:
        f.write('Subject,ACC_mean,K_mean,P_mean,R_mean,ACC_std,K_std,P_std,R_std\n')
        f.write(f'ALL,{np.mean(acc_all):.2f},{np.mean(kappa_all):.2f},{np.mean(precision_all):.2f},{np.mean(recall_all):.2f},{np.std(acc_all):.2f},{np.std(kappa_all):.2f},{np.std(precision_all):.2f},{np.std(recall_all):.2f}\n')
        for i in range(len(acc_all)):
            f.write(f'{i+1},{acc_all[i]:.2f},{kappa_all[i]:.2f},{np.mean(precision_all[i]):.2f},{np.mean(recall_all[i]):.2f},{np.std(acc_all[i]):.2f},{np.std(kappa_all[i]):.2f},{np.std(precision_all[i]):.2f},{np.std(recall_all[i]):.2f}\n')

       
        

            

    
