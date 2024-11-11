from ultralytics import YOLO



import torch
import yaml
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def calculate_metrics(outputs, targets, threshold=0.5):
    TP = FP = FN = TN = 0
    outputs = torch.sigmoid(outputs)  # Apply sigmoid if necessary

    for i in range(len(targets)):
        if outputs[i] >= threshold and targets[i] == 1:
            TP += 1
        elif outputs[i] >= threshold and targets[i] == 0:
            FP += 1
        elif outputs[i] < threshold and targets[i] == 1:
            FN += 1
        elif outputs[i] < threshold and targets[i] == 0:
            TN += 1

    return TP, FP, FN, TN

def plot_metrics(epoch, tp_list, fp_list, fn_list, tn_list):
    plt.figure(figsize=(10, 6))
    plt.plot(tp_list, label='True Positives', color='g')
    plt.plot(fp_list, label='False Positives', color='r')
    plt.plot(fn_list, label='False Negatives', color='orange')
    plt.plot(tn_list, label='True Negatives', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.title('Metrics Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

def train(model, dataloader, optimizer, criterion, scheduler, config, device):
    writer = SummaryWriter(log_dir=config.get('log_dir', './logs'))
    model.to(device)
    model.train()
    total_epochs = config['epochs']
    best_loss = float('inf')

    tp_list, fp_list, fn_list, tn_list = [], [], [], []

    for epoch in range(total_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        TP, FP, FN, TN = 0, 0, 0, 0

        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")):
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Scheduler step (optional)
            if scheduler is not None:
                scheduler.step()

            # Track running loss
            running_loss += loss.item()

            # Calculate metrics
            tp, fp, fn, tn = calculate_metrics(outputs, targets)
            TP += tp
            FP += fp
            FN += fn
            TN += tn

            # Log training statistics
            writer.add_scalar('Loss/Batch', loss.item(), epoch * len(dataloader) + batch_idx)

        # Epoch statistics
        avg_loss = running_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start

        print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {avg_loss:.4f}, TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, Duration: {epoch_duration:.2f}s")

        writer.add_scalar('Loss/Epoch', avg_loss, epoch)
        writer.add_scalar('Epoch Duration', epoch_duration, epoch)
        writer.add_scalar('Metrics/True Positives', TP, epoch)
        writer.add_scalar('Metrics/False Positives', FP, epoch)
        writer.add_scalar('Metrics/False Negatives', FN, epoch)
        writer.add_scalar('Metrics/True Negatives', TN, epoch)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, best_loss, path=config.get('best_model_path', 'best_model.pth'))

        # Update metrics lists
        tp_list.append(TP)
        fp_list.append(FP)
        fn_list.append(FN)
        tn_list.append(TN)

    # Plot metrics after training
    plot_metrics(total_epochs, tp_list, fp_list, fn_list, tn_list)

    writer.close()
    print("Training complete.")

# if __name__ == "__main__":
#     # Load configuration
#     config_path = 'config.yaml'
#     config = load_yaml_config(config_path)

#     # Device configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Model, dataloader, optimizer, and loss function
#     model = torch.load(config['model_path']).to(device)
#     dataloader = torch.utils.data.DataLoader(...)  # Customize this as needed
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#     criterion = torch.nn.BCEWithLogitsLoss()  # Suitable for binary classification

#     # Load checkpoint if fine-tuning
#     if config.get('fine_tune', False):
#         model, optimizer = load_checkpoint(model, optimizer, config['checkpoint_path'])

#     # Train the model
#     train(model, dataloader, optimizer, criterion, scheduler, config, device)
#    Load a model
    # model = YOLO("yolo11n.pt")  # change pretrained model
    # # Train the model with MPS
    # results = model.train(data="data.yaml", epochs=100, imgsz=640, device="mps")

    # print("Class indices with average precision:", results.ap_class_index)
    # print("Average precision for all classes:", results.box.all_ap)
    # print("Average precision:", results.box.ap)
    # print("Average precision at IoU=0.50:", results.box.ap50)
    # print("Class indices for average precision:", results.box.ap_class_index)
    # print("Class-specific results:", results.box.class_result)
    # print("F1 score:", results.box.f1)
    # print("F1 score curve:", results.box.f1_curve)
    # print("Overall fitness score:", results.box.fitness)
    # print("Mean average precision:", results.box.map)
    # print("Mean average precision at IoU=0.50:", results.box.map50)
    # print("Mean average precision at IoU=0.75:", results.box.map75)
    # print("Mean average precision for different IoU thresholds:", results.box.maps)
    # print("Mean results for different metrics:", results.box.mean_results)
    # print("Mean precision:", results.box.mp)
    # print("Mean recall:", results.box.mr)
    # print("Precision:", results.box.p)
    # print("Precision curve:", results.box.p_curve)
    # print("Precision values:", results.box.prec_values)
    # print("Specific precision metrics:", results.box.px)
    # print("Recall:", results.box.r)
    # print("Recall curve:", results.box.r_curve)