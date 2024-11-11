import os

labels_dir = '../datasets/VOC/labels/'
excluded_classes = {1, 6}  # Exclude ID based off training voc.yaml label for this instance boat and traing
# <class_id> <x_center> <y_center> <width> <height> josh can start using opencv to alter training set image and height to normalize data more.
#joeseph when retrainng the model python train.py --data voc.yaml --cfg fastyolo.yaml --weights pretrained.pt(use other pt file as weights --epochs 50 | tee training_log.txt
#implement tesnorbaord for training process

def filter_annotations():
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            file_path = os.path.join(labels_dir, label_file)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Filter out lines with excluded classes
            filtered_lines = [line for line in lines if int(line.split()[0]) not in excluded_classes]

            # Overwrite the file with filtered annotations
            with open(file_path, 'w') as f:
                f.writelines(filtered_lines)

    print("Annotation filtering complete.")

if __name__ == "__main__":
    filter_annotations()